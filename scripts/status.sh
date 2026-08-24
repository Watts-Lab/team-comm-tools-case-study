#!/usr/bin/env bash
# Pipeline status for the case study.
#
#   bash scripts/status.sh          one snapshot
#   bash scripts/status.sh -w       refresh every 10s until you ctrl-C
#
# Stage state is read from output files, never from process names: a wait-loop
# whose own command line mentions a script would otherwise look like the script.

cd "$(dirname "$0")/.." || exit 1

DONE="\033[32m✔\033[0m"; RUN="\033[33m…\033[0m"; TODO="\033[90m·\033[0m"
DIM="\033[90m"; BOLD="\033[1m"; OFF="\033[0m"

rows() { [ -f "$1" ] && echo "$(($(wc -l < "$1") - 1))" || echo "-"; }
age()  { [ -f "$1" ] && date -r "$1" "+%H:%M" || echo "  -  "; }

# Is a pipeline script actually executing? Match the interpreter path so that
# shells merely waiting on the script do not count.
# pgrep -f alone is not enough: the harness wraps each command in a shell whose
# command line quotes the whole pipeline, so a finished step still "matches".
# Only count a pid whose executable is actually python.
py_pid() {
  local pid
  for pid in $(pgrep -f "scripts/$1" 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in
      *python*) echo "$pid"; return 0 ;;
    esac
  done
  return 1
}
running() { py_pid "$1" >/dev/null 2>&1; }
elapsed() {
  local pid; pid=$(py_pid "$1") || return
  ps -o etime= -p "$pid" | tr -d ' '
}

mark() { if [ "$1" = done ]; then printf "$DONE"; elif [ "$1" = run ]; then printf "$RUN"; else printf "$TODO"; fi; }

snapshot() {
  printf "${BOLD}Team Comm Toolkit case study${OFF}  ${DIM}%s${OFF}\n\n" "$(date '+%a %H:%M:%S')"

  # ---- 1. prepare -------------------------------------------------------
  local st=todo
  [ -f data/processed/rounds_learn.csv ] && [ -f data/processed/rounds_val.csv ] && st=done
  running 01_prepare && st=run
  printf " %b 1  prepare data" "$(mark $st)"
  [ "$st" = run ] && printf "  ${DIM}running %s${OFF}" "$(elapsed 01_prepare)"
  printf "\n"
  if [ -f data/processed/rounds_learn.csv ]; then
    printf "      ${DIM}game-rounds  learn %-6s val %-6s   messages  learn %-6s val %s${OFF}\n" \
      "$(rows data/processed/rounds_learn.csv)" "$(rows data/processed/rounds_val.csv)" \
      "$(rows data/processed/chat_learn.csv)" "$(rows data/processed/chat_val.csv)"
  fi

  # ---- 2. featurize -----------------------------------------------------
  st=todo
  local conv=outputs/features/output/conv
  [ -f $conv/learn_conv_level.csv ] && st=run
  [ -f $conv/val_conv_level.csv ] && st=done
  running 02_extract && st=run
  printf " %b 2  extract toolkit features" "$(mark $st)"
  [ "$st" = run ] && running 02_extract && printf "  ${DIM}running %s${OFF}" "$(elapsed 02_extract)"
  printf "\n"
  for s in learn val; do
    [ -f $conv/${s}_conv_level.csv ] && \
      printf "      ${DIM}%-6s %6s conversations   %s${OFF}\n" "$s" \
        "$(rows $conv/${s}_conv_level.csv)" "$(age $conv/${s}_conv_level.csv)"
  done
  [ -f outputs/featurize.log ] && \
    printf "      ${DIM}%s${OFF}\n" "$(tr '\r' '\n' < outputs/featurize.log | grep -v 'it/s' | tail -1 | cut -c1-70)"

  # ---- 3. analysis table ------------------------------------------------
  st=todo
  [ -f data/processed/analysis_val.csv ] && st=done
  running 03_build && st=run
  printf " %b 3  build analysis table" "$(mark $st)"
  printf "\n"
  [ -f data/processed/analysis_learn.csv ] && \
    printf "      ${DIM}learn %s rows   val %s rows   %s${OFF}\n" \
      "$(rows data/processed/analysis_learn.csv)" \
      "$(rows data/processed/analysis_val.csv)" "$(age data/processed/analysis_val.csv)"

  # ---- 4. analysis ------------------------------------------------------
  st=todo
  [ -f outputs/tables/variance_decomposition.csv ] && st=run
  [ -f outputs/tables/round_stage.csv ] && [ -f outputs/tables/feature_effects.csv ] && st=done
  running 04_analysis && st=run
  printf " %b 4  analysis" "$(mark $st)"
  running 04_analysis && printf "  ${DIM}running %s${OFF}" "$(elapsed 04_analysis)"
  printf "\n"
  for t in channel_effect model_comparison variance_decomposition family_importance \
           round_stage feature_effects; do
    if [ -f outputs/tables/$t.csv ]; then
      printf "      ${DONE} ${DIM}%-24s %s${OFF}\n" "$t" "$(age outputs/tables/$t.csv)"
    else
      printf "      ${TODO} ${DIM}%s${OFF}\n" "$t"
    fi
  done
  if [ -f outputs/analysis_run.log ]; then
    printf "      ${DIM}last: %s${OFF}\n" \
      "$(grep -v 'Warning\|warnings.warn\|cd_fast' outputs/analysis_run.log | grep -v '^$' | tail -1 | cut -c1-70)"
  fi

  # ---- 5. figures -------------------------------------------------------
  st=todo
  local n; n=$(ls outputs/figures/*.png 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] && st=run; [ "$n" -ge 5 ] && st=done
  running 05_figures && st=run
  printf " %b 5  figures  ${DIM}%s of 5${OFF}\n" "$(mark $st)" "$n"
}

if [ "$1" = "-w" ] || [ "$1" = "--watch" ]; then
  while true; do clear; snapshot; sleep 10; done
else
  snapshot
fi
