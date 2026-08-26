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

# A stage counts as done only if its output is NEWER than every input it depends
# on - the data it was computed from AND the script that computed it. Existence
# alone is not enough: a file left over from an earlier design looks identical to
# a fresh one, and acting on stale results has been the main hazard here. Counting
# the script as an input matters just as much: a table can postdate the data and
# still predate the bug fix that changes what it says.
# Staleness is asymmetric on purpose: an output is stale only when an input is
# STRICTLY newer than it. Testing `out -nt input` instead looks equivalent but is
# not, because shell timestamps have one-second granularity - a figure written in
# the same second as the script that drew it would fail that test and be reported
# stale despite being perfectly current.
# Inputs accumulate down the pipeline. Checking a stage only against its immediate
# input is not enough: figures drawn from tables that are themselves out of date
# would look current, because they postdate those tables. Each stage is therefore
# checked against everything upstream of it, not just the step before.
UP1="data/raw/learning_set_master_data.pkl scripts/01_prepare_data.py"
UP2="$UP1 data/processed/chat_learn.csv data/processed/chat_val.csv \
     data/processed/rounds_learn.csv data/processed/rounds_val.csv \
     scripts/02_extract_features.py"
UP3="$UP2 outputs/features/output/conv/learn_conv_level.csv \
     outputs/features/output/conv/val_conv_level.csv scripts/03_build_analysis_table.py"
UP4="$UP3 data/processed/analysis_learn.csv data/processed/analysis_val.csv \
     scripts/04_analysis.py"
UP5="$UP4 outputs/tables/round_stage.csv outputs/tables/feature_effects.csv \
     outputs/tables/variance_decomposition.csv scripts/05_figures.py"

fresh() {
  local out=$1; shift
  [ -f "$out" ] || return 1
  local input
  for input in "$@"; do
    [ -f "$input" ] || continue
    [ "$input" -nt "$out" ] && return 1
  done
  return 0
}
STALE="\033[31m!\033[0m"
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

mark() {
  case "$1" in
    done)  printf "$DONE" ;;
    run)   printf "$RUN" ;;
    stale) printf "$STALE" ;;
    *)     printf "$TODO" ;;
  esac
}

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
  [ -f data/processed/analysis_val.csv ] && st=stale
  fresh data/processed/analysis_val.csv $UP3 && st=done
  running 03_build && st=run
  printf " %b 3  build analysis table" "$(mark $st)"
  [ "$st" = stale ] && printf "  ${DIM}stale: older than the feature files${OFF}"
  printf "\n"
  [ -f data/processed/analysis_learn.csv ] && \
    printf "      ${DIM}learn %s rows   val %s rows   %s${OFF}\n" \
      "$(rows data/processed/analysis_learn.csv)" \
      "$(rows data/processed/analysis_val.csv)" "$(age data/processed/analysis_val.csv)"

  # ---- 4. analysis ------------------------------------------------------
  st=todo
  [ -f outputs/tables/variance_decomposition.csv ] && st=stale
  fresh outputs/tables/feature_effects.csv $UP4 && st=done
  running 04_analysis && st=run
  printf " %b 4  analysis" "$(mark $st)"
  running 04_analysis && printf "  ${DIM}running %s${OFF}" "$(elapsed 04_analysis)"
  [ "$st" = stale ] && printf "  ${DIM}stale: older than the data or the script${OFF}"
  printf "\n"
  # Reported in the order the analysis writes them, so a run reads top to bottom.
  # The first four are archived analyses; the rest carry the case study's story.
  for t in channel_effect model_comparison variance_decomposition speech_vs_content \
           family_importance family_importance_opening \
           round_stage stage_profile stage_examples stage_feature_effects \
           stage_agreement feature_effects; do
    if fresh outputs/tables/$t.csv $UP4; then
      printf "      ${DONE} ${DIM}%-24s %s${OFF}\n" "$t" "$(age outputs/tables/$t.csv)"
    elif [ -f outputs/tables/$t.csv ]; then
      printf "      ${STALE} ${DIM}%-24s %s  stale${OFF}\n" "$t" "$(age outputs/tables/$t.csv)"
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
  local n fresh_n=0 f newest_table
  n=$(ls outputs/figures/fig[1-3]_*.png 2>/dev/null | wc -l | tr -d ' ')
  # Compare against the NEWEST table, not one chosen table. A figure can easily
  # postdate the single table it was checked against while being drawn from six
  # others that have since been rewritten - which is exactly how this line came
  # to report seven stale figures as current.
  newest_table=$(ls -t outputs/tables/*.csv 2>/dev/null | head -1)
  for f in outputs/figures/fig[1-3]_*.png; do
    fresh "$f" "$newest_table" $UP5 && fresh_n=$((fresh_n + 1))
  done
  [ "$n" -gt 0 ] && st=stale
  [ "$fresh_n" -ge 3 ] && st=done
  running 05_figures && st=run
  printf " %b 5  figures  ${DIM}%s of 3 drawn, %s current${OFF}\n" "$(mark $st)" "$n" "$fresh_n"
}

if [ "$1" = "-w" ] || [ "$1" = "--watch" ]; then
  while true; do clear; snapshot; sleep 10; done
else
  snapshot
fi
