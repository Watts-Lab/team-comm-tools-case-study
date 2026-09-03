#!/usr/bin/env bash
# Progress tracker for the two new analyses (steps 8-12).
#   bash scripts/windows_status.sh        one snapshot
#   bash scripts/windows_status.sh -w     refresh every 20s
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/outputs/logs/run"

# tqdm writes thousands of carriage-return updates into the log; keep the last
# real line and the last progress bar, and drop everything in between.
clean () { tr '\r' '\n' < "$1" 2>/dev/null | grep -v '^\s*$' | tail -n "${2:-1}"; }

age () { [ -f "$1" ] && echo "$(( ($(date +%s) - $(stat -f %m "$1")) / 60 ))m ago" || echo "-"; }

snapshot () {
  printf '\n\033[1m=== new-analysis pipeline  %s ===\033[0m\n' "$(date +%H:%M:%S)"

  printf '\n\033[1mrunning jobs\033[0m\n'
  local any=0
  for job in 09_cumulative_features 11_samples 11_paired 12_first_n chain; do
    pidf="$LOGS/${job%%_*}.pid"
    [ "$job" = chain ] && pidf="$LOGS/chain.pid"
    [ "$job" = 11_paired ] && pidf="$LOGS/11.pid"
    [ "$job" = 11_samples ] && pidf="$LOGS/11.pid"
    pid=$(grep -o '[0-9]\+' "$pidf" 2>/dev/null | head -1)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      cpu=$(ps -p "$pid" -o %cpu=,etime= 2>/dev/null | xargs)
      printf '  \033[32m● %-18s\033[0m pid %-7s cpu/elapsed %s\n' "$job" "$pid" "$cpu"
      any=1
    elif [ -f "$LOGS/$job.log" ]; then
      printf '  \033[90m○ %-18s\033[0m finished or stopped\n' "$job"
    fi
  done
  [ "$any" = 0 ] && printf '  (nothing running)\n'

  printf '\n\033[1mlatest output per job\033[0m\n'
  for job in 09_cumulative_features 11_samples 11_paired 12_first_n chain; do
    [ -f "$LOGS/$job.log" ] || continue
    printf '  \033[36m%s\033[0m\n' "$job"
    clean "$LOGS/$job.log" 3 | sed 's/^/    /' | cut -c1-160
  done

  printf '\n\033[1martifacts\033[0m\n'
  for f in \
    "data/processed/chat_cumulative_learn.csv" \
    "data/processed/chat_cumulative_val.csv" \
    "outputs/features_cumulative/output/conv/learn_conv_level.csv" \
    "outputs/features_cumulative/output/conv/val_conv_level.csv" \
    "data/processed/analysis_windows_learn.csv" \
    "data/processed/analysis_windows_val.csv" ; do
    p="$ROOT/$f"
    if [ -f "$p" ]; then
      printf '  \033[32m✓\033[0m %-58s %8s  %s\n' "$f" \
        "$(du -h "$p" | cut -f1)" "$(age "$p")"
    else
      printf '  \033[90m·\033[0m %-58s %8s\n' "$f" "pending"
    fi
  done

  printf '\n\033[1mresult tables (outputs/tables/windows/)\033[0m\n'
  if compgen -G "$ROOT/outputs/tables/windows/*.csv" > /dev/null; then
    for p in "$ROOT"/outputs/tables/windows/*.csv; do
      printf '  \033[32m✓\033[0m %-40s %6s rows  %s\n' "$(basename "$p")" \
        "$(( $(wc -l < "$p") - 1 ))" "$(age "$p")"
    done
  else
    printf '  \033[90m·\033[0m none yet\n'
  fi
  echo
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; snapshot; sleep 20; done
else
  snapshot
fi
