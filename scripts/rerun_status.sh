#!/usr/bin/env bash
# Progress for the single-indicator refit (step 11, launched by hand).
#   bash scripts/rerun_status.sh        one snapshot
#   bash scripts/rerun_status.sh -w     refresh every 60s
#
# The delta section dominates the wall clock: one line per (cell x model family x
# control set), 316 of them, and the ETA below is just the average rate so far
# projected forward. Sections after it (effects, paired) are counted separately
# because their lines are cheaper.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/outputs/logs/run/11_single_indicator.log"
DELTA_TOTAL=316

snapshot () {
  local pid elapsed secs done_ pct rate eta
  pid=$(pgrep -f "11_window_compare.py" | head -1)
  printf '\n\033[1m=== single-indicator refit  %s ===\033[0m\n' "$(date +%H:%M:%S)"
  if [ -z "${pid:-}" ]; then
    printf '  \033[33m○ not running\033[0m (finished, or never started)\n'
  else
    elapsed=$(ps -p "$pid" -o etime= | xargs)
    printf '  \033[32m● running\033[0m  pid %s  elapsed %s\n' "$pid" "$elapsed"
  fi
  [ -f "$LOG" ] || { printf '  no log yet at %s\n\n' "$LOG"; return; }

  secs=$(( $(date +%s) - $(stat -f %B "$LOG") ))
  done_=$(grep -c '^\[delta\]' "$LOG")
  pct=$(( done_ * 100 / DELTA_TOTAL ))
  printf '  delta   %3d/%d fits (%d%%)' "$done_" "$DELTA_TOTAL" "$pct"
  if [ "$done_" -gt 2 ] && [ "$secs" -gt 0 ]; then
    eta=$(( (DELTA_TOTAL - done_) * secs / done_ / 60 ))
    printf '   ~%dm left at the current rate' "$eta"
  fi
  printf '\n'
  printf '  effects %3d fits   paired %3d fits\n' \
    "$(grep -c '^\[effects\]' "$LOG")" "$(grep -c '^\[paired\]' "$LOG")"
  printf '  last: %s\n\n' "$(tail -1 "$LOG" | cut -c1-100)"
}

if [ "${1:-}" = "-w" ]; then
  while true; do snapshot; sleep 60; done
else
  snapshot
fi
