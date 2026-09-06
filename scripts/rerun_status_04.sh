#!/usr/bin/env bash
# Progress for the step 4 refit (04_analysis.py, launched by hand).
#   bash scripts/rerun_status_04.sh        one snapshot
#   bash scripts/rerun_status_04.sh -w     refresh every 60s
#
# Step 4 prints one lettered header per section and writes its tables at the end.
# The section the paper is waiting on is E (round_stage.csv, the proportional
# thirds); everything after it is bookkeeping for other tables.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/outputs/logs/run/04_single_indicator.log"
STAGES="$ROOT/outputs/tables/round_stage.csv"

# section letter -> rough minutes, from the estimates in 04_analysis.py's SECTIONS
ORDER="A B C D D E F G H"

snapshot () {
  printf '\n\033[1m=== step 4 refit  %s ===\033[0m\n' "$(date +%H:%M:%S)"
  local pid; pid=$(pgrep -f "04_analysis.py" | head -1)
  if [ -n "${pid:-}" ]; then
    printf '  \033[32m● running\033[0m  pid %s  elapsed %s\n' \
      "$pid" "$(ps -p "$pid" -o etime= | xargs)"
  else
    printf '  \033[33m○ not running\033[0m — finished, or never started\n'
  fi
  [ -f "$LOG" ] || { printf '  no log yet\n\n'; return; }

  local done_ total last
  done_=$(grep -c '^[A-H]\. ' "$LOG"); total=$(echo $ORDER | wc -w | xargs)
  last=$(grep '^[A-H]\. ' "$LOG" | tail -1)
  printf '  sections %d/%d   now: %s\n' "$done_" "$total" "${last:-starting}"

  # The one the paper needs.
  if grep -q '^E\. ' "$LOG"; then
    printf '  \033[1mE (round_stage / proportional thirds)\033[0m: reached\n'
  else
    printf '  E (round_stage / proportional thirds): not yet\n'
  fi
  printf '  round_stage.csv last written: %s\n' \
    "$([ -f "$STAGES" ] && date -r "$STAGES" +%H:%M:%S || echo -)"
  printf '  tables written: %s\n\n' \
    "$(grep -c 'tables written' "$LOG")"
}

if [ "${1:-}" = "-w" ]; then
  while true; do snapshot; sleep 60; done
else
  snapshot
fi
