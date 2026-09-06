#!/usr/bin/env bash
# Progress for the single-indicator refit (step 11, launched by hand).
#   bash scripts/rerun_status.sh        one snapshot, with a 20s rate sample
#   bash scripts/rerun_status.sh -w     refresh every 60s
#
# The log carries no timestamps, so the ETA comes from sampling the fit counts
# twice, 20 seconds apart, and projecting the remaining work in the section that
# is currently running. Section totals are what a full run writes:
#   delta   316 = 79 cells x 2 model families x 2 control sets
#   effects  80 = the cells that get a per-feature screen
#   paired   36 = bin x model family x control set, on the common subsample
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/outputs/logs/run/11_single_indicator.log"
DELTA_TOTAL=316; EFFECTS_TOTAL=80; PAIRED_TOTAL=36
SAMPLE=${SAMPLE:-20}

count () { grep -c "^\[$1\]" "$LOG" 2>/dev/null || echo 0; }

snapshot () {
  local pid d1 e1 p1 d2 e2 p2 done_ total rate left
  pid=$(pgrep -f "11_window_compare.py" | head -1)
  printf '\n\033[1m=== single-indicator refit  %s ===\033[0m\n' "$(date +%H:%M:%S)"
  if [ -z "${pid:-}" ]; then
    printf '  \033[33m○ not running\033[0m — finished, or never started\n'
  else
    printf '  \033[32m● running\033[0m  pid %s  elapsed %s\n' \
      "$pid" "$(ps -p "$pid" -o etime= | xargs)"
  fi
  [ -f "$LOG" ] || { printf '  no log yet\n\n'; return; }

  d1=$(count delta); e1=$(count effects); p1=$(count paired)
  printf '  delta   %3d/%d   effects %2d/%d   paired %2d/%d\n' \
    "$d1" "$DELTA_TOTAL" "$e1" "$EFFECTS_TOTAL" "$p1" "$PAIRED_TOTAL"
  [ -z "${pid:-}" ] && { printf '  last: %s\n\n' "$(tail -1 "$LOG" | cut -c1-90)"; return; }

  # Which section is live, and how fast is it going?
  sleep "$SAMPLE"
  d2=$(count delta); e2=$(count effects); p2=$(count paired)
  if   [ "$p1" -gt 0 ] || [ "$e1" -ge "$EFFECTS_TOTAL" ]; then
    done_=$p2; total=$PAIRED_TOTAL; rate=$(( p2 - p1 )); local name=paired
  elif [ "$e1" -gt 0 ] || [ "$d1" -ge "$DELTA_TOTAL" ]; then
    done_=$e2; total=$EFFECTS_TOTAL; rate=$(( e2 - e1 )); local name=effects
  else
    done_=$d2; total=$DELTA_TOTAL; rate=$(( d2 - d1 )); local name=delta
  fi
  left=$(( total - done_ ))
  if [ "$rate" -gt 0 ]; then
    printf '  %s: %d left, %d fits per %ds  ->  ~%dm to go\n' \
      "$name" "$left" "$rate" "$SAMPLE" "$(( left * SAMPLE / rate / 60 + 1 ))"
  else
    printf '  %s: %d left, none finished in the last %ds (a slow fit)\n' \
      "$name" "$left" "$SAMPLE"
  fi
  printf '  last: %s\n\n' "$(tail -1 "$LOG" | cut -c1-90)"
}

if [ "${1:-}" = "-w" ]; then
  while true; do snapshot; sleep 40; done
else
  snapshot
fi
