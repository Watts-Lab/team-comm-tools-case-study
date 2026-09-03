#!/usr/bin/env bash
# One-line progress for whichever section of 11_window_compare.py is running.
#   bash scripts/progress.sh        one snapshot
#   bash scripts/progress.sh -w     refresh every 20s
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/outputs/logs/run/11_samples.log"
PIDF="$ROOT/outputs/logs/run/11.pid"

snapshot () {
  # Analysable cells drive both sections: delta runs 4 fits per cell (2 control
  # sets x 2 model families), effects runs one regression sweep per cell.
  local cells delta_total eff_total delta_done eff_done sec done_ total pid el secs
  cells=$(awk -F, 'NR>1 && $0 ~ /True/ {n++} END{print n+0}' \
          "$ROOT/outputs/tables/windows/block_clustering.csv" 2>/dev/null)
  [ "${cells:-0}" -eq 0 ] && cells=79
  delta_total=$(( cells * 4 )); eff_total=$cells
  delta_done=$(grep -c "^\[delta\]" "$LOG" 2>/dev/null || echo 0)
  eff_done=$(grep -c "^\[effects\]" "$LOG" 2>/dev/null || echo 0)
  # The paired section refits every window on one common subsample; its cell count
  # is fixed by the design, and 37 is what a complete run emits.
  local paired_total=37
  local paired_done
  paired_done=$(grep -c "^\[paired\]" "$LOG" 2>/dev/null || echo 0)

  # Each section is timed from its own start, not from the start of the run, or
  # the paired ETA inherits the hour that delta spent.
  local T="$ROOT/outputs/tables/windows" anchor
  if [ "$delta_done" -lt "$delta_total" ]; then
    sec=delta;   done_=$delta_done;  total=$delta_total
    anchor="$T/block_clustering.csv"
  elif [ "$eff_done" -lt "$eff_total" ]; then
    sec=effects; done_=$eff_done;    total=$eff_total
    anchor="$T/block_delta_r2.csv"
  else
    sec=paired;  done_=$paired_done; total=$paired_total
    anchor="$T/block_agreement.csv"
  fi

  local bar="" filled i
  filled=$(( total > 0 ? done_ * 30 / total : 0 ))
  for ((i=0;i<30;i++)); do
    if [ $i -lt $filled ]; then bar="${bar}#"; else bar="${bar}."; fi
  done

  pid=$(grep -o "[0-9]\+" "$PIDF" 2>/dev/null | head -1)
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    el=$(ps -p "$pid" -o etime= | xargs)
    # BSD ps has no etimes; etime is [[dd-]hh:]mm:ss.
    secs=$(awk -v t="$el" 'BEGIN{ d=0; split(t,a,"-"); if(length(a)>1){d=a[1]; t=a[2]}
           n=split(t,b,":"); s=0; for(i=1;i<=n;i++) s=s*60+b[i]; print s+d*86400 }')
    # ETA uses this section's own rate, not the whole run's.
    local sec_start rate eta
    if [ "$sec" = effects ]; then
      sec_start=$(awk -v t="$el" 'BEGIN{print 0}')  # unknown; use overall below
    fi
    local sec_secs
    sec_secs=$(( $(date +%s) - $(stat -f %m "$anchor" 2>/dev/null || date +%s) ))
    [ "$sec_secs" -lt 1 ] && sec_secs=1
    if [ "$done_" -gt 0 ]; then
      eta=$(( (total - done_) * sec_secs / done_ / 60 ))
      printf "%-8s %s %d/%d (%d%%)  this section %d min  ~%d min left\n" \
        "$sec" "$bar" "$done_" "$total" $(( done_ * 100 / total )) \
        $(( sec_secs / 60 )) "$eta"
    else
      printf "%-8s %s %d/%d  warming up\n" "$sec" "$bar" "$done_" "$total"
    fi
  else
    printf "%-8s %s %d/%d  FINISHED\n" "$sec" "$bar" "$done_" "$total"
  fi
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; snapshot; sleep 20; done
else
  snapshot
fi
