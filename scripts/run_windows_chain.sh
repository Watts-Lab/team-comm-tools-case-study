#!/usr/bin/env bash
# Waits for the cumulative featurization (step 9) to finish, then builds the
# four-block analysis table (step 10) and runs the window comparison (step 11).
#
# Written as a chain rather than three separate launches because step 9 takes
# hours and steps 10-11 are worthless without it. Each stage is skipped if its
# output already exists and is newer than its input, so re-running is cheap.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/Users/xehu/.virtualenvs/team_comm_tools/bin/python
LOGS="$ROOT/outputs/logs/run"
mkdir -p "$LOGS"

say () { printf '\n[chain %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---- 1. wait for step 9 -----------------------------------------------------
PID=$(grep -o '[0-9]\+' "$LOGS/09.pid" 2>/dev/null | head -1)
if [ -n "${PID:-}" ]; then
  say "waiting on cumulative featurization (pid $PID)"
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
fi

for f in outputs/features_cumulative/output/conv/learn_conv_level.csv \
         outputs/features_cumulative/output/conv/val_conv_level.csv; do
  if [ ! -f "$f" ]; then
    say "ABORT: step 9 exited without producing $f"
    say "check outputs/logs/run/09_cumulative_features.log"
    exit 1
  fi
done
say "cumulative features present"

# ---- 2. step 10: the four-block analysis table ------------------------------
say "building the four-block analysis table"
$PY -u scripts/10_build_windows_table.py 2>&1 | tee "$LOGS/10_build_windows.log"
[ -f data/processed/analysis_windows_learn.csv ] || { say "ABORT: step 10 failed"; exit 1; }

# ---- 3. step 11: the window comparison --------------------------------------
if [ ! -f scripts/11_window_compare.py ]; then
  say "scripts/11_window_compare.py is not there yet; stopping after step 10."
  say "run it by hand when it lands."
  exit 0
fi
# The cheap sections first, so there are readable tables within the hour rather
# than only after the delta section (which is ~90% of the runtime) completes.
say "window comparison: clustering, per-feature effects, agreement (~30-60 min)"
$PY -u scripts/11_window_compare.py --only clustering,effects,agreement \
    > "$LOGS/11_window_compare.log" 2>&1 &
echo "PID $!" > "$LOGS/11.pid"
wait $!
say "cheap sections done; block_feature_effects.csv and friends are readable now"

say "window comparison: delta R2, the long pole (~4-7 h)"
$PY -u scripts/11_window_compare.py --only delta \
    >> "$LOGS/11_window_compare.log" 2>&1 &
echo "PID $!" > "$LOGS/11.pid"
wait $!
say "window comparison finished"

# ---- 4. the cumulative half of the first-N analysis -------------------------
if [ -f scripts/12_first_n.py ]; then
  say "running the cumulative half of the first-N analysis"
  $PY -u scripts/12_first_n.py --summary both \
      >> "$LOGS/12_first_n.log" 2>&1
fi
say "chain complete"
