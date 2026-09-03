"""Run the whole case study end to end.

    python scripts/run_all.py [--force-features]

Each step is also runnable on its own; this just runs them in order in a fresh
interpreter each time so that a failure points at one step.
"""

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = ["01_prepare_data.py", "02_extract_features.py",
         "03_build_analysis_table.py", "04_analysis.py",
         "08_cumulative_chat.py", "09_extract_cumulative_features.py",
         "10_build_windows_table.py", "11_window_compare.py",
         "12_first_n.py", "13_window_figures.py",
         "15_short_game_scope.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-features", action="store_true",
                    help="re-run the toolkit even if feature files exist")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    for step in STEPS:
        cmd = [sys.executable, str(here / step)]
        if step.startswith("02") and args.force_features:
            cmd.append("--force")
        print(f"\n{'=' * 70}\n{step}\n{'=' * 70}")
        subprocess.run(cmd, check=True, cwd=here)
