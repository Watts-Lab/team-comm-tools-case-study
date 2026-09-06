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
         "06_feature_examples.py", "07_lexicon_words.py",
         "08_cumulative_chat.py", "09_extract_cumulative_features.py",
         "10_build_windows_table.py", "11_window_compare.py",
         "13_window_figures.py", "14_descriptive_figures.py",
         "15_short_game_scope.py", "16_runtime_table.py",
         "18_pre_post_power.py", "19_pipeline_figure.py",
         "20_appendix_cells_table.py", "21_short_game_table.py"]

# Steps that need an argument to write the output the paper uses. Without a
# category, step 7 lists the lexicons and exits.
STEP_ARGS = {"07_lexicon_words.py": ["relative"]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-features", action="store_true",
                    help="re-run the toolkit even if feature files exist")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    for step in STEPS:
        cmd = [sys.executable, str(here / step)] + STEP_ARGS.get(step, [])
        if step.startswith("02") and args.force_features:
            cmd.append("--force")
        print(f"\n{'=' * 70}\n{step}\n{'=' * 70}")
        subprocess.run(cmd, check=True, cwd=here)
