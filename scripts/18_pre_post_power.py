"""Step 18 - how much talk each conversation boundary actually holds.

The Pre boundary returns a null in every time period. A null is only informative
if the boundary that produced it had enough material to detect an effect, and Pre
holds much less material than Post: fewer rounds contain a message at all, and the
rounds that do contain fewer of them. This step measures that gap so the main
article can state it alongside the null.

Reads  data/processed/chat_{split}.csv and analysis_{split}.csv
Writes outputs/tables/pre_post_power.csv

Run:  python scripts/18_pre_post_power.py
"""

import pandas as pd

from config import DATA_PROCESSED, SPLITS, TABLES

BLOCKS = ("pre", "post")
STAGES = ("opening", "middle", "endgame")


def rows_for(split):
    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv")
    chat["n_words"] = chat["text"].fillna("").str.split().str.len()
    per_conv = chat.groupby(["block", "conv_id"]).agg(
        n_messages=("text", "size"), n_words=("n_words", "sum"),
        n_speakers=("playerId", "nunique")).reset_index()

    analysis = pd.read_csv(DATA_PROCESSED / f"analysis_{split}.csv")
    analysis = analysis[analysis["has_chat_channel"]]

    out = []
    for block in BLOCKS:
        ids = analysis[["stage_absolute", f"conv_id_{block}"]].rename(
            columns={f"conv_id_{block}": "conv_id"})
        merged = ids.merge(per_conv[per_conv["block"] == block],
                           on="conv_id", how="left")
        for stage in (*STAGES, "all"):
            cell = merged if stage == "all" else merged[
                merged["stage_absolute"] == stage]
            spoke = cell[cell["n_messages"].notna()]
            out.append({
                "split": split,
                "block": block,
                "stage": stage,
                "game_rounds": len(cell),
                "rounds_with_talk": len(spoke),
                "pct_silent": 100 * (1 - len(spoke) / len(cell)) if len(cell) else float("nan"),
                "median_messages": spoke["n_messages"].median(),
                "mean_messages": spoke["n_messages"].mean(),
                "median_words": spoke["n_words"].median(),
                "mean_words": spoke["n_words"].mean(),
                "median_speakers": spoke["n_speakers"].median(),
                "total_messages": spoke["n_messages"].sum(),
            })
    return out


def main():
    rows = [r for split in SPLITS for r in rows_for(split)]
    table = pd.DataFrame(rows)
    path = TABLES / "pre_post_power.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
