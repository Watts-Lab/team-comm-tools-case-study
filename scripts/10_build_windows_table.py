"""Step 10 - one analysis table carrying all four ways of bounding the talk.

Step 3 built the published analysis table with three blocks (``pre``, ``post``,
``window``). This rebuilds it with a fourth, the cumulative block from steps 8-9,
and writes it alongside rather than over the original:

    pre     this round's contribution phase          (this round, before the outcome)
    post    last round's outcome + summary phases    (last round, after the outcome)
    window  pre and post merged
    cumulative     every message spoken so far in the game

**The table stays wide: one row per game-round, one outcome per row.** That is the
answer to the obvious worry about this design. The four blocks are four *column
families* describing the same decision, not four rows repeating it, so no
contribution is duplicated in any table a model is fitted on and no model sees the
same outcome twice. What is shared is across *reported results*: the ``pre``
row and the ``cumulative`` row of the results table are estimated on overlapping
game-rounds, so those two numbers are correlated with each other. They are read as
four descriptions of one sample, never as four independent findings, and the
per-block results carry ``n_games`` so the clustering behind each is visible.

The blocks are also nested by construction - ``window`` contains ``pre`` and
``post``; ``cumulative`` contains all three - so no two are ever entered into a single
model together.

Silence is handled exactly as in step 3: a round with no channel gets the neutral
fill everywhere (flagged by ``has_chat_channel``); a round with a channel where
nobody spoke gets a truthful zero on the count features and the neutral fill on the
rest (flagged by ``chose_silence_{block}``); a round that talked gets the toolkit's
values. See scripts/03_build_analysis_table.py for why filling with the mean rather
than zero is the right call for both model families.

One deliberate difference from step 3. Features are z-scored **within block**, on
the learning split's conversations of that block, rather than on one pooled set of
moments. A cumulative conversation is an order of magnitude longer than a
single-round one, so pooled moments would put ``sum_num_words__cumulative`` many SDs above
every ``pre`` value and make a coefficient "per SD" mean something different in each
block. Scaling within block makes every coefficient an effect per SD of the
variation that block actually has, which is what the comparison needs.

The feature set is *not* re-selected. It is read from
``outputs/tables/feature_manifest.csv``, chosen once on the learning split's
per-round conversations, so all four blocks are measured with the same ruler.

Outputs:
  * ``data/processed/analysis_windows_{split}.csv``

Run:  python scripts/10_build_windows_table.py
"""

import numpy as np
import pandas as pd

from config import (CONV_FEATURES, CONV_FEATURES_CUMULATIVE, DATA_PROCESSED, SPLITS,
                    TABLES)

BLOCKS = ("pre", "post", "window", "cumulative")

# Features whose truthful value is zero when nobody spoke, as opposed to undefined.
TRUE_TOTALS = ["sum_num_words", "sum_num_chars", "sum_num_messages"]


def kept_features():
    """The feature set chosen once, on the learning split, in step 3."""
    m = pd.read_csv(TABLES / "feature_manifest.csv")
    return m.loc[m["kept"], "feature"].tolist()


def load_conv(path):
    conv = pd.read_csv(path, low_memory=False)
    if "conv_id" not in conv.columns and "conversation_num" in conv.columns:
        conv = conv.rename(columns={"conversation_num": "conv_id"})
    conv = conv.loc[:, ~conv.columns.str.startswith("Unnamed")]
    conv["conv_id"] = conv["conv_id"].astype(str)
    return conv


def conv_tables(split):
    """{block: conversation-level frame} for one split."""
    per_round = load_conv(CONV_FEATURES / f"{split}_conv_level.csv")
    cumulative_path = CONV_FEATURES_CUMULATIVE / f"{split}_conv_level.csv"
    if not cumulative_path.exists():
        raise SystemExit(
            f"{cumulative_path} is missing - run scripts/09_extract_cumulative_features.py first.")
    cumulative = load_conv(cumulative_path)
    return {"pre": per_round, "post": per_round, "window": per_round, "cumulative": cumulative}


def block_moments(features):
    """Mean and SD per feature per block, from the learning split. See the docstring."""
    moments = {}
    for block, conv in conv_tables("learn").items():
        sub = conv[conv["conv_id"].astype(str).str.endswith(f"_{block}")] \
            if block != "window" else conv[conv["conv_id"].str.endswith("_win")]
        present = [c for c in features if c in sub.columns]
        stats = sub[present].agg(["mean", "std"])
        stats.loc["std"] = stats.loc["std"].replace(0, np.nan)
        moments[block] = stats
    return moments


def block_frame(rounds, conv, features, stats, block):
    """One block's features, z-scored within block and filled by kind of silence."""
    key = f"conv_id_{block}"
    present = [c for c in features if c in conv.columns and c in stats.columns]

    if rounds[key].isin(set(conv["conv_id"])).sum() == 0:
        raise SystemExit(f"No {block.upper()} conversation ids matched the toolkit "
                         f"output; the feature files are stale.")

    joined = (rounds[[key]].rename(columns={key: "conv_id"})
              .merge(conv[["conv_id"] + present], on="conv_id", how="left"))
    scaled = ((joined[present] - stats.loc["mean", present])
              / stats.loc["std", present])
    for col in features:
        if col not in scaled.columns:
            scaled[col] = np.nan

    described = scaled.notna().any(axis=1).to_numpy()
    scaled = scaled[features]
    scaled.index = rounds.index

    has_channel = rounds["has_chat_channel"].astype(bool).to_numpy()
    chose_silence = has_channel & ~described
    for col in (c for c in TRUE_TOTALS if c in present):
        zero_in_z = (0.0 - stats.loc["mean", col]) / stats.loc["std", col]
        scaled.loc[chose_silence, col] = zero_in_z

    scaled = scaled.fillna(0.0)
    scaled.columns = [f"{c}__{block}" for c in features]
    scaled[f"has_features_{block}"] = described
    scaled[f"chose_silence_{block}"] = chose_silence
    return scaled


def build(split, features, moments):
    rounds = pd.read_csv(DATA_PROCESSED / f"rounds_{split}.csv",
                         low_memory=False).reset_index(drop=True)
    rounds["gameId"] = rounds["gameId"].astype(str)
    # Step 1 names the pre/post/window ids; the cumulative one is named here.
    rounds["conv_id_cumulative"] = (rounds["gameId"] + "_r"
                             + rounds["round_index"].astype(str) + "_cumulative")
    for b in BLOCKS:
        rounds[f"conv_id_{b}"] = rounds[f"conv_id_{b}"].astype(str)

    tables = conv_tables(split)
    analysis = pd.concat(
        [rounds] + [block_frame(rounds, tables[b], features, moments[b], b)
                    for b in BLOCKS], axis=1)
    analysis["lagged_contribution"] = analysis["lagged_contribution"].fillna(
        analysis["lagged_contribution"].mean())

    out = DATA_PROCESSED / f"analysis_windows_{split}.csv"
    analysis.to_csv(out, index=False)
    print(f"[{split}] {len(analysis)} game-rounds (one row per outcome) x "
          f"{analysis.shape[1]} columns -> {out.name}")
    for b in BLOCKS:
        talked = analysis[f"has_features_{b}"].astype(bool)
        print(f"         {b.upper():<7} talked {int(talked.sum()):5d} game-rounds "
              f"in {analysis.loc[talked, 'gameId'].nunique():4d} games   "
              f"chose silence {int(analysis[f'chose_silence_{b}'].sum()):5d}")
    return analysis


if __name__ == "__main__":
    features = kept_features()
    print(f"carrying {len(features)} features from the step-3 manifest, "
          f"across {len(BLOCKS)} blocks")
    moments = block_moments(features)
    for split in SPLITS:
        build(split, features, moments)
