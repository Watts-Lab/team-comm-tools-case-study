"""Step 3 - join toolkit features to game outcomes and choose a feature set.

Produces one analysis row per game:
    outcome + design parameters + (for groups that talked) conversation features.

Groups without a chat channel keep every conversation column as NaN by
construction - that absence is the comparison the research question rests on.

Outputs:
  * ``data/processed/analysis_{split}.csv``
  * ``outputs/tables/feature_manifest.csv`` - the conversation features kept for
    modeling, with the reason any candidate was dropped.

Run:  python scripts/03_build_analysis_table.py
"""

import re

import numpy as np
import pandas as pd

from config import (CONFIG_COLS, CONV_FEATURES, DATA_PROCESSED, SPLITS,
                    TABLES, TCT_ID_COLS)

# Screening thresholds for candidate conversation features. These guard against
# columns that cannot support a regression rather than against uninteresting
# ones; substantive selection is left to the penalized models in step 4.
MAX_MISSING = 0.30      # drop features missing in >30% of conversations
MIN_UNIQUE = 5          # drop near-constant features
MAX_ABS_CORR = 0.95     # drop one of any near-duplicate pair


def load_conv_features(split):
    """Load the conversation-level toolkit output, keyed by game."""
    conv = pd.read_csv(CONV_FEATURES / f"{split}_conv_level.csv")
    if "gameId" not in conv.columns and "conversation_num" in conv.columns:
        conv = conv.rename(columns={"conversation_num": "gameId"})
    conv = conv.loc[:, ~conv.columns.str.startswith("Unnamed")]
    conv["gameId"] = conv["gameId"].astype(str)
    return conv


def candidate_feature_columns(conv):
    """Numeric conversation features, restricted to one aggregation per construct.

    The toolkit reports every chat-level feature four ways at the conversation level
    (mean / min / max / stdev), and again after first aggregating within speaker
    (``mean_user_*``). With 147 conversations to learn from, carrying all 3,000
    columns would be mostly redundancy. This keeps:

      * the toolkit's native conversation-level features (turn-taking, burstiness,
        discursive diversity, Gini coefficients of participation, and so on), which
        have no aggregation prefix because they are already properties of the group;
      * the conversation ``mean_*`` of each chat-level feature - the most
        interpretable summary of "how much of this was in the conversation".

    Dropping the min/max/stdev and per-speaker variants is a deliberate simplification,
    not a claim that they carry no signal.
    """
    numeric = set(conv.select_dtypes(include=[np.number]).columns) - TCT_ID_COLS
    agg_prefix = re.compile(r"^(mean|min|max|stdev)_")
    return [c for c in conv.columns
            if c in numeric
            and (not agg_prefix.match(c)
                 or (c.startswith("mean_") and not c.startswith("mean_user_")))]


def screen_features(conv, candidates):
    """Drop features that cannot support a regression; log why for each drop."""
    reasons = {}
    kept = []

    for col in candidates:
        s = conv[col]
        if s.isna().mean() > MAX_MISSING:
            reasons[col] = f"missing in {s.isna().mean():.0%} of conversations"
        elif s.nunique(dropna=True) < MIN_UNIQUE:
            reasons[col] = f"near-constant ({s.nunique(dropna=True)} unique values)"
        else:
            kept.append(col)

    # Remove near-duplicate columns, keeping whichever appears first so the
    # decision is deterministic and reproducible from the learn split alone.
    corr = conv[kept].corr().abs()
    dropped_corr = set()
    for i, col in enumerate(kept):
        if col in dropped_corr:
            continue
        for other in kept[i + 1:]:
            if other in dropped_corr:
                continue
            if corr.loc[col, other] > MAX_ABS_CORR:
                dropped_corr.add(other)
                reasons[other] = f"|r| > {MAX_ABS_CORR} with {col}"

    kept = [c for c in kept if c not in dropped_corr]
    return kept, reasons


def build(split, feature_cols=None):
    games = pd.read_csv(DATA_PROCESSED / f"games_{split}.csv")
    games["gameId"] = games["gameId"].astype(str)
    conv = load_conv_features(split)

    if feature_cols is None:
        candidates = candidate_feature_columns(conv)
        feature_cols, reasons = screen_features(conv, candidates)
        manifest = pd.DataFrame({"feature": candidates})
        manifest["kept"] = manifest["feature"].isin(feature_cols)
        manifest["drop_reason"] = manifest["feature"].map(reasons).fillna("")
        manifest.to_csv(TABLES / "feature_manifest.csv", index=False)
        print(f"screened {len(candidates)} candidate features -> "
              f"{len(feature_cols)} kept (manifest in outputs/tables/)")

    keep = ["gameId"] + [c for c in feature_cols if c in conv.columns]
    analysis = games.merge(conv[keep], on="gameId", how="left")

    out = DATA_PROCESSED / f"analysis_{split}.csv"
    analysis.to_csv(out, index=False)
    print(f"[{split}] {len(analysis)} games x {analysis.shape[1]} columns -> {out.name}")
    return feature_cols


if __name__ == "__main__":
    # The feature set is chosen on the learning split only, then applied
    # unchanged to the held-out split.
    feature_cols = build("learn")
    for split in SPLITS:
        if split != "learn":
            build(split, feature_cols)

    missing_configs = [c for c in CONFIG_COLS
                       if c not in pd.read_csv(DATA_PROCESSED / "analysis_learn.csv").columns]
    if missing_configs:
        print(f"warning: config columns absent from the data: {missing_configs}")
