"""Step 3 - join toolkit features to game-round outcomes and choose a feature set.

Produces one analysis row per game-round:

    outcome + design parameters + timing + momentum + conversation features

Conversation features arrive in **two blocks**, suffixed ``__pre`` and ``__post``:
deliberation about this round's decision, and reaction to the previous round's
result. Keeping them separate is the point of the exercise - they can be entered
into a model independently, so the analysis can say which kind of talk predicts.

The awkward part is silence, and there are two kinds of it. A round with **no
channel** could not have a conversation; a round with a channel where **nobody
spoke** chose not to have one. The second is a behaviour, not a missing value, and
it is the largest group of the three - so the two are encoded differently:

    no channel        neutral fill everywhere; the counterfactual, flagged by
                      has_chat_channel = 0
    channel, silent   0 on the handful of features that are genuinely counts
                      (a group that said nothing said zero words), neutral fill on
                      the rest, flagged by chose_silence_{block} = 1
    talked            the toolkit's actual values

Filling everything with zero would be wrong for 138 of the 140 features. Nearly all
of them are per-message means or conversation-level ratios, which are *undefined*
without a conversation rather than zero: a raw 0 on ``mean_positive_bert`` asserts
maximally non-positive content, which is a claim about speech that never happened.
Only true totals - ``sum_num_words`` and ``sum_num_messages`` - have a truthful zero.

Neutral-filling rather than zero-filling matters for two different reasons:

  * For a linear model the choice is only a reparameterization - the fill value is
    constant across all no-chat rows, so the channel indicator absorbs it and the
    fit is identical either way. What changes is meaning. Filled with the mean, the
    channel coefficient reads as "the effect of having a channel, holding the
    conversation at a typical one", which is the estimand of interest. Filled with
    zero it would read as the effect relative to a conversation scoring zero on
    every feature at once, which is not a conversation anyone had.
  * For the random forest the choice changes the fit. Zero-filling would park every
    no-chat round at the extreme edge of all 136 features, letting the trees
    identify them from any feature at all and dissolving the channel/content
    distinction. Mean-filling puts them in the middle of every distribution, so the
    only clean way to tell them apart is the channel indicator itself.

Features are z-scored first, on the learning split's conversing rounds only, which
makes the neutral value exactly zero and keeps the held-out split out of the
scaling decision.

Outputs:
  * ``data/processed/analysis_{split}.csv``
  * ``outputs/tables/feature_manifest.csv`` - the conversation features kept for
    modeling with their family, plus the reason any candidate was dropped. The
    manifest lists each feature once; the analysis table carries it twice, once
    per block.

Run:  python scripts/03_build_analysis_table.py
"""

import re

import numpy as np
import pandas as pd

from config import (CONV_FEATURES, DATA_PROCESSED, FEATURE_FAMILIES, SPLITS,
                    TABLES, TCT_ID_COLS)

# Redundancy is now handled by the toolkit itself (see step 2): the learning
# split is written with `drop_redundant_columns=True`, which groups features
# correlated above 0.9 and keeps one representative per group, and also drops
# columns that are mostly missing or mostly zero. What remains here is a thin
# safety net for columns that survived reduction but still cannot support a
# regression - the toolkit reduces on its own output, not on the joined analysis
# table, so a feature can be fine there and near-constant once merged.
MIN_UNIQUE = 5          # drop near-constant features


def family_of(feature):
    """Map a toolkit column to its construct family; first pattern wins."""
    for name, pattern in FEATURE_FAMILIES:
        if re.search(pattern, feature):
            return name
    return "Other"


def load_conv_features(split):
    """Load the conversation-level toolkit output, keyed by game-round."""
    conv = pd.read_csv(CONV_FEATURES / f"{split}_conv_level.csv", low_memory=False)
    if "conv_id" not in conv.columns and "conversation_num" in conv.columns:
        conv = conv.rename(columns={"conversation_num": "conv_id"})
    conv = conv.loc[:, ~conv.columns.str.startswith("Unnamed")]
    conv["conv_id"] = conv["conv_id"].astype(str)
    return conv


def candidate_feature_columns(conv):
    """Every numeric feature the toolkit kept, minus identifiers.

    This used to impose its own rule - keep the conversation ``mean_*`` of each
    chat-level feature and the toolkit's native conversation-level columns, drop
    the min/max/stdev and per-speaker variants - because the raw output repeated
    every construct four or five ways.

    That rule is now not just redundant but harmful. The toolkit's reduction
    deduplicates *across* aggregations: a group typically contains the mean, min,
    max, stdev and per-speaker forms of one construct, and the representative it
    keeps is whichever has the most valid data and the highest variance. That is
    usually a ``max_*`` or per-speaker column, not the mean. Filtering to means
    afterwards discarded 235 of the 238 surviving features and left 13, chosen by
    the accident of which variant happened to represent its group.
    """
    numeric = set(conv.select_dtypes(include=[np.number]).columns) - TCT_ID_COLS
    return [c for c in conv.columns if c in numeric]


def screen_features(conv, candidates):
    """Drop the few survivors that still cannot support a regression."""
    reasons, kept = {}, []
    for col in candidates:
        values = conv[col]
        if values.nunique(dropna=True) < MIN_UNIQUE:
            reasons[col] = f"near-constant ({values.nunique(dropna=True)} unique values)"
        else:
            kept.append(col)
    return kept, reasons


def write_manifest(candidates, kept, reasons):
    manifest = pd.DataFrame({"feature": candidates})
    manifest["kept"] = manifest["feature"].isin(kept)
    manifest["family"] = manifest["feature"].map(family_of)
    manifest["drop_reason"] = manifest["feature"].map(reasons).fillna("")
    manifest.to_csv(TABLES / "feature_manifest.csv", index=False)

    unclassified = manifest.loc[manifest["kept"] & (manifest["family"] == "Other"),
                                "feature"].tolist()
    if unclassified:
        print(f"  ! {len(unclassified)} kept features fell outside every family: "
              f"{unclassified[:8]}")
    counts = manifest[manifest["kept"]]["family"].value_counts()
    print("  features per family: "
          + ", ".join(f"{fam} {n}" for fam, n in counts.items()))


def scaling_moments(conv, feature_cols):
    """Mean and SD of each feature across the learning split's conversations.

    Computed once, over both blocks pooled, and reused for every split and block so
    that a PRE and a POST feature are on the same scale and the held-out data has no
    influence on either the scale or the neutral fill value.
    """
    stats = conv[feature_cols].agg(["mean", "std"])
    stats.loc["std"] = stats.loc["std"].replace(0, np.nan)
    return stats


# Features whose truthful value is zero when nobody spoke, as opposed to undefined.
TRUE_TOTALS = ["sum_num_words", "sum_num_chars", "sum_num_messages"]


def block_frame(rounds, conv, feature_cols, moments, block):
    """One block's features, z-scored and filled by kind of silence, suffixed by block."""
    key = f"conv_id_{block}"
    present = [c for c in feature_cols if c in conv.columns]

    # Guard against feature files left over from an earlier run with a different
    # conversation key. Every feature would silently become a neutral fill and the
    # content term would be a guaranteed zero, which looks like a finding.
    matched = rounds[key].isin(set(conv["conv_id"])).sum()
    if matched == 0:
        raise SystemExit(
            f"No {block.upper()} conversation ids matched the toolkit output. The "
            f"feature files are stale - re-run 02_extract_features.py --force.")

    joined = (rounds[[key]].rename(columns={key: "conv_id"})
              .merge(conv[["conv_id"] + present], on="conv_id", how="left"))
    scaled = ((joined[present] - moments.loc["mean", present])
              / moments.loc["std", present])
    for col in feature_cols:
        if col not in scaled.columns:
            scaled[col] = np.nan

    described = scaled.notna().any(axis=1).to_numpy()
    scaled = scaled[feature_cols]
    scaled.index = rounds.index

    # Groups with a channel that said nothing: zero on the true totals, on the raw
    # scale, so they land below every real conversation rather than at its average.
    has_channel = rounds["has_chat_channel"].astype(bool).to_numpy()
    chose_silence = has_channel & ~described
    for col in (c for c in TRUE_TOTALS if c in scaled.columns):
        zero_in_z = (0.0 - moments.loc["mean", col]) / moments.loc["std", col]
        scaled.loc[chose_silence, col] = zero_in_z

    scaled = scaled.fillna(0.0)          # everything else: neutral, i.e. undefined
    scaled.columns = [f"{c}__{block}" for c in feature_cols]
    scaled[f"has_features_{block}"] = described
    scaled[f"chose_silence_{block}"] = chose_silence
    return scaled


def build(split, feature_cols, moments):
    rounds = pd.read_csv(DATA_PROCESSED / f"rounds_{split}.csv").reset_index(drop=True)
    for col in ("conv_id_pre", "conv_id_post"):
        rounds[col] = rounds[col].astype(str)
    conv = load_conv_features(split)

    analysis = pd.concat(
        [rounds] + [block_frame(rounds, conv, feature_cols, moments, b)
                    for b in ("pre", "post")], axis=1)

    # The first round of a game has no previous round to average, so momentum is
    # undefined there; those rows were already dropped in step 1, but a game whose
    # first played round is missing can still leave a gap.
    analysis["lagged_contribution"] = analysis["lagged_contribution"].fillna(
        analysis["lagged_contribution"].mean())

    out = DATA_PROCESSED / f"analysis_{split}.csv"
    analysis.to_csv(out, index=False)
    print(f"[{split}] {len(analysis)} game-rounds x {analysis.shape[1]} columns "
          f"-> {out.name}")
    for block in ("pre", "post"):
        print(f"         {block.upper():<4} talked {int(analysis[f'has_features_{block}'].sum()):5d}"
              f"   chose silence {int(analysis[f'chose_silence_{block}'].sum()):5d}"
              f"   no channel {int((~analysis.has_chat_channel.astype(bool)).sum()):5d}")
    return analysis


if __name__ == "__main__":
    # The feature set and the scale are both chosen on the learning split alone.
    learn_conv = load_conv_features("learn")
    candidates = candidate_feature_columns(learn_conv)
    feature_cols, reasons = screen_features(learn_conv, candidates)
    print(f"toolkit reduction left {len(candidates)} candidate features; "
          f"{len(feature_cols)} survive the near-constant check")
    write_manifest(candidates, feature_cols, reasons)

    # The held-out split keeps every column, so the learning split's surviving
    # features must all be present in it. Anything absent is a real mismatch
    # rather than something to paper over with a neutral fill.
    val_cols = set(load_conv_features("val").columns)
    missing = [c for c in feature_cols if c not in val_cols]
    if missing:
        raise SystemExit(f"{len(missing)} learn features absent from the held-out "
                         f"split, e.g. {missing[:5]} - re-run 02 for both splits.")

    moments = scaling_moments(learn_conv, feature_cols)
    for split in SPLITS:
        build(split, feature_cols, moments)
