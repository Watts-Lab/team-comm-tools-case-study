"""Step 3 - join toolkit features to game-round outcomes and choose a feature set.

Produces one analysis row per game-round:

    outcome + design parameters + timing + momentum + conversation features

Conversation features arrive in **two blocks**, suffixed ``__pre`` and ``__post``:
deliberation about this round's decision, and reaction to the previous round's
result. Keeping them separate is the point of the exercise - they can be entered
into a model independently, so the analysis can say which kind of talk predicts.

The awkward part is that game-rounds with no chat channel have no conversation to
describe. They are not dropped - they are the counterfactual the whole question
rests on - so their conversation features are set to a **neutral value**: the mean
of the feature among rounds that did have a conversation.

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

# Screening thresholds for candidate conversation features. These guard against
# columns that cannot support a regression rather than against uninteresting
# ones; substantive selection is left to the models in step 4.
MAX_MISSING = 0.30      # drop features missing in >30% of conversations
MIN_UNIQUE = 5          # drop near-constant features
MAX_ABS_CORR = 0.95     # drop one of any near-duplicate pair


def family_of(feature):
    """Map a toolkit column to its construct family; first pattern wins."""
    for name, pattern in FEATURE_FAMILIES:
        if re.search(pattern, feature):
            return name
    return "Other"


def load_conv_features(split):
    """Load the conversation-level toolkit output, keyed by game-round."""
    conv = pd.read_csv(CONV_FEATURES / f"{split}_conv_level.csv")
    if "conv_id" not in conv.columns and "conversation_num" in conv.columns:
        conv = conv.rename(columns={"conversation_num": "conv_id"})
    conv = conv.loc[:, ~conv.columns.str.startswith("Unnamed")]
    conv["conv_id"] = conv["conv_id"].astype(str)
    return conv


def candidate_feature_columns(conv):
    """Numeric conversation features, restricted to one aggregation per construct.

    The toolkit reports every chat-level feature four ways at the conversation level
    (mean / min / max / stdev), and again after first aggregating within speaker
    (``mean_user_*``). This keeps:

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
    """Drop features that cannot support a regression; record why for each drop."""
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
    dropped = set()
    for i, col in enumerate(kept):
        if col in dropped:
            continue
        for other in kept[i + 1:]:
            if other not in dropped and corr.loc[col, other] > MAX_ABS_CORR:
                dropped.add(other)
                reasons[other] = f"|r| > {MAX_ABS_CORR} with {col}"

    return [c for c in kept if c not in dropped], reasons


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


def block_frame(rounds, conv, feature_cols, moments, block):
    """One block's features, z-scored and neutral-filled, suffixed by block."""
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

    described = scaled.notna().any(axis=1)
    scaled = scaled[feature_cols].fillna(0.0)
    scaled.columns = [f"{c}__{block}" for c in feature_cols]
    scaled.index = rounds.index
    scaled[f"has_features_{block}"] = described.to_numpy()
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
    print(f"         described by the toolkit: "
          f"{int(analysis.has_features_pre.sum())} PRE, "
          f"{int(analysis.has_features_post.sum())} POST")
    return analysis


if __name__ == "__main__":
    # The feature set and the scale are both chosen on the learning split alone.
    learn_conv = load_conv_features("learn")
    candidates = candidate_feature_columns(learn_conv)
    feature_cols, reasons = screen_features(learn_conv, candidates)
    print(f"screened {len(candidates)} candidate features -> {len(feature_cols)} kept")
    write_manifest(candidates, feature_cols, reasons)

    moments = scaling_moments(learn_conv, feature_cols)
    for split in SPLITS:
        build(split, feature_cols, moments)
