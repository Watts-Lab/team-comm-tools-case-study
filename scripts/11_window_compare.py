"""Step 11 - which conversation features matter, from which window, and when.

Step 4 answered "does talk predict contribution" with the talk bounded two narrow
ways: ``pre`` (this round's contribution phase, before the outcome is revealed) and
``post`` (last round's outcome and summary phases, after it). Neither block can see
anything the group said earlier in the game, so a null result for them is a null
result about *recent* talk only. Step 10 adds ``cumulative`` - every message spoken so far
this game - and this script asks the same questions of all four bounds at once:

    pre     this round, before the outcome
    post    last round, after the outcome
    window  pre and post merged
    cumulative     every message so far this game

crossed with where the round sits in its game, because the published finding is
that talk predicts contribution in the opening rounds and nowhere else. If that is
a fact about *when in a game* talk matters, a cumulative window should not rescue
the later rounds. If it is instead an artifact of how little talk a single-round
window contains by round 12, ``cumulative`` should do better there than ``pre`` does.

**Why the analysis table stays wide, and why the blocks are never combined.**
``analysis_windows_{split}.csv`` has one row per game-round and one outcome per row;
the four blocks are four *column families* describing the same decision, not four
copies of it. So no model here is ever fitted on a table in which a contribution
appears twice, and no model sees the same outcome more than once. The blocks are
also nested by construction - ``window`` contains ``pre`` and ``post``, ``cumulative``
contains all three - which is the reason no two of them are entered into one model:
their columns are different summaries of overlapping message sets, so a joint fit
would be asking which of two descriptions of the same conversation is the real one.
Each block gets its own model, against its own baseline, and the results are read
side by side.

**What the correlation between reported rows does and does not mean.** Because the
blocks are nested and the round bins overlap in games (a 20-round game contributes
rounds to every bin), the rows of every table below are correlated with each other.
A ``cumulative`` ΔR² and a ``pre`` ΔR² in the same bin are two estimates from largely the
same game-rounds, and their difference has no standard error attached here. So
these tables support statements of the form "this block, in this bin, adds this
much, with this interval", and comparisons of the form "one interval excludes zero
and the other does not". They do **not** support "block A is significantly better
than block B", and no such test is computed. The honest use of the comparison is
descriptive: four descriptions of one sample, each carrying the number of games
behind it so the reader can see how thin any given cell is.

Four tables, all under ``outputs/tables/windows/``:

  block_delta_r2.csv        what conversation *content* adds over rules, timing and
                            the channel indicator, cross-validated on the learning
                            split and scored on held-out games, per (block, bin,
                            model family, control set). Two control sets: the
                            published one, and a robustness variant that adds last
                            round's contribution. The second matters most for
                            ``cumulative``, which has by far the most room to proxy for a
                            group's own trajectory rather than for anything said.
  block_feature_effects.csv one clustered OLS per (block, bin, feature), on the
                            learning split and again on held-out games. FDR is
                            applied within each (block, bin), because each of those
                            is a separate screen asked of a separate subsample.
  block_agreement.csv       do the blocks agree about which features matter, and do
                            the bins? Correlations between coefficient vectors, plus
                            counts of features significant in both with the same sign.
  block_clustering.csv      the power inventory: every cell that could have been
                            fitted, whether it was, and why not. Skipped cells are
                            in this table by design - a cell dropped for having 14
                            games is a finding about the design, and silently
                            omitting it would make the surviving cells look like the
                            whole picture.

Every row of every table carries the rounds and games behind it, on both splits.
Rounds within a game are not independent - a 30-round game contributes 30 of them,
sharing a group, a treatment and often a conversation - so a cell with 200 rounds
from 9 games is a far weaker claim than one with 200 rounds from 90, and only the
game count shows the difference. For the same reason every fold and every bootstrap
draw in this script holds out or resamples whole games; that machinery lives in
``modeling.py`` and is used rather than reimplemented.

Cells are restricted to rounds that actually had that kind of talk
(``has_features_{block}``), exactly as ``round_stage()`` in 04_analysis.py does.
Across all rounds only about a fifth do, and on the rest every feature column is the
same neutral fill, so an unrestricted content model would mostly be fitting the fill.

Run:  python scripts/11_window_compare.py
      python scripts/11_window_compare.py --table analysis --blocks pre,post --quick
"""

import argparse
import functools

import itertools

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import modeling
from config import (BLOCK_CONTAINS, BLOCK_DEFINITION, BLOCK_MEANING,
                    DATA_PROCESSED, TABLES, TABLES_WINDOWS)
from modeling import CHANNEL, GROUP, MOMENTUM, OUTCOME

# Two ways of saying where a round sits in its game, both carried in a ``binning``
# column so one table answers both questions.
#
#   stage      the published partition (04_analysis.py's ``stage_absolute``), kept
#              verbatim so these results are comparable with the paper's.
#   round_bin  a finer cut on the round counter - this is the "how many rounds have
#              already happened" axis, and it is the one that can distinguish
#              "round 1 is special" from "the first third is special". Games run
#              from 3 to 30 rounds, so the later bins pool increasingly wide spans
#              to stay above the clustering thresholds.
#
# A third pseudo-binning, ``all``, pools every round: the block comparison without
# any timing cut, which is the right thing to read first.
STAGE_COL = "stage_absolute"
STAGE_ORDER = ["opening", "middle", "endgame"]
ALL_ROUNDS = "all rounds"

# Offsets from the first round index actually present in the data - the counter is
# 0-based in this dataset, and hard-coding 1..3 here would silently shift every bin.
ROUND_BIN_SPANS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 9), (10, None)]

# Cheaper settings for smoke tests. The bootstrap count is the only thing that has
# to be injected rather than passed: modeling.delta_r2 calls the module-global
# cluster_bootstrap, whose n_boot default is bound at definition.
QUICK_FOLDS = 3
QUICK_BOOT = 200

MODEL_KINDS = ("elastic net", "random forest")

# The published control set, and the robustness variant. Momentum is kept out of
# the default for the reason 04_analysis.py gives: it predicts contribution better
# than everything else combined (r = 0.87 round to round), so conditioning on it
# leaves nothing for talk to explain. It is exactly the right robustness check for
# ``cumulative``, though, whose whole design gives it more chance to encode a group's
# trajectory than a single round's talk has.
CONTROL_SETS = {"rules+timing": (), "rules+timing+momentum": (MOMENTUM,)}

# Which rounds a cell is fitted on. The two answer different questions and both
# are reported.
#
#   channel   every round in which a chat channel was open. Channel availability
#             was randomized, so this conditions only on the treatment. Rounds
#             where nobody spoke are kept, carrying the neutral fill described in
#             10_build_windows_table.py, and the two silence indicators enter the
#             baseline *and* the full model so that the reported quantity is still
#             content net of whether anyone spoke.
#
#   talkers   only the rounds that actually produced a message. This is the
#             sharper test of content - speech is constant, so it cannot leak into
#             the estimate - but it conditions on a post-treatment behaviour:
#             groups that choose to talk may differ from those that do not in ways
#             that also move contribution. It is reported as descriptive of
#             conversing rounds, not as a causal effect of what was said.
#
# The two differ mostly in scale rather than in sign: the fill carries no content
# variance, so including it divides the same improvement over a larger total
# variance and shrinks the estimate roughly in proportion to the filled fraction.
SAMPLES = ("channel", "talkers")

SECTIONS = {
    "clustering": "power/clustering inventory of every cell   (seconds)",
    "delta":      "content ΔR² per block per bin              (hours)",
    "effects":    "per-feature effects per block per bin      (~10 min)",
    "agreement":  "do blocks and bins agree                   (seconds)",
    "paired":     "blocks compared on ONE common subsample    (~40 min)",
}


# ------------------------------------------------------------- loading ------
def load(table, split):
    path = DATA_PROCESSED / f"{table}_{split}.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Build it with scripts/10_build_windows_table.py, "
            f"or smoke-test against the published table with --table analysis.")
    return modeling.to_numeric_bools(pd.read_csv(path, low_memory=False))


def available_blocks(df):
    return [b for b in BLOCK_MEANING if f"has_features_{b}" in df.columns]


def round_bins(df):
    """Label each round by how many rounds have already happened.

    Returns a Series of labels; the spans are read against the smallest round index
    present so the labels stay truthful whatever base the counter uses.
    """
    first = int(df["round_index"].min())
    labels = pd.Series(pd.NA, index=df.index, dtype=object)
    idx = pd.to_numeric(df["round_index"], errors="coerce")
    for lo, hi in ROUND_BIN_SPANS:
        mask = idx >= first + lo
        if hi is not None:
            mask &= idx <= first + hi
        labels[mask] = _round_bin_label(lo, hi)
    return labels


def _round_bin_label(lo, hi):
    """Human 1-based label for a span given as offsets from the first round."""
    if hi is None:
        return f"rounds {lo + 1}+"
    if lo == hi:
        return f"round {lo + 1}"
    return f"rounds {lo + 1}-{hi + 1}"


ROUND_BIN_ORDER = [_round_bin_label(lo, hi) for lo, hi in ROUND_BIN_SPANS]


def binnings(df):
    """{binning: (Series of bin labels, order)} for one frame."""
    return {
        "all": (pd.Series(ALL_ROUNDS, index=df.index), [ALL_ROUNDS]),
        "stage": (df[STAGE_COL], STAGE_ORDER),
        "round_bin": (round_bins(df), ROUND_BIN_ORDER),
    }


# --------------------------------------------------------------- cells ------
def cell_frames(learn, val, block, binning, bin_label, learn_bins, val_bins,
                sample="talkers"):
    """The learning and held-out rows of one (block, bin, sample) cell."""
    def take(df, bins):
        in_bin = bins[binning][0] == bin_label
        keep = (df[CHANNEL].astype(bool) if sample == "channel"
                else df[f"has_features_{block}"].astype(bool))
        return df[in_bin & keep]
    return take(learn, learn_bins), take(val, val_bins)


def indicators_for(df, block, sample):
    """Silence indicators, which vary only when silent rounds are in the sample."""
    if sample != "channel":
        return []
    return [c for c in (f"has_features_{block}", f"chose_silence_{block}")
            if c in df.columns and df[c].nunique() > 1]


def skip_reason(l_sub, v_sub):
    """Why this cell cannot be fitted, or None if it can.

    The thresholds are modeling's: ten group-held-out folds cannot be cut from nine
    games, and a cluster bootstrap over a handful of games has no resolution. Both
    splits are checked, because a held-out estimate over four games would be
    reported to three decimal places and mean nothing.
    """
    for name, sub in (("learning", l_sub), ("held-out", v_sub)):
        rows, games = modeling.cell_size(sub)
        if rows == 0:
            return f"no {name} rounds in this cell"
        if rows < modeling.MIN_ROWS:
            return f"{rows} {name} rounds (< {modeling.MIN_ROWS})"
        if games < modeling.MIN_GAMES:
            return f"{games} {name} games (< {modeling.MIN_GAMES})"
    return None


def sizes(l_sub, v_sub):
    """The four counts every row in every table carries."""
    rows, games = modeling.cell_size(l_sub)
    rows_v, games_v = modeling.cell_size(v_sub)
    return {"n_rounds": rows, "n_games": games,
            "n_rounds_heldout": rows_v, "n_games_heldout": games_v}


def iter_cells(learn, val, blocks, learn_bins, val_bins, analysable_only=True,
               samples=SAMPLES):
    """Every (binning, bin, block, sample) cell, in reporting order."""
    for binning, (_, order) in learn_bins.items():
        for bin_label in order:
            for block in blocks:
                for sample in samples:
                    l_sub, v_sub = cell_frames(learn, val, block, binning,
                                               bin_label, learn_bins, val_bins,
                                               sample)
                    reason = skip_reason(l_sub, v_sub)
                    if analysable_only and reason:
                        continue
                    yield binning, bin_label, block, sample, l_sub, v_sub, reason


def tag(binning, bin_label, block):
    return f"{binning}/{bin_label}/{block}"


# ------------------------------------------- 1. the power inventory ---------
def clustering(learn, val, blocks, learn_bins, val_bins):
    """Every cell the design contains, fitted or not, with its clustering.

    This table exists because "which cells did we analyse" is itself a result. The
    median rounds per game is the part that is easy to miss: a cell of 300 rounds
    from 25 games is 12 correlated rounds per game, and its effective sample size is
    much closer to 25 than to 300.
    """
    rows = []
    for binning, bin_label, block, sample, l_sub, v_sub, reason in iter_cells(
            learn, val, blocks, learn_bins, val_bins, analysable_only=False):
        per_game = (l_sub.groupby(GROUP).size().median() if len(l_sub) else np.nan)
        per_game_v = (v_sub.groupby(GROUP).size().median() if len(v_sub) else np.nan)
        rows.append({"binning": binning, "bin": bin_label, "block": block,
                     "block_meaning": BLOCK_MEANING[block], "sample": sample,
                     **sizes(l_sub, v_sub),
                     "median_rounds_per_game": per_game,
                     "median_rounds_per_game_heldout": per_game_v,
                     "analysed": reason is None,
                     "skip_reason": reason or ""})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_WINDOWS / "block_clustering.csv", index=False)
    kept = int(out["analysed"].sum())
    print(f"[clustering] {kept} of {len(out)} cells clear "
          f"{modeling.MIN_ROWS} rounds and {modeling.MIN_GAMES} games", flush=True)
    return out


# ------------------------------------------------- 2. what content adds -----
def delta_tables(learn, val, blocks, learn_bins, val_bins, kinds, n_folds):
    """ΔR² of conversation content over rules, timing and the channel, per cell.

    The baseline is deliberately ``controls + channel`` rather than ``controls``
    alone: within a cell every round had talk, so the channel indicator is constant
    and the comparison is content against everything knowable before a word was
    said. The block's own indicators are left out for the same reason - inside a
    cell they do not vary.
    """
    rows = []
    for binning, bin_label, block, sample, l_sub, v_sub, _ in iter_cells(
            learn, val, blocks, learn_bins, val_bins):
        content = modeling.block_features(l_sub, block, TABLES, with_indicator=False)
        ind = indicators_for(l_sub, block, sample)
        size = sizes(l_sub, v_sub)
        for ctrl_name, extra in CONTROL_SETS.items():
            # The indicators sit in the baseline as well as the full model, so the
            # difference between them is content and never "did anyone speak".
            base = modeling.controls(l_sub, extra=extra) + [CHANNEL] + ind
            for kind in kinds:
                print(f"[delta] {tag(binning, bin_label, block)} | {sample} | "
                      f"{kind} | {ctrl_name} | {size['n_rounds']} rounds, "
                      f"{size['n_games']} games, {len(content)} features",
                      flush=True)
                res = modeling.delta_r2(kind, l_sub, v_sub, base, base + content,
                                        n_folds=n_folds)
                rows.append({"binning": binning, "bin": bin_label, "block": block,
                             "block_meaning": BLOCK_MEANING[block],
                             "sample": sample, "model_family": kind,
                             "controls": ctrl_name, "n_features": len(content),
                             **size, **res})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_WINDOWS / "block_delta_r2.csv", index=False)
    return out


# ------------------------------------------ 3. which features, per cell -----
def feature_effects(learn, val, blocks, learn_bins, val_bins):
    """One clustered OLS per feature per cell, on both splits.

    ``modeling.one_feature`` re-standardizes the feature within the cell, so a
    coefficient is an effect per SD of variation that this cell actually contains -
    which is the only way a ``cumulative`` coefficient and a ``pre`` coefficient can be put
    on the same axis, since a cumulative conversation and a single-round one have
    very different spreads.
    """
    fam_of = modeling.manifest(TABLES).set_index("feature")["family"].to_dict()
    rows, attempted = [], 0
    for binning, bin_label, block, sample, l_sub, v_sub, _ in iter_cells(
            learn, val, blocks, learn_bins, val_bins):
        # In the channel sample the silence indicators are controls, or a feature
        # would be credited with the difference between speaking and not.
        ctrl_l = modeling.controls(l_sub) + indicators_for(l_sub, block, sample)
        ctrl_v = modeling.controls(v_sub) + indicators_for(v_sub, block, sample)
        content = modeling.block_features(l_sub, block, TABLES, with_indicator=False)
        size = sizes(l_sub, v_sub)
        print(f"[effects] {tag(binning, bin_label, block)} | {sample} | "
              f"{len(content)} features | {size['n_rounds']} rounds, "
              f"{size['n_games']} games", flush=True)
        for feature in content:
            attempted += 1
            coef, p, lo, hi = modeling.one_feature(l_sub, feature, ctrl_l)
            if np.isnan(coef):
                # Too little variation in this cell to estimate; dropping it here
                # also keeps it out of the FDR denominator, which is correct - an
                # inestimable regression is not a test that was performed.
                continue
            v_coef, v_p, v_lo, v_hi = modeling.one_feature(v_sub, feature, ctrl_v)
            base = feature.rsplit("__", 1)[0]
            rows.append({"binning": binning, "bin": bin_label, "block": block,
                         "block_meaning": BLOCK_MEANING[block], "sample": sample,
                         "feature": base,
                         "family": fam_of.get(base, "Other"), **size,
                         "coef_learn": coef, "p_learn": p,
                         "ci_low_learn": lo, "ci_high_learn": hi,
                         "coef_val": v_coef, "p_val": v_p,
                         "ci_low_val": v_lo, "ci_high_val": v_hi})

    out = pd.DataFrame(rows)
    if out.empty:
        out.to_csv(TABLES_WINDOWS / "block_feature_effects.csv", index=False)
        return out

    # FDR within each (block, bin): each cell is its own screen over the same ~150
    # features, asked of a different subsample. Pooling them into one correction
    # would let a cell where nothing happened buy significance for a cell where
    # something did, and vice versa.
    out["q_learn"] = np.nan
    for _, grp in out.groupby(["binning", "bin", "block", "sample"]):
        out.loc[grp.index, "q_learn"] = multipletests(grp["p_learn"],
                                                      method="fdr_bh")[1]
    # The same convention as 04_analysis.py: survive the screen on the learning
    # split, hold up unadjusted on games that played no part in selecting it, and
    # point the same way in both.
    out["replicates"] = ((out["q_learn"] < 0.05) & (out["p_val"] < 0.05)
                         & (np.sign(out["coef_learn"]) == np.sign(out["coef_val"])))
    out = out.sort_values(["binning", "bin", "block", "sample",
                           "p_learn"]).reset_index(drop=True)
    out.to_csv(TABLES_WINDOWS / "block_feature_effects.csv", index=False)
    print(f"[effects] kept {len(out)} of {attempted} regressions; "
          f"{int((out.q_learn < 0.05).sum())} clear FDR on learn, "
          f"{int(out.replicates.sum())} replicate", flush=True)
    return out


# ---------------------------------------------- 4. do they agree at all -----
def _pair_row(sub_a, sub_b, key_a, key_b):
    """Correlation and co-significance between two cells' coefficient vectors."""
    a = sub_a.set_index("feature")["coef_learn"]
    b = sub_b.set_index("feature")["coef_learn"]
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    sig_a = set(sub_a.loc[sub_a["p_learn"] < 0.05, "feature"])
    sig_b = set(sub_b.loc[sub_b["p_learn"] < 0.05, "feature"])
    both = sig_a & sig_b
    agree = sum(1 for f in both
                if f in pair.index and np.sign(pair.loc[f, "a"]) == np.sign(pair.loc[f, "b"]))
    smaller = sub_a if sub_a["n_rounds"].iloc[0] <= sub_b["n_rounds"].iloc[0] else sub_b
    return {
        **key_a, **key_b,
        "n_features": len(pair),
        "coef_correlation": pair["a"].corr(pair["b"]) if len(pair) > 2 else np.nan,
        "n_sig_a": len(sig_a), "n_sig_b": len(sig_b),
        "n_sig_both": len(both), "n_sig_both_same_sign": agree,
        # The binding sample is the smaller of the two cells, so that is what the
        # standard size columns report; the per-side counts are alongside.
        **{c: smaller[c].iloc[0] for c in ("n_rounds", "n_games",
                                           "n_rounds_heldout", "n_games_heldout")},
        "n_rounds_a": sub_a["n_rounds"].iloc[0], "n_games_a": sub_a["n_games"].iloc[0],
        "n_rounds_b": sub_b["n_rounds"].iloc[0], "n_games_b": sub_b["n_games"].iloc[0],
    }


def agreement(effects, sample="talkers"):
    """Two questions, one table.

    *Blocks within a bin*: given the same rounds, do four different bounds on the
    talk rank the features the same way? A high correlation means the bound barely
    matters - the extra messages ``cumulative`` sees are not changing which features look
    predictive. A low one means "which window you measure" is itself a finding.

    *Bins within a block*: the published question, asked per block. A feature that
    helps in the opening and does nothing later is a different phenomenon from one
    that helps throughout, and a pooled coefficient hides both.

    A correlation here is between two vectors estimated on overlapping game-rounds,
    so it is a descriptive summary of agreement, not a test of anything. No p-value
    is attached to it and none should be.

    Computed on one sample at a time, since a coefficient estimated among talkers
    and one estimated across all channel rounds are on different scales.
    """
    if "sample" in effects.columns:
        effects = effects[effects["sample"] == sample]
    rows = []
    for (binning, bin_label), grp in effects.groupby(["binning", "bin"]):
        blocks = [b for b in BLOCK_MEANING if b in set(grp["block"])]
        for i, a in enumerate(blocks):
            for b in blocks[i + 1:]:
                rows.append({"comparison": "blocks within bin", "binning": binning,
                             **_pair_row(grp[grp.block == a], grp[grp.block == b],
                                         {"bin_a": bin_label, "bin_b": bin_label},
                                         {"block_a": a, "block_b": b})})

    for (binning, block), grp in effects.groupby(["binning", "block"]):
        order = ROUND_BIN_ORDER if binning == "round_bin" else STAGE_ORDER
        present = [x for x in order if x in set(grp["bin"])]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                rows.append({"comparison": "bins within block", "binning": binning,
                             **_pair_row(grp[grp.bin == a], grp[grp.bin == b],
                                         {"bin_a": a, "bin_b": b},
                                         {"block_a": block, "block_b": block})})

    cols = ["comparison", "binning", "block_a", "block_b", "bin_a", "bin_b"]
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[cols + [c for c in out.columns if c not in cols]]
    out.to_csv(TABLES_WINDOWS / "block_agreement.csv", index=False)
    return out


# ------------------------------- 5. blocks compared on one common subsample --
# Everything above fits each block on the rounds that had *that* block's talk, so
# the four ΔR² values in a bin come from four overlapping-but-different samples and
# each outcome is predicted once per block - in the opening stage, 330 distinct
# outcomes predicted 3.34 times each. Every one of those estimates is individually
# valid, but they are not independent of each other, so "cumulative found more than
# post" cannot be given an interval from that table.
#
# This section fixes that by changing the unit of analysis: restrict to the rounds
# that have **all** blocks, and fit every block on exactly those rows. Then a
# contrast is a paired quantity. Because the two models share a baseline and share
# the outcome vector, the baseline cancels,
#
#     ΔR²_a - ΔR²_b = R²(y, full_a) - R²(y, full_b)
#
# and resampling games for both terms with the *same* draw removes the variance the
# two share instead of ignoring it. That is what makes the interval on the
# difference honest where a comparison of two separately-bootstrapped intervals is
# not.
#
# The cost is sample size and generality: the common subsample is the intersection,
# so it is smaller than any single block's, and it conditions on being a round where
# the group talked in every sense at once. Read the per-block ΔR² here as "what this
# block is worth on the rounds where all four are measurable", not as a replacement
# for the marginal estimates above.

def common_frames(learn, val, blocks, binning, bin_label, learn_bins, val_bins):
    """The rounds of one bin that have every block's talk, in both splits."""
    def take(df, bins):
        mask = bins[binning][0] == bin_label
        for b in blocks:
            mask = mask & df[f"has_features_{b}"].astype(bool)
        return df[mask]
    return take(learn, learn_bins), take(val, val_bins)


def paired_tables(learn, val, blocks, learn_bins, val_bins, kinds, n_folds):
    """Per-block ΔR² and every pairwise contrast, all on one common subsample."""
    per_block, contrasts = [], []

    for binning, (_, order) in learn_bins.items():
        for bin_label in order:
            l_sub, v_sub = common_frames(learn, val, blocks, binning, bin_label,
                                         learn_bins, val_bins)
            reason = skip_reason(l_sub, v_sub)
            if reason:
                print(f"[paired] {binning}/{bin_label}: skipped - {reason}",
                      flush=True)
                continue
            size = sizes(l_sub, v_sub)
            y = l_sub[OUTCOME].to_numpy(dtype=float)
            y_val = v_sub[OUTCOME].to_numpy(dtype=float)
            groups, vgroups = l_sub[GROUP].to_numpy(), v_sub[GROUP].to_numpy()

            for ctrl_name, extra in CONTROL_SETS.items():
                base_cols = modeling.controls(l_sub, extra=extra) + [CHANNEL]
                for kind in kinds:
                    print(f"[paired] {binning}/{bin_label} | {kind} | {ctrl_name} | "
                          f"{size['n_rounds']} rounds, {size['n_games']} games",
                          flush=True)
                    # One baseline, shared by every block in this cell.
                    base = modeling.out_of_fold(kind, l_sub, base_cols, n_folds)
                    _, base_v = modeling.fit_predict_heldout(kind, l_sub, v_sub,
                                                             base_cols)
                    base_r2, base_v_r2 = modeling.r2(y, base), modeling.r2(y_val, base_v)

                    full, full_v = {}, {}
                    for block in blocks:
                        cols = base_cols + modeling.block_features(
                            l_sub, block, TABLES, with_indicator=False)
                        full[block] = modeling.out_of_fold(kind, l_sub, cols, n_folds)
                        _, full_v[block] = modeling.fit_predict_heldout(
                            kind, l_sub, v_sub, cols)
                        lo, hi = modeling.cluster_bootstrap(
                            groups, lambda i, f=full[block]: (modeling.r2(y[i], f[i])
                                                              - modeling.r2(y[i], base[i])))
                        lo_v, hi_v = modeling.cluster_bootstrap(
                            vgroups, lambda i, f=full_v[block]: (modeling.r2(y_val[i], f[i])
                                                                 - modeling.r2(y_val[i], base_v[i])))
                        per_block.append({
                            "binning": binning, "bin": bin_label, "block": block,
                            "sample": "talkers",
                            "block_meaning": BLOCK_MEANING[block],
                            "model_family": kind, "controls": ctrl_name, **size,
                            "delta_cv_r2": modeling.r2(y, full[block]) - base_r2,
                            "ci_low": lo, "ci_high": hi,
                            "delta_heldout_r2": modeling.r2(y_val, full_v[block]) - base_v_r2,
                            "ci_low_heldout": lo_v, "ci_high_heldout": hi_v})

                    for a, b in itertools.combinations(blocks, 2):
                        lo, hi = modeling.cluster_bootstrap(
                            groups, lambda i, fa=full[a], fb=full[b]:
                                (modeling.r2(y[i], fa[i]) - modeling.r2(y[i], fb[i])))
                        lo_v, hi_v = modeling.cluster_bootstrap(
                            vgroups, lambda i, fa=full_v[a], fb=full_v[b]:
                                (modeling.r2(y_val[i], fa[i]) - modeling.r2(y_val[i], fb[i])))
                        contrasts.append({
                            "binning": binning, "bin": bin_label,
                            "sample": "talkers",
                            "block_a": a, "block_b": b,
                            "nested": (b in BLOCK_CONTAINS[a]
                                       or a in BLOCK_CONTAINS[b]),
                            "model_family": kind, "controls": ctrl_name, **size,
                            "diff_cv_r2": modeling.r2(y, full[a]) - modeling.r2(y, full[b]),
                            "ci_low": lo, "ci_high": hi,
                            "diff_heldout_r2": (modeling.r2(y_val, full_v[a])
                                                - modeling.r2(y_val, full_v[b])),
                            "ci_low_heldout": lo_v, "ci_high_heldout": hi_v})

    pb = pd.DataFrame(per_block)
    ct = pd.DataFrame(contrasts)
    pb.to_csv(TABLES_WINDOWS / "block_paired.csv", index=False)
    ct.to_csv(TABLES_WINDOWS / "block_paired_contrasts.csv", index=False)
    return pb, ct


def write_definitions():
    """The four blocks, spelled out, next to the results that use them."""
    rows = [{"block": b, "short": BLOCK_MEANING[b], "definition": BLOCK_DEFINITION[b],
             "contains": ", ".join(BLOCK_CONTAINS[b]) or "-"}
            for b in BLOCK_MEANING]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_WINDOWS / "block_definitions.csv", index=False)
    return out


# ------------------------------------------------------------------- main ---
def main(args):
    run = set(args.only.split(",")) if args.only else set(SECTIONS)
    unknown = run - set(SECTIONS)
    if unknown:
        raise SystemExit(f"unknown section(s) {sorted(unknown)}; "
                         f"choose from {sorted(SECTIONS)}")

    kinds = ("elastic net",) if args.quick else MODEL_KINDS
    n_folds = QUICK_FOLDS if args.quick else modeling.N_FOLDS
    if args.quick:
        # delta_r2 reaches for the module-global, so the cheap bootstrap has to be
        # installed rather than passed. Smoke tests only; never for a reported run.
        modeling.cluster_bootstrap = functools.partial(modeling.cluster_bootstrap,
                                                       n_boot=QUICK_BOOT)

    learn, val = load(args.table, "learn"), load(args.table, "val")
    blocks = ([b.strip() for b in args.blocks.split(",")] if args.blocks
              else available_blocks(learn))
    missing = [b for b in blocks if f"has_features_{b}" not in learn.columns]
    if missing:
        raise SystemExit(f"{args.table} has no columns for block(s) {missing}; "
                         f"available: {available_blocks(learn)}")

    learn_bins, val_bins = binnings(learn), binnings(val)
    print(f"table {args.table}: learn {len(learn)} game-rounds from "
          f"{learn[GROUP].nunique()} games, val {len(val)} from "
          f"{val[GROUP].nunique()}", flush=True)
    print(f"blocks: {', '.join(blocks)}   round index "
          f"{int(learn.round_index.min())}-{int(learn.round_index.max())}   "
          f"bins: {', '.join(ROUND_BIN_ORDER)}", flush=True)
    print(f"running: {', '.join(sorted(run))}"
          f"{'  (quick)' if args.quick else ''}\n", flush=True)

    write_definitions()

    if "clustering" in run:
        inventory = clustering(learn, val, blocks, learn_bins, val_bins)
        print(inventory.to_string(index=False), "\n", flush=True)

    if "delta" in run:
        deltas = delta_tables(learn, val, blocks, learn_bins, val_bins, kinds, n_folds)
        show = ["binning", "bin", "block", "model_family", "controls", "n_rounds",
                "n_games", "delta_cv_r2", "ci_low", "ci_high", "delta_heldout_r2"]
        print(deltas[show].round(4).to_string(index=False), "\n", flush=True)

    effects = None
    if "effects" in run:
        effects = feature_effects(learn, val, blocks, learn_bins, val_bins)

    if "agreement" in run:
        if effects is None:
            path = TABLES_WINDOWS / "block_feature_effects.csv"
            if not path.exists():
                raise SystemExit("agreement needs block_feature_effects.csv; "
                                 "run --only effects,agreement")
            effects = pd.read_csv(path)
        agree = agreement(effects)
        print(agree.round(3).to_string(index=False), "\n", flush=True)

    if "paired" in run:
        pb, ct = paired_tables(learn, val, blocks, learn_bins, val_bins,
                               kinds, n_folds)
        show = ["binning", "bin", "block", "model_family", "controls", "n_rounds",
                "n_games", "delta_cv_r2", "ci_low", "ci_high", "delta_heldout_r2"]
        print(pb[show].round(4).to_string(index=False), "\n", flush=True)
        showc = ["binning", "bin", "block_a", "block_b", "nested", "model_family",
                 "controls", "diff_cv_r2", "ci_low", "ci_high", "diff_heldout_r2"]
        print(ct[showc].round(4).to_string(index=False), "\n", flush=True)

    print(f"tables written to {TABLES_WINDOWS}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compare the four talk windows across points in a game.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="sections:\n  " + "\n  ".join(f"{k:<12}{v}" for k, v in SECTIONS.items()))
    ap.add_argument("--only", help="comma-separated section names; default is all")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: elastic net only, fewer folds and bootstraps")
    ap.add_argument("--table", default="analysis_windows",
                    help="analysis table stem in data/processed "
                         "(default analysis_windows; 'analysis' has no cumulative block)")
    ap.add_argument("--blocks", help="comma-separated subset of "
                                     f"{', '.join(BLOCK_MEANING)}")
    main(ap.parse_args())
