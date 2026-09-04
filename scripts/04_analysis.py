"""Step 4 - decompose contribution into channel, deliberation, reaction, and momentum.

    Which conversation features predict greater contribution in groups that
    communicate, versus groups that do not?

Every game-round is in the sample, whether or not the group had a channel, and the
models are built up one block at a time:

    M0  design parameters + round timing     the rules of the game
    M1  + chat channel indicator             the *mere channel*
    M2  + PRE features   (deliberation)      what they said while deciding
    M3  + POST features  (reaction)          what they said about the last result
    M4  + both blocks

M2, M3 and M4 are each compared against M1 rather than chained, so the two talk
blocks are not competing for whichever happens to be entered first.

Last round's contribution is not a control here. It is by far the strongest single
predictor (r = 0.87 round to round) and conditioning on it leaves nothing for talk
to explain, so it answers a different and narrower question. The cost is that any
talk effect below is partly confounded with a group's own trajectory - POST-block
talk in particular reacts to the previous result. Read the talk terms as
"associated with", not "adds beyond what the group was already doing".

Follow-ups: which toolkit feature family carries any content term, whether talk
matters more early or late in a game, and which individual features move
contribution - screened with FDR on the learning split and re-tested on held-out data.

Two model families throughout: a penalized linear model and a random forest. Both
have their hyperparameters chosen inside each training fold, so neither gets to
peek at the data its cross-validated score is computed on.
Everything is fit on the learning split; held-out data is scored once, at the end.

Run:  python scripts/04_analysis.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr as scipy_qr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from config import (CONFIG_COLS, DATA_PROCESSED, SEED, TABLES,
                    TABLES_DIAGNOSTICS, TIMING_COLS)

OUTCOME = "contribution_rate"
GROUP = "gameId"          # rounds within a game are not independent observations
CHANNEL = "has_chat_channel"
MOMENTUM = "lagged_contribution"
# Three ways of bounding the talk that precedes one contribution decision. The
# window is PRE and POST merged; it is the same messages, so it is not independent
# of the other two and is never entered alongside them in a single model.
BLOCKS = {"pre": "deliberation", "post": "reaction", "window": "both merged"}
SPLIT_BLOCKS = ["pre", "post"]

# Two ways of saying where a round sits in its game; they answer different
# questions and disagree substantially, because games run from 3 to 30 rounds.
# Absolute staging is primary: round 1 is round 1 in every game, which is what
# makes "what gets said early" readable. Relative staging is the robustness check.
STAGINGS = {
    "absolute": ("stage_absolute", ["opening", "middle", "endgame"]),
    "relative": ("stage_relative", ["early", "middle", "late"]),
}
PRIMARY_STAGING = "absolute"
N_FOLDS = 10
N_BOOT = 2000

# The outcome is a share of an endowment, so it cannot fall outside [0, 1]. Every
# model here is unconstrained and will occasionally predict outside that range; on
# a small subsample a single degenerate penalized fit once produced -5.6, which by
# itself dragged that subsample's R² to -0.66. Predictions are therefore clipped to
# the feasible range before scoring. This is a statement about the outcome, not a
# tuning knob: it is applied identically to every model, split, and subsample.
OUTCOME_BOUNDS = (0.0, 1.0)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- helpers ---
def load(split):
    df = pd.read_csv(DATA_PROCESSED / f"analysis_{split}.csv", low_memory=False)
    for col in df.columns:
        if df[col].dtype == object:
            lowered = df[col].astype(str).str.lower()
            if lowered.isin({"true", "false", "nan"}).all():
                df[col] = lowered.map({"true": 1.0, "false": 0.0})
    return df


def manifest():
    return pd.read_csv(TABLES / "feature_manifest.csv")


def block_features(df, block, with_indicator=True):
    """The kept toolkit features for one talk block, as they appear in the table.

    Two indicators ride along by default. Without them the models cannot tell a
    round with no conversation from a round with a perfectly average one - both
    carry the neutral fill on all 140 columns - and since only about a fifth of
    game-rounds have talk of a given kind, most rows would otherwise be
    indistinguishable constants that the features can only overfit.

    ``chose_silence_{block}`` is the more interesting of the two: it marks a group
    that *had* a channel and said nothing, which is a behaviour rather than a
    missing value, and which describes more game-rounds than actual talk does. A
    linear model cannot construct it from the channel and talk indicators on its
    own, since it is their interaction.
    """
    m = manifest()
    cols = [f"{f}__{block}" for f in m.loc[m["kept"], "feature"]]
    cols = [c for c in cols if c in df.columns]
    if with_indicator:
        for flag in (f"has_features_{block}", f"chose_silence_{block}"):
            if flag in df.columns and df[flag].nunique() > 1:
                cols.append(flag)
    return cols


def families(df):
    """{family: [feature columns]} for the merged-window block."""
    m = manifest()
    m = m[m["kept"]]
    out = {}
    for fam, sub in m.groupby("family"):
        cols = [f"{f}__window" for f in sub["feature"]]
        out[fam] = [c for c in cols if c in df.columns]
    return out


def controls(df, extra=()):
    """Everything known before this round's talk: the game's rules and the clock.

    Momentum (last round's contribution) is deliberately **not** here. It predicts
    contribution better than everything else combined (r = 0.87 round to round), so
    conditioning on it leaves almost nothing for talk to explain, and it costs the
    opening round of every game. It is available as a column for robustness checks.

    Two prunes, both necessary. Constant columns carry no information, and linearly
    dependent ones leave the regression rank-deficient - which OLS reports as an
    astronomically wide confidence interval rather than an error. The config set
    contains exact identities by construction (MPCR is the multiplier divided by the
    player count), and dependence is split-specific, so this runs per sample.
    """
    cols = [c for c in CONFIG_COLS + TIMING_COLS + list(extra) if c in df.columns]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return drop_dependent_columns(design(df, cols))


def drop_dependent_columns(X, tol=1e-8):
    """Keep a maximal set of linearly independent columns, via pivoted QR."""
    A = X.to_numpy(dtype=float)
    sd = A.std(axis=0)
    A = (A - A.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
    _, R, pivots = scipy_qr(A, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * max(diag[0], 1.0)))
    return [X.columns[i] for i in sorted(pivots[:rank])]


def design(df, cols):
    """Float design matrix with median imputation for the few missing cells."""
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    return X.fillna(X.median())


def r2(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def cluster_bootstrap(groups, stat, n_boot=N_BOOT):
    """Percentile CI from resampling whole games, not individual rounds.

    Rounds within a game share a group, a treatment, and a conversation, so
    resampling rows would badly understate the uncertainty.
    """
    groups = np.asarray(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    unique = np.array(list(index_of))
    draws = [stat(np.concatenate([index_of[g] for g in
                                  rng.choice(unique, size=len(unique), replace=True)]))
             for _ in range(n_boot)]
    return np.percentile(draws, [2.5, 97.5])


# ----------------------------------------------------------- model zoo ------
# Forest regularization is not optional here. Because folds hold out whole games, a
# default forest (leaf=5, every feature considered per split) memorizes
# game-specific patterns and scores a *negative* cross-validated R² - worse than
# predicting the mean. Leaf size is therefore tuned, and tuned *inside* each
# training fold rather than once on the whole learning split: selecting on the same
# folds whose R² is then reported would bias that R² upward, and the elastic net's
# alpha is already chosen by an inner CV, so a fixed forest would enjoy an
# advantage the linear model does not have.
FOREST_LEAF_GRID = [20, 50, 100, 200]
FOREST_TREES_TUNE = 200      # cheaper forests while comparing candidates
FOREST_TREES_FINAL = 500


def make_model(kind, min_samples_leaf=100, n_estimators=FOREST_TREES_FINAL):
    """One model family, as a fitted-from-scratch pipeline."""
    if kind == "elastic net":
        # alpha and l1_ratio are chosen by an inner CV, so this is already nested.
        estimator = ElasticNetCV(l1_ratio=[0.5, 1.0], n_alphas=30, cv=3,
                                 random_state=SEED, max_iter=5000, n_jobs=-1)
    elif kind == "random forest":
        estimator = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_leaf=min_samples_leaf,
                                          max_features=0.3, random_state=SEED,
                                          n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def fit_tuned(kind, X, y, groups):
    """Fit one model, tuning the forest's leaf size on an inner group split.

    The inner split holds out whole games too, so hyperparameter selection faces
    the same generalization problem the outer folds measure. Returns the fitted
    pipeline and whichever leaf size was chosen (None for the linear model).
    """
    if kind != "random forest":
        return make_model(kind).fit(X, y), None

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(inner.split(X, y, groups=groups))
    best_leaf, best_score = FOREST_LEAF_GRID[0], -np.inf
    for leaf in FOREST_LEAF_GRID:
        candidate = make_model(kind, leaf, FOREST_TREES_TUNE).fit(X.iloc[tr], y[tr])
        score = r2(y[te], np.clip(candidate.predict(X.iloc[te]), *OUTCOME_BOUNDS))
        if score > best_score:
            best_leaf, best_score = leaf, score
    return make_model(kind, best_leaf).fit(X, y), best_leaf


def out_of_fold(kind, df, cols):
    """Out-of-fold predictions with games held out whole, tuning inside each fold.

    Splitting on rows would leak: two rounds of the same game share a group, a
    treatment, and often a conversation, so a row-wise fold would be predicting a
    game partly from itself. The loop is written out rather than delegated to
    `cross_val_predict` so that the inner tuning split can also be group-aware.
    """
    X, y = design(df, cols), df[OUTCOME].to_numpy()
    groups = df[GROUP].to_numpy()
    pred = np.empty(len(y), dtype=float)
    for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        model, _ = fit_tuned(kind, X.iloc[train_idx], y[train_idx], groups[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])
    return np.clip(pred, *OUTCOME_BOUNDS)


def fit_predict_heldout(kind, learn, val, cols):
    """Fit on the whole learning split (tuning within it), then score the held-out."""
    X, y = design(learn, cols), learn[OUTCOME].to_numpy()
    pipe, _ = fit_tuned(kind, X, y, learn[GROUP].to_numpy())
    pred = pipe.predict(design(val, cols)[X.columns])
    return pipe, np.clip(pred, *OUTCOME_BOUNDS)


# ------------------------------------- A. the channel effect, as a coefficient
def channel_effect(splits):
    """Regress contribution on channel availability, controlling for the rest.

    Standard errors are clustered by game, since a game contributes many rounds.
    """
    rows = []
    for split, df in splits.items():
        X = pd.concat([df[CHANNEL].astype(float).rename(CHANNEL),
                       design(df, controls(df))], axis=1)
        model = sm.OLS(df[OUTCOME], sm.add_constant(X)).fit(
            cov_type="cluster", cov_kwds={"groups": df[GROUP]})
        ci = model.conf_int().loc[CHANNEL]
        rows.append({
            "split": split, "n_game_rounds": len(df), "n_games": df[GROUP].nunique(),
            "mean_no_channel": df.loc[df[CHANNEL] == 0, OUTCOME].mean(),
            "mean_channel": df.loc[df[CHANNEL] == 1, OUTCOME].mean(),
            "adj_coef": model.params[CHANNEL], "ci_low": ci[0], "ci_high": ci[1],
            "p_value": model.pvalues[CHANNEL], "model_r2": model.rsquared,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "channel_effect.csv", index=False)
    return out


# ------------------------------------------- B. the nested decomposition ----
def decomposition(learn, val):
    """R² of each nested model, and the ΔR² each block is worth."""
    base = controls(learn)                       # rules + timing
    pre, post = block_features(learn, "pre"), block_features(learn, "post")

    window = block_features(learn, "window")
    specs = {
        "M0 rules + timing": base,
        "M1 + chat channel": base + [CHANNEL],
        "M2 + deliberation (PRE)": base + [CHANNEL] + pre,
        "M3 + reaction (POST)": base + [CHANNEL] + post,
        "M4 + both blocks": base + [CHANNEL] + pre + post,
        "M5 + merged window": base + [CHANNEL] + window,
    }

    rows, preds = [], {}
    for kind in ("elastic net", "random forest"):
        for name, cols in specs.items():
            oof = out_of_fold(kind, learn, cols)
            _, val_pred = fit_predict_heldout(kind, learn, val, cols)
            preds[(kind, name)] = (oof, val_pred)
            y = learn[OUTCOME].to_numpy()
            lo, hi = cluster_bootstrap(
                learn[GROUP], lambda idx, o=oof: r2(y[idx], o[idx]))
            rows.append({"model_family": kind, "model": name, "n_features": len(cols),
                         "cv_r2_learn": r2(y, oof),
                         "cv_r2_ci_low": lo, "cv_r2_ci_high": hi,
                         "r2_heldout_val": r2(val[OUTCOME], val_pred)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES_DIAGNOSTICS / "model_comparison.csv", index=False)

    # Each step measured against the model it should be judged against.
    steps = [("channel", "M0 rules + timing", "M1 + chat channel"),
             ("deliberation (PRE)", "M1 + chat channel", "M2 + deliberation (PRE)"),
             ("reaction (POST)", "M1 + chat channel", "M3 + reaction (POST)"),
             ("both talk blocks", "M1 + chat channel", "M4 + both blocks"),
             ("merged window", "M1 + chat channel", "M5 + merged window")]
    y_learn = learn[OUTCOME].to_numpy()
    y_val = val[OUTCOME].to_numpy()
    deltas = []
    for kind in ("elastic net", "random forest"):
        for label, base_name, plus_name in steps:
            b_oof, b_val = preds[(kind, base_name)]
            p_oof, p_val = preds[(kind, plus_name)]
            lo, hi = cluster_bootstrap(
                learn[GROUP],
                lambda idx, b=b_oof, p=p_oof: r2(y_learn[idx], p[idx]) - r2(y_learn[idx], b[idx]))
            deltas.append({
                "model_family": kind, "step": label,
                "delta_cv_r2": r2(y_learn, p_oof) - r2(y_learn, b_oof),
                "ci_low": lo, "ci_high": hi,
                "delta_r2_heldout": r2(y_val, p_val) - r2(y_val, b_val)})
    decomp = pd.DataFrame(deltas)
    decomp.to_csv(TABLES_DIAGNOSTICS / "variance_decomposition.csv", index=False)
    return comparison, decomp


# ----------------------------------- speaking at all, versus what was said ---
def speech_vs_content(learn, val):
    """Split each talk block into "did they speak" and "what did they say".

    This is the check that decides what the case study is allowed to claim. A talk
    block carries two quite different things: indicators for whether the group used
    an open channel at all, and 140 features describing the conversation when they
    did. Only the second is content, and only the second is what the toolkit is
    for. Entered together, a block can look predictive on the strength of the
    indicators alone - which would be a finding about silence, not about talk.
    """
    base = controls(learn) + [CHANNEL]
    y, y_val = learn[OUTCOME].to_numpy(), val[OUTCOME].to_numpy()
    rows = []

    for block in BLOCKS:
        indicators = [c for c in (f"has_features_{block}", f"chose_silence_{block}")
                      if c in learn.columns and learn[c].nunique() > 1]
        content = block_features(learn, block, with_indicator=False)
        specs = {"base": base,
                 "indicators": base + indicators,
                 "full": base + indicators + content}

        for kind in ("elastic net", "random forest"):
            scores = {}
            for name, cols in specs.items():
                oof = out_of_fold(kind, learn, cols)
                _, val_pred = fit_predict_heldout(kind, learn, val, cols)
                scores[name] = (r2(y, oof), r2(y_val, val_pred), oof)

            for label, lo_key, hi_key in [("spoke at all", "base", "indicators"),
                                          ("what was said", "indicators", "full")]:
                lo_r2, lo_val, lo_oof = scores[lo_key]
                hi_r2, hi_val, hi_oof = scores[hi_key]
                ci_lo, ci_hi = cluster_bootstrap(
                    learn[GROUP],
                    lambda idx, h=hi_oof, l=lo_oof: (r2(y[idx], h[idx])
                                                     - r2(y[idx], l[idx])))
                rows.append({"block": block, "block_meaning": BLOCKS[block],
                             "model_family": kind, "component": label,
                             "n_features": (len(indicators) if label == "spoke at all"
                                            else len(content)),
                             "delta_cv_r2": hi_r2 - lo_r2,
                             "ci_low": ci_lo, "ci_high": ci_hi,
                             "delta_r2_heldout": hi_val - lo_val})

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIAGNOSTICS / "speech_vs_content.csv", index=False)
    return out


# ------------------------------------------ which kind of talk carries it ---
def family_importance(learn, val):
    """Drop one feature family at a time (from both blocks) and watch R² fall.

    Leave-one-family-out rather than family-alone: it asks what a family adds that
    nothing else in the toolkit already captures, which is the question a researcher
    choosing what to measure actually faces. Families overlap, so these do not sum
    to the content term.
    """
    full_cols = controls(learn) + [CHANNEL] + block_features(learn, "window")
    rows = []
    for kind in ("elastic net", "random forest"):
        full_oof = out_of_fold(kind, learn, full_cols)
        _, full_val = fit_predict_heldout(kind, learn, val, full_cols)
        full_r2, full_val_r2 = r2(learn[OUTCOME], full_oof), r2(val[OUTCOME], full_val)

        for fam, cols in families(learn).items():
            reduced = [c for c in full_cols if c not in set(cols)]
            oof = out_of_fold(kind, learn, reduced)
            _, val_pred = fit_predict_heldout(kind, learn, val, reduced)
            rows.append({"model_family": kind, "feature_family": fam,
                         "n_features": len(cols),
                         "drop_in_cv_r2": full_r2 - r2(learn[OUTCOME], oof),
                         "drop_in_heldout_r2": full_val_r2 - r2(val[OUTCOME], val_pred)})
    out = pd.DataFrame(rows).sort_values(["model_family", "drop_in_cv_r2"],
                                         ascending=[True, False])
    out.to_csv(TABLES_DIAGNOSTICS / "family_importance.csv", index=False)
    return out


def family_importance_opening(learn, val, block="post"):
    """Leave-one-family-out, restricted to the cell where content actually predicts.

    The pooled version asks which family matters across every round of every game,
    and answers "none". That is the right answer to that question but the wrong
    question: the content effect lives entirely in the opening rounds, so the
    families are worth re-examining there rather than averaged against thousands of
    rounds where nothing is happening.
    """
    col, _ = STAGINGS[PRIMARY_STAGING]
    flag = f"has_features_{block}"
    l_sub = learn[(learn[col] == "opening") & learn[flag].astype(bool)]
    v_sub = val[(val[col] == "opening") & val[flag].astype(bool)]

    base = controls(l_sub) + [CHANNEL]
    feats = block_features(l_sub, block, with_indicator=False)
    full_cols = base + feats

    rows = []
    for kind in ("elastic net", "random forest"):
        full_oof = out_of_fold(kind, l_sub, full_cols)
        _, full_val = fit_predict_heldout(kind, l_sub, v_sub, full_cols)
        full_r2 = r2(l_sub[OUTCOME], full_oof)
        full_val_r2 = r2(v_sub[OUTCOME], full_val)

        m = manifest()
        m = m[m["kept"]]
        for fam, sub in m.groupby("family"):
            cols = [f"{f}__{block}" for f in sub["feature"]]
            cols = [c for c in cols if c in l_sub.columns]
            if not cols:
                continue
            reduced = [c for c in full_cols if c not in set(cols)]
            oof = out_of_fold(kind, l_sub, reduced)
            _, val_pred = fit_predict_heldout(kind, l_sub, v_sub, reduced)
            rows.append({"model_family": kind, "feature_family": fam,
                         "n_features": len(cols), "n_conversations": len(l_sub),
                         "drop_in_cv_r2": full_r2 - r2(l_sub[OUTCOME], oof),
                         "drop_in_heldout_r2": full_val_r2 - r2(v_sub[OUTCOME], val_pred)})

    out = pd.DataFrame(rows).sort_values(["model_family", "drop_in_cv_r2"],
                                         ascending=[True, False])
    out.to_csv(TABLES_DIAGNOSTICS / "family_importance_opening.csv", index=False)
    return out


# --------------------------------------------------- when does talk matter --
def round_stage(learn, val):
    """Each talk block's ΔR² within each stage, among rounds that had that talk.

    This is the non-parametric version of interacting every feature with time: if
    talk matters more at one point in a game, the block's ΔR² should differ across
    stages. Running it under both staging schemes shows whether any pattern is
    about the clock or about the fraction of the game elapsed.
    """
    rows = []
    for staging, (col, order) in STAGINGS.items():
        for stage in order:
            for block in BLOCKS:
                # Restricted to rounds that actually had this kind of talk. Across
                # all rounds only about a fifth do, so the content question would
                # otherwise be asked mostly of rows whose features are all the same
                # neutral fill - and the answer would be about the fill, not the talk.
                flag = f"has_features_{block}"
                l_sub = learn[(learn[col] == stage) & learn[flag].astype(bool)]
                v_sub = val[(val[col] == stage) & val[flag].astype(bool)]
                if len(l_sub) < 150 or l_sub[GROUP].nunique() < N_FOLDS:
                    continue
                base_cols = controls(l_sub) + [CHANNEL]
                feat_cols = base_cols + block_features(l_sub, block,
                                                       with_indicator=False)
                y = l_sub[OUTCOME].to_numpy()

                y_val = v_sub[OUTCOME].to_numpy()
                for kind in ("elastic net", "random forest"):
                    base = out_of_fold(kind, l_sub, base_cols)
                    full = out_of_fold(kind, l_sub, feat_cols)
                    _, base_val = fit_predict_heldout(kind, l_sub, v_sub, base_cols)
                    _, full_val = fit_predict_heldout(kind, l_sub, v_sub, feat_cols)
                    lo, hi = cluster_bootstrap(
                        l_sub[GROUP],
                        lambda idx, f=full, b=base: (r2(y[idx], f[idx])
                                                     - r2(y[idx], b[idx])))
                    # The held-out estimate gets its own interval, bootstrapped over
                    # the held-out games. Reusing the cross-validated interval would
                    # attach uncertainty from one sample to an estimate from another.
                    lo_v, hi_v = cluster_bootstrap(
                        v_sub[GROUP],
                        lambda idx, f=full_val, b=base_val: (r2(y_val[idx], f[idx])
                                                             - r2(y_val[idx], b[idx])))
                    rows.append({"staging": staging, "stage": stage, "block": block,
                                 "model_family": kind,
                                 "n_conversations": len(l_sub),
                                 "n_conversations_heldout": len(v_sub),
                                 "delta_cv_r2_content": r2(y, full) - r2(y, base),
                                 "ci_low": lo, "ci_high": hi,
                                 "delta_heldout_r2_content": (r2(y_val, full_val)
                                                              - r2(y_val, base_val)),
                                 "ci_low_heldout": lo_v, "ci_high_heldout": hi_v})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "round_stage.csv", index=False)
    return out


# ------------------------------------------- what is actually said, by stage -
def stage_profile(learn):
    """How each toolkit feature differs across stages, in SD units.

    Features are already z-scored against the pooled conversation distribution, so
    a stage mean of +0.3 reads as "this stage runs 0.3 SD above a typical
    conversation on this feature". Only conversations that actually happened are
    included - neutral-filled rounds would drag every stage toward zero.
    """
    col, order = STAGINGS[PRIMARY_STAGING]
    rows = []
    for block in BLOCKS:
        described = learn[learn[f"has_features_{block}"].astype(bool)]
        feats = block_features(learn, block)
        for stage in order:
            sub = described[described[col] == stage]
            if sub.empty:
                continue
            means = sub[feats].mean()
            for feature, value in means.items():
                rows.append({"block": block, "stage": stage,
                             "feature": feature.rsplit("__", 1)[0],
                             "mean_z": value, "n_conversations": len(sub)})
    profile = pd.DataFrame(rows)

    fam = manifest().set_index("feature")["family"].to_dict()
    profile["family"] = profile["feature"].map(fam)

    # How far each feature swings between the first and last stage, which is what
    # makes it interesting to look at rather than merely present.
    wide = profile.pivot_table(index=["block", "feature", "family"],
                               columns="stage", values="mean_z")
    wide["swing"] = wide[order[-1]] - wide[order[0]]
    wide = wide.reset_index().sort_values("swing", key=abs, ascending=False)
    wide.to_csv(TABLES_DIAGNOSTICS / "stage_profile.csv", index=False)
    return wide


def stage_examples(split="learn", per_stage=8, seed=SEED):
    """A sample of real messages from each stage, so the profile can be read.

    Feature names describe a construct; the messages show what the construct looked
    like in this dataset. Sampled at random within stage and block rather than
    hand-picked, so they are representative rather than illustrative.
    """
    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv")
    rounds = pd.read_csv(DATA_PROCESSED / f"rounds_{split}.csv")
    col, order = STAGINGS[PRIMARY_STAGING]

    # A message is grouped by the stage of the round it *predicts*, not the round
    # it was spoken in. POST-block talk comes from the previous round, so the two
    # differ, and the analysis groups by the predicted round - these samples have to
    # match it or they would illustrate a different partition than the results.
    stage_of = rounds.set_index(["gameId", "round_index"])[col]
    chat = chat.join(stage_of.rename("stage"), on=["gameId", "target_round"])
    chat = chat.dropna(subset=["stage"])

    rng_local = np.random.default_rng(seed)
    picks = []
    for block in BLOCKS:
        for stage in order:
            sub = chat[(chat["block"] == block) & (chat["stage"] == stage)]
            if sub.empty:
                continue
            take = sub.iloc[rng_local.choice(len(sub), min(per_stage, len(sub)),
                                             replace=False)]
            picks.append(take.assign(block=block, stage=stage)
                         [["block", "stage", "round_index", "text"]])
    out = pd.concat(picks).reset_index(drop=True)
    out.to_csv(TABLES_DIAGNOSTICS / "stage_examples.csv", index=False)
    return out


# A feature must actually vary within a subsample before a regression on it means
# anything. Testing `std == 0` is not enough: a column that is constant in a stage
# subset still carries floating-point dust from the z-scoring, so its SD comes back
# as something like 7e-18 rather than 0. statsmodels then solves a singular system
# by pseudo-inverse and reports a coefficient of -11.8 with p = 7e-08 - a feature
# with no variance at all looking like the strongest effect in the study. Features
# are z-scored on the pooled distribution, so this floor is in pooled SD units;
# 5% of feature-by-stage combinations fall below 0.35, and only these degenerate
# ones fall below 0.05.
MIN_FEATURE_SD = 0.05
MIN_FEATURE_LEVELS = 3


def _one_feature(df, feature, ctrl):
    """OLS of the outcome on one z-scored feature plus controls, clustered by game.

    Returns (coefficient, p, ci_low, ci_high), or NaNs when the feature does not
    vary enough in this subsample to support a regression.
    """
    x = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    if (len(df) < 30 or x.std() < MIN_FEATURE_SD
            or x.nunique() < MIN_FEATURE_LEVELS):
        return (np.nan,) * 4

    # Re-standardize within this subsample. Features arrive z-scored against the
    # pooled distribution, but a subsample can contain far less variation than the
    # pool: within one stage, `mean_positivity_zscore_conversation` varies by only
    # 0.06 pooled SDs. Reporting its effect "per pooled SD" then extrapolates an
    # order of magnitude beyond any observed value and returns a coefficient of
    # 0.23 on an outcome bounded in [0, 1]. Rescaling makes every coefficient an
    # effect per SD of variation that actually occurs in the data being fitted.
    x = (x - x.mean()) / x.std()
    X = sm.add_constant(pd.concat([x.rename("feature"), design(df, ctrl)], axis=1))
    model = sm.OLS(df[OUTCOME], X).fit(cov_type="cluster",
                                       cov_kwds={"groups": df[GROUP]})
    ci = model.conf_int().loc["feature"]
    return model.params["feature"], model.pvalues["feature"], ci[0], ci[1]


# ------------------------------------ are the same features predictive when? -
def stage_feature_effects(learn, val):
    """Refit every feature's effect separately within each stage.

    The question is not only whether talk predicts contribution, but whether the
    *same* talk predicts it throughout. A feature that helps in the opening and
    hurts in the endgame is a different phenomenon from one that helps throughout,
    and a single pooled coefficient hides both.
    """
    col, order = STAGINGS[PRIMARY_STAGING]
    fam_of = manifest().set_index("feature")["family"].to_dict()
    rows = []

    for block in BLOCKS:
        flag = f"has_features_{block}"
        for stage in order:
            l_sub = learn[(learn[col] == stage) & learn[flag].astype(bool)]
            v_sub = val[(val[col] == stage) & val[flag].astype(bool)]
            if len(l_sub) < 100:
                continue
            ctrl_l, ctrl_v = controls(l_sub), controls(v_sub)

            for feature in block_features(learn, block):
                coef, p_val, lo, hi = _one_feature(l_sub, feature, ctrl_l)
                if np.isnan(coef):
                    continue
                v_coef, v_p, v_lo, v_hi = _one_feature(v_sub, feature, ctrl_v)
                base = feature.rsplit("__", 1)[0]
                rows.append({"block": block, "stage": stage, "feature": base,
                             "family": fam_of.get(base, "Other"),
                             "n_game_rounds": len(l_sub),
                             "coef_learn": coef, "p_learn": p_val,
                             "ci_low_learn": lo, "ci_high_learn": hi,
                             "coef_val": v_coef, "p_val": v_p,
                             "ci_low_val": v_lo, "ci_high_val": v_hi})

    out = pd.DataFrame(rows)
    tested = len(block_features(learn, "pre", with_indicator=False)) * len(order) * 2
    print(f"   (kept {len(out)} of {tested} feature-by-stage regressions; the rest "
          f"had too little variance within their stage to estimate)")
    # FDR is applied within each block-and-stage, since each is its own screen.
    out["q_learn"] = np.nan
    for (b, st), grp in out.groupby(["block", "stage"]):
        out.loc[grp.index, "q_learn"] = multipletests(grp["p_learn"],
                                                      method="fdr_bh")[1]
    out["replicates"] = ((out["q_learn"] < 0.05) & (out["p_val"] < 0.05)
                         & (np.sign(out["coef_learn"]) == np.sign(out["coef_val"])))
    out = out.sort_values(["block", "stage", "p_learn"]).reset_index(drop=True)
    out.to_csv(TABLES / "stage_feature_effects.csv", index=False)
    return out


def stage_agreement(stage_effects):
    """Do the stages agree about which features matter?

    Two summaries per pair of stages: the correlation between their full
    coefficient vectors (do they rank features the same way?) and how many features
    reach p<.05 in both with the same sign (do they agree on the strong ones?).
    A low correlation means "different talk matters at different times"; a high one
    means the pooled estimate was hiding nothing.
    """
    col, order = STAGINGS[PRIMARY_STAGING]
    rows = []
    for block in BLOCKS:
        sub = stage_effects[stage_effects["block"] == block]
        wide = sub.pivot_table(index="feature", columns="stage", values="coef_learn")
        sig = {st: set(g.loc[g["p_learn"] < 0.05, "feature"])
               for st, g in sub.groupby("stage")}
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                if a not in wide.columns or b not in wide.columns:
                    continue
                pair = wide[[a, b]].dropna()
                both = sig.get(a, set()) & sig.get(b, set())
                agree = sum(1 for f in both
                            if np.sign(wide.loc[f, a]) == np.sign(wide.loc[f, b]))
                rows.append({"block": block, "stage_a": a, "stage_b": b,
                             "n_features": len(pair),
                             "coef_correlation": pair[a].corr(pair[b]),
                             "n_sig_a": len(sig.get(a, set())),
                             "n_sig_b": len(sig.get(b, set())),
                             "n_sig_both": len(both),
                             "n_sig_both_same_sign": agree})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIAGNOSTICS / "stage_agreement.csv", index=False)
    return out


# --------------------------------------------- which individual features ----
def feature_effects(learn, val):
    """One regression per feature per block, among rounds that had that kind of talk.

    Features are already z-scored, so a coefficient reads as "change in contribution
    rate per SD", holding rules, timing and momentum fixed. Errors are clustered by
    game. Learn-split p-values get an FDR correction across every feature tested;
    the held-out column is the stronger test, since it uses data that played no part
    in selection.
    """
    fam_of = manifest().set_index("feature")["family"].to_dict()
    rows = []

    for block in BLOCKS:
        flag = f"has_features_{block}"
        l_t = learn[learn[flag].astype(bool)]
        v_t = val[val[flag].astype(bool)]
        l_controls, v_controls = controls(l_t), controls(v_t)

        for feature in block_features(learn, block):
            coef, p, lo, hi = _one_feature(l_t, feature, l_controls)
            if np.isnan(coef):
                continue
            v_coef, v_p, v_lo, v_hi = _one_feature(v_t, feature, v_controls)
            base = feature.rsplit("__", 1)[0]
            rows.append({"feature": base, "block": block,
                         "block_meaning": BLOCKS[block],
                         "family": fam_of.get(base, "Other"),
                         "n_game_rounds": len(l_t),
                         "coef_learn": coef, "p_learn": p,
                         "ci_low_learn": lo, "ci_high_learn": hi,
                         "coef_val": v_coef, "p_val": v_p,
                         "ci_low_val": v_lo, "ci_high_val": v_hi})

    out = pd.DataFrame(rows)
    out["q_learn"] = multipletests(out["p_learn"], method="fdr_bh")[1]
    out["replicates"] = ((out["q_learn"] < 0.05) & (out["p_val"] < 0.05)
                         & (np.sign(out["coef_learn"]) == np.sign(out["coef_val"])))
    out = out.sort_values("p_learn").reset_index(drop=True)
    out.to_csv(TABLES_DIAGNOSTICS / "feature_effects.csv", index=False)
    return out


# ------------------------------------------------------------------- main ---
# Each section, with a rough cost. The expensive ones refit both model families
# inside every cross-validation fold; the cheap ones are regressions or plain
# aggregation over tables that already exist. Being able to re-run one section is
# what makes iterating on a figure practical - a full run is about 40 minutes,
# almost all of it model fitting, and most edits do not invalidate most sections.
SECTIONS = {
    "channel":        "A. channel effect                       (seconds)",
    "decomposition":  "B. nested decomposition                 (~4 min)",
    "speech":         "C. speaking at all vs. what was said    (~6 min)",
    "families":       "D. feature families, pooled             (~5 min)",
    "families_open":  "D. feature families, opening rounds     (~1 min)",
    "stages":         "E. when talk matters                    (~15 min)",
    "profile":        "F. what is said at each stage           (seconds)",
    "stage_features": "G. per-feature effects by stage         (~2 min)",
    "features":       "H. per-feature effects, pooled          (~1 min)",
}


def main(only=None):
    run = set(only) if only else set(SECTIONS)
    learn, val = load("learn"), load("val")
    for name, df in (("learn", learn), ("val", val)):
        print(f"{name}: {len(df)} game-rounds from {df[GROUP].nunique()} games "
              f"({int(df[CHANNEL].sum())} with a channel, "
              f"{int(df.has_features_post.sum())} with previous-round talk)")
    print(f"features per block: "
          f"{len(block_features(learn, 'post', with_indicator=False))}")
    print(f"running: {', '.join(sorted(run))}\n")

    if "channel" in run:
        print("A. Channel effect")
        print(channel_effect({"learn": learn, "val": val}).to_string(index=False), "\n")

    if "decomposition" in run:
        print("B. Nested decomposition")
        comparison, decomp = decomposition(learn, val)
        print(comparison.to_string(index=False), "\n")
        print(decomp.to_string(index=False), "\n")

    if "speech" in run:
        print("C. Speaking at all, versus what was said")
        print(speech_vs_content(learn, val).to_string(index=False), "\n")

    if "families_open" in run:
        print("D. Feature families, opening rounds only")
        print(family_importance_opening(learn, val).to_string(index=False), "\n")

    if "families" in run:
        print("D. Feature families, pooled across all rounds")
        print(family_importance(learn, val).to_string(index=False), "\n")

    if "stages" in run:
        print("E. When in a game does talk matter")
        stages = round_stage(learn, val)
        print(stages[stages.staging == PRIMARY_STAGING].to_string(index=False), "\n")
        print("   robustness, relative staging:")
        print(stages[stages.staging == "relative"].to_string(index=False), "\n")

    if "profile" in run:
        print("F. What is actually said at each stage")
        profile = stage_profile(learn)
        print(profile[profile.block == "post"].head(10).round(3).to_string(index=False))
        print("\n   sample messages:")
        ex = stage_examples()
        for stage in STAGINGS[PRIMARY_STAGING][1]:
            sub = ex[(ex.stage == stage) & (ex.block == "post")].head(3)
            for row in sub.itertuples():
                print(f"     [{stage:<7} r{row.round_index:<2}] {str(row.text)[:78]}")
        print()

    if "stage_features" in run:
        print("G. Are the same features predictive at each stage?")
        stage_effects = stage_feature_effects(learn, val)
        agreement = stage_agreement(stage_effects)
        print(agreement.round(3).to_string(index=False), "\n")

    if "features" in run:
        print("H. Individual features, pooled across stages")
        effects = feature_effects(learn, val)
        cols = ["feature", "block", "family", "coef_learn", "q_learn", "coef_val", "p_val"]
        print(effects.head(10)[cols].to_string(index=False))
        print(f"tested: {len(effects)}; clearing FDR q<.05 on learn: "
              f"{(effects.q_learn < 0.05).sum()}; also holding up on held-out data: "
              f"{effects.replicates.sum()}")

    print("\ntables written to outputs/tables/ and outputs/tables/diagnostics/")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the case study analysis, or one section of it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="sections:\n  " + "\n  ".join(f"{k:<16}{v}" for k, v in SECTIONS.items()))
    ap.add_argument("--only", help="comma-separated section names; default is all")
    args = ap.parse_args()

    chosen = None
    if args.only:
        chosen = [c.strip() for c in args.only.split(",")]
        unknown = [c for c in chosen if c not in SECTIONS]
        if unknown:
            raise SystemExit(f"unknown section(s) {unknown}; "
                             f"choose from {sorted(SECTIONS)}")
    main(chosen)
