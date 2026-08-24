"""Step 4 - decompose contribution into channel, deliberation, reaction, and momentum.

    Which conversation features predict greater contribution in groups that
    communicate, versus groups that do not?

Every game-round is in the sample, whether or not the group had a channel, and the
models are built up one block at a time:

    M0  design parameters + round timing     the rules of the game
    M1  + chat channel indicator             the *mere channel*
    M2  + last round's contribution          momentum: what they were already doing
    M3  + PRE features   (deliberation)      what they said while deciding
    M4  + POST features  (reaction)          what they said about the last result
    M5  + both blocks

M3, M4 and M5 are each compared against M2 rather than chained, so the two talk
blocks are not competing for whichever happens to be entered first.

The order of the channel and momentum is deliberate, and it is not the obvious one.
Contribution is autocorrelated at r = 0.87, so last round's contribution is by far
the strongest single predictor - it alone takes cross-validated R² from 0.08 to
0.76. It is tempting to control for it first and ask what else survives. That would
be a mistake for the channel: the channel was randomized at the *game* level and
raises contribution in every round, so last round's contribution is a **mediator**
of the channel effect rather than a confounder of it. Entering it first blocks the
channel's own causal pathway and shrinks its apparent contribution from about 0.10
to 0.004 - an artifact of over-controlling, not a finding.

So the channel is measured against the game's rules alone, which is valid because
it was randomized and needs no adjustment. Momentum then enters *after* it, and the
talk blocks are judged against that much tougher baseline - which is the right test
for them, since POST-block talk is a reaction to the previous result and could
otherwise look predictive purely by proxying for it.

Follow-ups: which toolkit feature family carries any content term, whether talk
matters more early or late in a game, and which individual features move
contribution - screened with FDR on the learning split and re-tested on held-out data.

Two model families throughout: a penalized linear model and a random forest.
Everything is fit on the learning split; held-out data is scored once, at the end.

Run:  python scripts/04_analysis.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr as scipy_qr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from config import CONFIG_COLS, DATA_PROCESSED, SEED, TABLES, TIMING_COLS

OUTCOME = "contribution_rate"
GROUP = "gameId"          # rounds within a game are not independent observations
CHANNEL = "has_chat_channel"
MOMENTUM = "lagged_contribution"
BLOCKS = {"pre": "deliberation", "post": "reaction"}
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


def block_features(df, block):
    """The kept toolkit features for one talk block, as they appear in the table."""
    m = manifest()
    cols = [f"{f}__{block}" for f in m.loc[m["kept"], "feature"]]
    return [c for c in cols if c in df.columns]


def families(df):
    """{family: [feature columns across both blocks]}."""
    m = manifest()
    m = m[m["kept"]]
    out = {}
    for fam, sub in m.groupby("family"):
        cols = [f"{f}__{b}" for f in sub["feature"] for b in BLOCKS]
        out[fam] = [c for c in cols if c in df.columns]
    return out


def controls(df, extra=()):
    """Everything known before this round's talk: rules, timing, and momentum.

    Two prunes, both necessary. Constant columns carry no information, and linearly
    dependent ones leave the regression rank-deficient - which OLS reports as an
    astronomically wide confidence interval rather than an error. The config set
    contains exact identities by construction (MPCR is the multiplier divided by the
    player count), and dependence is split-specific, so this runs per sample.
    """
    cols = [c for c in CONFIG_COLS + TIMING_COLS + [MOMENTUM] + list(extra)
            if c in df.columns]
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
def make_model(kind):
    """The two model families, behind one interface."""
    if kind == "elastic net":
        estimator = ElasticNetCV(l1_ratio=[0.5, 1.0], n_alphas=30, cv=3,
                                 random_state=SEED, max_iter=5000, n_jobs=-1)
    elif kind == "random forest":
        # Heavily regularized on purpose. Because folds hold out whole games, a
        # default forest (leaf=5, all features per split) memorizes game-specific
        # patterns and scores a *negative* cross-validated R² - worse than
        # predicting the mean. Large leaves and a feature subsample fix that: on
        # learning-split CV, leaf=5 scores -0.10 and leaf=100 scores +0.21.
        # Chosen on the learning split alone.
        estimator = RandomForestRegressor(n_estimators=500, min_samples_leaf=100,
                                          max_features=0.3, random_state=SEED,
                                          n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def out_of_fold(kind, df, cols):
    """Out-of-fold predictions with games held out whole.

    Splitting on rows would leak: two rounds of the same game share a group, a
    treatment, and often a conversation, so a row-wise fold would be predicting a
    game partly from itself.
    """
    X, y = design(df, cols), df[OUTCOME].to_numpy()
    pred = cross_val_predict(make_model(kind), X, y,
                             cv=GroupKFold(n_splits=N_FOLDS), groups=df[GROUP])
    return np.clip(pred, *OUTCOME_BOUNDS)


def fit_predict_heldout(kind, learn, val, cols):
    X, y = design(learn, cols), learn[OUTCOME].to_numpy()
    pipe = make_model(kind).fit(X, y)
    pred = pipe.predict(design(val, cols)[X.columns])
    return pipe, np.clip(pred, *OUTCOME_BOUNDS)


# ------------------------------------- A. the channel effect, as a coefficient
def channel_effect(splits):
    """Regress contribution on channel availability, controlling for the rest.

    Standard errors are clustered by game, since a game contributes many rounds.
    """
    rows = []
    for split, df in splits.items():
        # Momentum is excluded here for the same reason it is entered after the
        # channel in the decomposition: it is a mediator of the channel effect, and
        # adjusting for it would report the channel's direct effect only.
        ctrl = [c for c in controls(df) if c != MOMENTUM]
        X = pd.concat([df[CHANNEL].astype(float).rename(CHANNEL),
                       design(df, ctrl)], axis=1)
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
    base = controls(learn)                       # rules + timing + momentum
    rules = [c for c in base if c != MOMENTUM]
    pre, post = block_features(learn, "pre"), block_features(learn, "post")

    specs = {
        "M0 rules + timing": rules,
        "M1 + chat channel": rules + [CHANNEL],
        "M2 + momentum": base + [CHANNEL],
        "M3 + deliberation (PRE)": base + [CHANNEL] + pre,
        "M4 + reaction (POST)": base + [CHANNEL] + post,
        "M5 + both blocks": base + [CHANNEL] + pre + post,
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
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)

    # Each step measured against the model it should be judged against.
    steps = [("channel", "M0 rules + timing", "M1 + chat channel"),
             ("momentum", "M1 + chat channel", "M2 + momentum"),
             ("deliberation (PRE)", "M2 + momentum", "M3 + deliberation (PRE)"),
             ("reaction (POST)", "M2 + momentum", "M4 + reaction (POST)"),
             ("both talk blocks", "M2 + momentum", "M5 + both blocks")]
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
    decomp.to_csv(TABLES / "variance_decomposition.csv", index=False)
    return comparison, decomp


# ------------------------------------------ which kind of talk carries it ---
def family_importance(learn, val):
    """Drop one feature family at a time (from both blocks) and watch R² fall.

    Leave-one-family-out rather than family-alone: it asks what a family adds that
    nothing else in the toolkit already captures, which is the question a researcher
    choosing what to measure actually faces. Families overlap, so these do not sum
    to the content term.
    """
    full_cols = (controls(learn) + [CHANNEL]
                 + block_features(learn, "pre") + block_features(learn, "post"))
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
    out.to_csv(TABLES / "family_importance.csv", index=False)
    return out


# --------------------------------------------------- when does talk matter --
def round_stage(learn, val):
    """Each talk block's contribution, recomputed within thirds of a game.

    This is the non-parametric version of interacting every feature with time: if
    talk matters more at one point in a game, the block's ΔR² should differ across
    the three stages.
    """
    rows = []
    for stage, lo_q, hi_q in [("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3),
                              ("late", 2 / 3, 1.01)]:
        l_sub = learn[(learn["round_position"] >= lo_q) & (learn["round_position"] < hi_q)]
        v_sub = val[(val["round_position"] >= lo_q) & (val["round_position"] < hi_q)]
        base_cols = controls(l_sub) + [CHANNEL]
        y = l_sub[OUTCOME].to_numpy()

        for kind in ("elastic net", "random forest"):
            base = out_of_fold(kind, l_sub, base_cols)
            _, base_val = fit_predict_heldout(kind, l_sub, v_sub, base_cols)
            base_val_r2 = r2(v_sub[OUTCOME], base_val)
            for block in BLOCKS:
                cols = base_cols + block_features(l_sub, block)
                full = out_of_fold(kind, l_sub, cols)
                _, full_val = fit_predict_heldout(kind, l_sub, v_sub, cols)
                lo, hi = cluster_bootstrap(
                    l_sub[GROUP],
                    lambda idx, f=full, b=base: r2(y[idx], f[idx]) - r2(y[idx], b[idx]))
                rows.append({"stage": stage, "block": block,
                             "model_family": kind, "n_game_rounds": len(l_sub),
                             "delta_cv_r2_content": r2(y, full) - r2(y, base),
                             "ci_low": lo, "ci_high": hi,
                             "delta_heldout_r2_content": (r2(v_sub[OUTCOME], full_val)
                                                          - base_val_r2)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "round_stage.csv", index=False)
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

        def fit_one(df, feature, ctrl):
            x = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
            if x.std() == 0:
                return (np.nan,) * 4
            X = sm.add_constant(pd.concat([x.rename("feature"),
                                           design(df, ctrl)], axis=1))
            model = sm.OLS(df[OUTCOME], X).fit(cov_type="cluster",
                                               cov_kwds={"groups": df[GROUP]})
            ci = model.conf_int().loc["feature"]
            return model.params["feature"], model.pvalues["feature"], ci[0], ci[1]

        for feature in block_features(learn, block):
            coef, p, lo, hi = fit_one(l_t, feature, l_controls)
            if np.isnan(coef):
                continue
            v_coef, v_p, v_lo, v_hi = fit_one(v_t, feature, v_controls)
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
    out.to_csv(TABLES / "feature_effects.csv", index=False)
    return out


# ------------------------------------------------------------------- main ---
def main():
    learn, val = load("learn"), load("val")
    for name, df in (("learn", learn), ("val", val)):
        print(f"{name}: {len(df)} game-rounds from {df[GROUP].nunique()} games "
              f"({int(df[CHANNEL].sum())} with a channel, "
              f"{int(df.has_features_pre.sum())} with deliberation talk, "
              f"{int(df.has_features_post.sum())} with reaction talk)")
    print(f"features per block: {len(block_features(learn, 'pre'))}\n")

    print("A. Channel effect")
    print(channel_effect({"learn": learn, "val": val}).to_string(index=False), "\n")

    print("B. Nested decomposition")
    comparison, decomp = decomposition(learn, val)
    print(comparison.to_string(index=False), "\n")
    print(decomp.to_string(index=False), "\n")

    print("C. Which kind of talk carries any content term")
    print(family_importance(learn, val).to_string(index=False), "\n")

    print("D. When in a game does talk matter")
    print(round_stage(learn, val).to_string(index=False), "\n")

    print("E. Individual features (top 12 by learn-split p-value)")
    effects = feature_effects(learn, val)
    cols = ["feature", "block", "family", "coef_learn", "q_learn", "coef_val", "p_val"]
    print(effects.head(12)[cols].to_string(index=False))
    print(f"tested: {len(effects)}; clearing FDR q<.05 on learn: "
          f"{(effects.q_learn < 0.05).sum()}; also holding up on held-out data: "
          f"{effects.replicates.sum()}")
    print("\ntables written to outputs/tables/")


if __name__ == "__main__":
    main()
