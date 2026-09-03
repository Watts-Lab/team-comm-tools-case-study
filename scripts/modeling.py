"""Modeling helpers shared by the new window and first-N analyses (steps 11-12).

Lifted verbatim in behaviour from scripts/04_analysis.py so that the new analyses
are comparable with the published ones: same model families, same group-aware
cross-validation, same clipping, same cluster bootstrap, same single-feature
regression. ``04_analysis.py`` is deliberately left untouched - it produces the
tables the paper already cites - so the duplication here is the price of not
disturbing a published result.

The one thing this module adds is ``n_games`` reporting: every result the new
analyses emit carries the number of games behind it, because rounds within a game
are not independent and a cell with 200 rounds from 9 games is a much weaker claim
than one with 200 rounds from 90.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr as scipy_qr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG_COLS, SEED, TIMING_COLS

OUTCOME = "contribution_rate"
GROUP = "gameId"
CHANNEL = "has_chat_channel"
MOMENTUM = "lagged_contribution"

N_FOLDS = 10
N_BOOT = 2000
OUTCOME_BOUNDS = (0.0, 1.0)

FOREST_LEAF_GRID = [20, 50, 100, 200]
FOREST_TREES_TUNE = 200
FOREST_TREES_FINAL = 500
MODEL_KINDS = ("elastic net", "random forest")

# A cell needs enough independent clusters before a group-held-out CV means
# anything: GroupKFold cannot make ten folds from nine games, and a bootstrap over
# a handful of games has essentially no resolution. Cells below these are skipped
# and reported as skipped rather than silently dropped.
MIN_ROWS = 60
MIN_GAMES = 20

MIN_FEATURE_SD = 0.05
MIN_FEATURE_LEVELS = 3

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------- design -------
def design(df, cols):
    """Float design matrix with median imputation for the few missing cells."""
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    return X.fillna(X.median())


def drop_dependent_columns(X, tol=1e-8):
    """Keep a maximal set of linearly independent columns, via pivoted QR."""
    A = X.to_numpy(dtype=float)
    sd = A.std(axis=0)
    A = (A - A.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
    _, R, pivots = scipy_qr(A, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * max(diag[0], 1.0)))
    return [X.columns[i] for i in sorted(pivots[:rank])]


def controls(df, extra=(), timing=True):
    """The game's rules and (optionally) the clock: everything known before talk.

    Momentum is not included by default, matching 04_analysis.py - it predicts the
    outcome better than everything else combined and leaves nothing for talk to
    explain. Pass it through ``extra`` for a robustness check.
    """
    cols = [c for c in CONFIG_COLS + (TIMING_COLS if timing else []) + list(extra)
            if c in df.columns]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    if not cols:
        return []
    return drop_dependent_columns(design(df, cols))


def r2(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def cluster_bootstrap(groups, stat, n_boot=N_BOOT):
    """Percentile CI from resampling whole games, not individual rounds."""
    groups = np.asarray(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    unique = np.array(list(index_of), dtype=object)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(unique), size=len(unique), replace=True)
        idx = np.concatenate([index_of[unique[i]] for i in pick])
        try:
            draws.append(stat(idx))
        except Exception:
            continue
    if not draws:
        return np.nan, np.nan
    return tuple(np.percentile(draws, [2.5, 97.5]))


# -------------------------------------------------------------- models ------
def make_model(kind, min_samples_leaf=100, n_estimators=FOREST_TREES_FINAL):
    if kind == "elastic net":
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
    """Fit one model, tuning the forest's leaf size on an inner group-aware split."""
    if kind != "random forest":
        return make_model(kind).fit(X, y), None
    if pd.Series(groups).nunique() < 4:
        return make_model(kind).fit(X, y), None
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(inner.split(X, y, groups=groups))
    best_leaf, best_score = FOREST_LEAF_GRID[0], -np.inf
    for leaf in FOREST_LEAF_GRID:
        cand = make_model(kind, leaf, FOREST_TREES_TUNE).fit(X.iloc[tr], y[tr])
        score = r2(y[te], np.clip(cand.predict(X.iloc[te]), *OUTCOME_BOUNDS))
        if score > best_score:
            best_leaf, best_score = leaf, score
    return make_model(kind, best_leaf).fit(X, y), best_leaf


def out_of_fold(kind, df, cols, n_folds=N_FOLDS, group_col=GROUP):
    """Out-of-fold predictions with whole games held out, tuning inside each fold.

    When ``group_col`` is None the unit of analysis is already the game (the
    first-N analysis) and a plain KFold is the group-aware split.
    """
    X, y = design(df, cols), df[OUTCOME].to_numpy(dtype=float)
    if group_col is None:
        groups = np.arange(len(df))
        splitter = KFold(n_splits=min(n_folds, len(df)), shuffle=True,
                         random_state=SEED).split(X)
    else:
        groups = df[group_col].to_numpy()
        n = min(n_folds, pd.Series(groups).nunique())
        splitter = GroupKFold(n_splits=n).split(X, y, groups)
    pred = np.empty(len(y), dtype=float)
    for train_idx, test_idx in splitter:
        model, _ = fit_tuned(kind, X.iloc[train_idx], y[train_idx],
                             groups[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])
    return np.clip(pred, *OUTCOME_BOUNDS)


def fit_predict_heldout(kind, learn, val, cols, group_col=GROUP):
    """Fit on the learning sample (tuning within it), then score the held-out one."""
    X, y = design(learn, cols), learn[OUTCOME].to_numpy(dtype=float)
    groups = (np.arange(len(learn)) if group_col is None
              else learn[group_col].to_numpy())
    pipe, _ = fit_tuned(kind, X, y, groups)
    pred = pipe.predict(design(val, cols)[X.columns])
    return pipe, np.clip(pred, *OUTCOME_BOUNDS)


def delta_r2(kind, learn, val, base_cols, full_cols, group_col=GROUP,
             n_folds=N_FOLDS):
    """How much a feature block adds, cross-validated and on held-out data.

    Returns a dict with both estimates and a cluster-bootstrap interval for each,
    bootstrapped over the sample the estimate came from.
    """
    y = learn[OUTCOME].to_numpy(dtype=float)
    base = out_of_fold(kind, learn, base_cols, n_folds, group_col)
    full = out_of_fold(kind, learn, full_cols, n_folds, group_col)
    groups = (np.arange(len(learn)) if group_col is None
              else learn[group_col].to_numpy())
    lo, hi = cluster_bootstrap(
        groups, lambda i, f=full, b=base: r2(y[i], f[i]) - r2(y[i], b[i]))

    out = {"delta_cv_r2": r2(y, full) - r2(y, base),
           "ci_low": lo, "ci_high": hi,
           "cv_r2_base": r2(y, base), "cv_r2_full": r2(y, full)}

    if val is not None and len(val):
        yv = val[OUTCOME].to_numpy(dtype=float)
        _, bv = fit_predict_heldout(kind, learn, val, base_cols, group_col)
        _, fv = fit_predict_heldout(kind, learn, val, full_cols, group_col)
        vgroups = (np.arange(len(val)) if group_col is None
                   else val[group_col].to_numpy())
        lo_v, hi_v = cluster_bootstrap(
            vgroups, lambda i, f=fv, b=bv: r2(yv[i], f[i]) - r2(yv[i], b[i]))
        out.update({"delta_heldout_r2": r2(yv, fv) - r2(yv, bv),
                    "ci_low_heldout": lo_v, "ci_high_heldout": hi_v,
                    "heldout_r2_base": r2(yv, bv), "heldout_r2_full": r2(yv, fv)})
    return out


# --------------------------------------------------- one feature at a time --
def one_feature(df, feature, ctrl, group_col=GROUP):
    """OLS of the outcome on one feature plus controls, clustered by game.

    The feature is re-standardized within this subsample, so the coefficient is an
    effect per SD of variation that actually occurs in the rows being fitted -
    never an extrapolation from the pooled distribution. Returns NaNs when the
    feature does not vary enough here to support a regression.
    """
    x = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    if (len(df) < 30 or x.std() < MIN_FEATURE_SD
            or x.nunique() < MIN_FEATURE_LEVELS):
        return (np.nan,) * 4
    x = (x - x.mean()) / x.std()
    X = sm.add_constant(pd.concat([x.rename("feature"), design(df, ctrl)], axis=1))
    fit_kw = ({"cov_type": "cluster", "cov_kwds": {"groups": df[group_col]}}
              if group_col else {"cov_type": "HC1"})
    try:
        model = sm.OLS(df[OUTCOME].astype(float), X).fit(**fit_kw)
    except Exception:
        return (np.nan,) * 4
    ci = model.conf_int().loc["feature"]
    return model.params["feature"], model.pvalues["feature"], ci[0], ci[1]


# ------------------------------------------------------------- loading ------
def to_numeric_bools(df):
    """Turn the CSV's 'True'/'False' strings back into 1.0/0.0."""
    for col in df.columns:
        if df[col].dtype == object:
            lowered = df[col].astype(str).str.lower()
            if lowered.isin({"true", "false", "nan"}).all():
                df[col] = lowered.map({"true": 1.0, "false": 0.0})
    return df


def manifest(tables_dir):
    return pd.read_csv(tables_dir / "feature_manifest.csv")


def kept_features(tables_dir):
    m = manifest(tables_dir)
    return m.loc[m["kept"], "feature"].tolist()


def block_features(df, block, tables_dir, with_indicator=True):
    """The kept toolkit features for one talk block, as they appear in the table."""
    cols = [f"{f}__{block}" for f in kept_features(tables_dir)]
    cols = [c for c in cols if c in df.columns]
    if with_indicator:
        for flag in (f"has_features_{block}", f"chose_silence_{block}"):
            if flag in df.columns and df[flag].nunique() > 1:
                cols.append(flag)
    return cols


def cell_size(df, group_col=GROUP):
    """(rows, games) - reported on every result so clustering stays visible."""
    return len(df), (df[group_col].nunique() if group_col else len(df))


def big_enough(df, group_col=GROUP, min_rows=MIN_ROWS, min_games=MIN_GAMES):
    rows, games = cell_size(df, group_col)
    return rows >= min_rows and games >= min_games
