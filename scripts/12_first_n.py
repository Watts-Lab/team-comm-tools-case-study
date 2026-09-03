"""Step 12 - does a group's opening talk predict how it ends?

    Does what a group says in its first N rounds predict how much it contributes
    in its FINAL round?

Everything else in this repository asks about one contribution decision at a time,
and pays for it with a clustering problem: a 30-round game contributes sixty
game-rounds that share a group, a treatment and often a conversation, so every
standard error has to be clustered and every fold has to hold out whole games.

**This analysis has no clustering problem at all.** The unit is the game. One row
per game, one outcome per row: the group's contribution in the last round it
played, predicted from the conversation it had in its first N rounds. Rounds within
a game are not independent, but games are - they are separate groups of people
playing separately - so a plain KFold *is* the group-aware split and HC1 *is* the
right standard error. The ``modeling`` helpers take ``group_col=None`` for exactly
this case. What the design buys in cleanliness it pays for in size: a few hundred
game-rounds become a few dozen games, which is the honest limit of what follows.

The question is also a different question. "Talk in round k predicts contribution
in round k+1" is a claim about local influence. This is a claim about *durability*:
whether an opening is diagnostic of an endpoint ten or twenty rounds later, long
after the conversation itself is over.

Restricted to long games (default: at least 15 rounds played), so that the final
round is genuinely far from the opening. N runs 1 through 10; round indices start
at 0, so "the first N rounds" is ``round_index <= N``, i.e. N+1 rounds of play.
N is always well short of the final round, so no talk used as a predictor comes
from the round being predicted.

Two ways to summarize the opening, run as separate specifications and carried in a
``summary`` column, because they are genuinely different objects:

  mean_of_rounds  average the per-round ``__window`` features over the first-N
                  rounds that actually had talk. The mean of eight conversations.
  cumulative      the ``__cumulative`` features of the single cumulative conversation at
                  round N - all the first-N talk featurized as ONE conversation.

The mean of eight conversations' discursive diversity is not the diversity of the
eight concatenated: the first is a within-conversation average, the second sees the
whole opening as one stretch of talk and can notice a group that changed the
subject between rounds. Neither is the "right" one, so both are reported.
The cumulative half depends on steps 8-10 and is skipped, loudly, until they land.

Controls are the game's randomized design parameters plus how long the game ran
(``n_rounds_played`` / ``last_round``) and its mean active headcount. The timing
block is deliberately **not** here: the outcome is always the last round, so
``round_index``, ``rounds_remaining`` and ``round_position`` are constant by
construction within a cell and would only add rank-deficiency.

Every specification is run twice, carried in a ``baseline`` column:

  no_baseline           controls only.
  early_contribution    controls PLUS the group's mean contribution over the
                        first N rounds.

The second is the one that decides what may be claimed. Contribution is strongly
autocorrelated, and groups that talk are groups that cooperate. Without the early
contribution in the model, a talk feature can look predictive purely because
talkative groups were already cooperative in round 1 and simply stayed that way -
the finding would be about the *group*, not about what it said. A feature that
survives the ``early_contribution`` baseline is saying something the group's own
opening behaviour did not already say; one that does not survive is a description
of cooperative groups, not a prediction from talk.

**The final round is not a well-behaved continuous outcome.** In the analysed
subsample about half the games sit exactly at the ceiling: every player contributes
their whole endowment in the last round, so ``contribution_rate`` is 1.0 and the
distribution is a point mass at the boundary with a left tail hanging off it. Half
the "variance" a linear model is being scored on is therefore not interval-scaled
variance at all, and an R² against it is close to meaningless - which is a second,
independent reason the ΔR² comes back negative, on top of 151 features fitted to a
few dozen games. **The honest reading of the continuous ΔR² table is a statement
about power and outcome shape, not about talk.** ``first_n_sample.csv`` reports
``frac_at_ceiling`` and ``frac_above_0.9`` per cell so this is visible before any
result is.

So every specification is also run against a companion outcome, carried in an
``outcome`` column:

  contribution_rate   the group's mean final-round contribution. Continuous,
                      scored by R², and subject to the ceiling above.
  maxed_out           1 if that rate is >= 0.999, i.e. the group went all in.
                      The boundary atom turned into the thing being predicted
                      instead of the thing wrecking the scale.

The binary version is fitted with logistic models rather than OLS: per-feature
coefficients are **log-odds per SD** (flagged in the ``coef_units`` column), and
the block-level fit is scored by **AUC from out-of-fold predictions** rather than
R², with a ``metric`` column saying which of ``delta_r2`` / ``delta_auc`` a row
carries. ``modeling.py`` is R²-oriented and is not edited, so the classification
path lives in this script and deliberately mirrors it fold for fold.

**This analysis is small, and the tables say so on every row.** With a 15-round
minimum roughly 70-85 learning games and 105-165 held-out games have any talk in
their first N rounds, against 151 candidate features. Cells below 40 games are
skipped and recorded as skipped in ``first_n_sample.csv`` rather than dropped, the
sample table is written before any model runs, and the final printed summary states
how many features were screened, how many would be expected significant by chance,
how many cleared FDR and how many replicated - so that a null reads as a null.

Outputs (into outputs/tables/windows/):
  * ``first_n_sample.csv``          - the power inventory, including how much of
                                      the outcome is stacked on the ceiling.
                                      Read this first.
  * ``first_n_delta_r2.csv``        - what the opening talk adds over controls,
                                      as ΔR² and as ΔAUC.
  * ``first_n_feature_effects.csv`` - long, one row per feature per specification,
                                      for both outcomes.
  * ``first_n_robustness.csv``      - the same deltas at 12-, 15- and 20-round cuts.

Run:  python scripts/12_first_n.py --summary mean_of_rounds
      python scripts/12_first_n.py --quick --n-max 2 --summary mean_of_rounds
"""

import os

# A long featurization job (step 9) is usually running alongside this one, and both
# ElasticNetCV and RandomForestRegressor are configured with n_jobs=-1 in
# modeling.py. Capping the worker pool here - before sklearn/joblib are imported -
# keeps this analysis from starving that job. The data are tiny; the parallelism
# was never buying much.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import modeling  # noqa: E402
from config import CONFIG_COLS, DATA_PROCESSED, TABLES, TABLES_WINDOWS  # noqa: E402

# Both a penalized fit on 151 features over ~70 games and a logistic fit near
# separation warn on essentially every call. Both are *handled* - the coordinate
# descent still returns the penalized solution, and one_feature_logit checks
# convergence and returns NaN - so the warnings are noise that would bury the
# progress lines windows_status.sh tails. Only these two are silenced.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUTCOME = modeling.OUTCOME
GROUP = "gameId"

# The two summaries, and the feature block each one reads.
SUMMARIES = {"mean_of_rounds": "window", "cumulative": "cumulative"}

BASELINES = ("no_baseline", "early_contribution")
EARLY_CONTRIB = "early_contribution_rate"

# The continuous outcome and its ceiling-aware companion. 0.999 rather than 1.0
# because the rate is a mean of integer contributions over an endowment of 20 and
# arrives through a CSV round-trip; nothing real sits between 0.999 and 1.0.
CONTINUOUS, BINARY = "contribution_rate", "maxed_out"
OUTCOMES = (CONTINUOUS, BINARY)
CEILING = 0.999

# Binary cells need both classes on both sides before an AUC means anything.
MIN_PER_CLASS = 8

# Long games only: the point is that the final round is far from the opening.
DEFAULT_MIN_ROUNDS = 15
ROBUSTNESS_MIN_ROUNDS = (12, 15, 20)
DEFAULT_N_MAX = 10

# A cell needs this many games before a cross-validated R² over 151 features means
# anything. Cells below it are skipped and the skip is recorded, never dropped.
MIN_GAMES_CELL = 40

QUICK_FOLDS = 5
QUICK_BOOT = 200

# Folds for the binary outcome are stratified. Pooling out-of-fold probabilities
# across unstratified folds manufactures AUC out of nothing: a test fold that
# happens to be positive-heavy is predicted by a training set that is
# correspondingly positive-light, so the whole fold's scores shift together and the
# pooled ranking picks it up. On simulated null data with n=71 that artifact moved
# the *absolute* AUC between 0.43 and 0.78. Stratifying removes the balance shift;
# the residual instability is why only the **difference** between the full and base
# models is read as a result here, and why both absolute AUCs are reported next to
# it. The two models share their folds, so the artifact largely cancels in ΔAUC.
CLF_FOLDS = 5


# ------------------------------------------------------------- loading ------
def load(split, summary):
    """The analysis table one summary needs, or None if it has not been built yet."""
    name = ("analysis" if summary == "mean_of_rounds" else "analysis_windows")
    path = DATA_PROCESSED / f"{name}_{split}.csv"
    if not path.exists():
        return None
    df = modeling.to_numeric_bools(pd.read_csv(path, low_memory=False))
    df[GROUP] = df[GROUP].astype(str)
    return df


# --------------------------------------------------- one row per game -------
def game_frame(df, n, summary, min_rounds):
    """Collapse the game-round table to one row per game.

    Returns (frame, feature_cols, n_eligible). ``n_eligible`` is every long-enough
    game, whether or not it talked, so that the sample table can report the talk
    rate rather than only the surviving count.
    """
    block = SUMMARIES[summary]
    long_games = df[df["n_rounds_played"] >= min_rounds]
    n_eligible = long_games[GROUP].nunique()
    if not len(long_games):
        return None, [], 0

    # The predictors must come from strictly before the round being predicted.
    # min_rounds > n + 1 guarantees it for every game in the cell; the caller keeps
    # n well under the cut, and this makes the guarantee an error rather than a
    # silent overlap if that ever stops being true.
    if n >= long_games["last_round"].min():
        raise ValueError(f"N={n} reaches the final round of the shortest game in "
                         f"the cell (last_round={long_games['last_round'].min()})")

    final = long_games[long_games["rounds_remaining"] == 0].set_index(GROUP)
    early = long_games[long_games["round_index"] <= n]

    feature_cols = modeling.block_features(df, block, TABLES, with_indicator=False)
    talk_flag = f"has_features_{block}"

    if summary == "mean_of_rounds":
        # Average over the first-N rounds that actually had talk. Rounds with no
        # conversation carry the neutral fill on all 151 columns (step 3), so
        # averaging them in would shrink every opening toward the grand mean in
        # proportion to how silent the group was - which is an indicator, not a
        # description of what was said.
        spoke = early[early[talk_flag].astype(bool)]
        feats = spoke.groupby(GROUP)[feature_cols].mean()
    else:
        # The cumulative conversation *at* round N: one row, already the whole
        # opening featurized as a single conversation. No averaging to do.
        at_n = early[early["round_index"] == n]
        spoke = at_n[at_n[talk_flag].astype(bool)]
        feats = spoke.set_index(GROUP)[feature_cols]

    ctrl_cols = [c for c in CONFIG_COLS if c in long_games.columns]
    per_game = final[ctrl_cols + ["n_rounds_played", "last_round", OUTCOME]].copy()
    per_game["mean_n_players_active"] = long_games.groupby(GROUP)["n_players_active"].mean()
    per_game[EARLY_CONTRIB] = early.groupby(GROUP)[OUTCOME].mean()

    frame = per_game.join(feats, how="inner").reset_index()
    # The ceiling-aware companion outcome: did the group go all in on its last
    # round? This is the boundary atom promoted to the question, rather than left
    # to sit on the edge of a scale a linear model is trying to score.
    frame[BINARY] = (frame[CONTINUOUS] >= CEILING).astype(int)
    return frame, feature_cols, n_eligible


def control_cols(frame, baseline):
    """Design parameters, game length, headcount - and optionally early behaviour.

    ``timing=False``: the outcome is always the last round, so every timing column
    is constant here by construction.
    """
    extra = ["n_rounds_played", "last_round", "mean_n_players_active"]
    if baseline == "early_contribution":
        extra.append(EARLY_CONTRIB)
    return modeling.controls(frame, extra=extra, timing=False)


# -------------------------------------- the ceiling-aware (binary) path -----
# modeling.py is written around a continuous outcome and an R² score, and it is not
# editable here, so the classification path is written out below. It mirrors
# modeling.py deliberately - same two model families, same leaf-size grid, same
# tuning-inside-the-fold discipline, same percentile bootstrap - so that a ΔAUC row
# and a ΔR² row describe the same procedure applied to two readings of the same
# final round, and any difference between them is about the outcome rather than
# about the machinery.


def make_classifier(kind, min_samples_leaf=20, n_estimators=modeling.FOREST_TREES_FINAL):
    """The classification twin of modeling.make_model."""
    if kind == "elastic net":
        estimator = LogisticRegressionCV(
            Cs=6, cv=3, penalty="elasticnet", solver="saga", l1_ratios=[0.5, 1.0],
            max_iter=2000, scoring="roc_auc", random_state=modeling.SEED, n_jobs=1)
    elif kind == "random forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features=0.3, random_state=modeling.SEED, n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


# Leaf sizes for the classifier start smaller than modeling.FOREST_LEAF_GRID: a
# leaf of 200 on a 70-game training fold is the whole sample, so every grid point
# would return the base rate and the "tuning" would be a coin flip.
CLF_LEAF_GRID = [5, 10, 20, 40]


def fit_tuned_clf(kind, X, y):
    """Fit one classifier, tuning the forest's leaf size on an inner stratified split."""
    if kind != "random forest":
        return make_classifier(kind).fit(X, y)
    if min(np.bincount(y, minlength=2)) < 2 * MIN_PER_CLASS:
        return make_classifier(kind).fit(X, y)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y,
                              random_state=modeling.SEED)
    best_leaf, best_score = CLF_LEAF_GRID[0], -np.inf
    for leaf in CLF_LEAF_GRID:
        cand = make_classifier(kind, leaf, modeling.FOREST_TREES_TUNE).fit(X.iloc[tr],
                                                                          y[tr])
        score = safe_auc(y[te], cand.predict_proba(X.iloc[te])[:, 1])
        if not np.isnan(score) and score > best_score:
            best_leaf, best_score = leaf, score
    return make_classifier(kind, best_leaf).fit(X, y)


def safe_auc(y, score):
    """AUC, or NaN when the sample is single-class and the question is undefined."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return roc_auc_score(y, score)
    except ValueError:
        return np.nan


def out_of_fold_proba(kind, df, cols, n_folds=CLF_FOLDS):
    """Out-of-fold P(maxed out), on stratified folds. See CLF_FOLDS for why."""
    X, y = modeling.design(df, cols), df[BINARY].to_numpy(dtype=int)
    n_splits = max(2, min(n_folds, int(min(np.bincount(y, minlength=2)))))
    pred = np.empty(len(y), dtype=float)
    for train_idx, test_idx in StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=modeling.SEED).split(X, y):
        model = fit_tuned_clf(kind, X.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return pred


def delta_auc(kind, learn, val, base_cols, full_cols, n_folds=CLF_FOLDS):
    """The AUC twin of modeling.delta_r2: what the talk block adds, out of fold.

    Keys are named ``*_auc`` so that a row of the results table can never be read
    as an R² by accident.
    """
    y = learn[BINARY].to_numpy(dtype=int)
    base = out_of_fold_proba(kind, learn, base_cols, n_folds)
    full = out_of_fold_proba(kind, learn, full_cols, n_folds)
    idx_all = np.arange(len(learn))
    lo, hi = modeling.cluster_bootstrap(
        idx_all,
        lambda i, f=full, b=base: safe_auc(y[i], f[i]) - safe_auc(y[i], b[i]))

    out = {"delta_cv_auc": safe_auc(y, full) - safe_auc(y, base),
           "auc_ci_low": lo, "auc_ci_high": hi,
           "cv_auc_base": safe_auc(y, base), "cv_auc_full": safe_auc(y, full)}

    if val is not None and len(val):
        yv = val[BINARY].to_numpy(dtype=int)
        Xl = modeling.design(learn, base_cols)
        bv = fit_tuned_clf(kind, Xl, y).predict_proba(
            modeling.design(val, base_cols)[Xl.columns])[:, 1]
        Xlf = modeling.design(learn, full_cols)
        fv = fit_tuned_clf(kind, Xlf, y).predict_proba(
            modeling.design(val, full_cols)[Xlf.columns])[:, 1]
        lo_v, hi_v = modeling.cluster_bootstrap(
            np.arange(len(val)),
            lambda i, f=fv, b=bv: safe_auc(yv[i], f[i]) - safe_auc(yv[i], b[i]))
        out.update({"delta_heldout_auc": safe_auc(yv, fv) - safe_auc(yv, bv),
                    "auc_ci_low_heldout": lo_v, "auc_ci_high_heldout": hi_v,
                    "heldout_auc_base": safe_auc(yv, bv),
                    "heldout_auc_full": safe_auc(yv, fv)})
    return out


def one_feature_logit(df, feature, ctrl):
    """Logistic twin of modeling.one_feature: log-odds of maxing out, per SD.

    Same guards and the same within-subsample re-standardization, so the two
    outcomes' per-feature tables are screening on identical terms. Perfect or
    quasi-perfect separation is common with ~70 games and returns NaN rather than
    the enormous coefficient and meaningless p-value statsmodels would otherwise
    hand back.
    """
    x = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    y = df[BINARY].to_numpy(dtype=float)
    if (len(df) < 30 or x.std() < modeling.MIN_FEATURE_SD
            or x.nunique() < modeling.MIN_FEATURE_LEVELS
            or min(np.bincount(y.astype(int), minlength=2)) < MIN_PER_CLASS):
        return (np.nan,) * 4
    x = (x - x.mean()) / x.std()
    X = sm.add_constant(pd.concat([x.rename("feature"),
                                   modeling.design(df, ctrl)], axis=1))
    try:
        model = sm.Logit(y, X).fit(cov_type="HC1", disp=0, maxiter=200)
    except Exception:
        return (np.nan,) * 4
    if not getattr(model, "mle_retvals", {}).get("converged", True):
        return (np.nan,) * 4
    coef, p = model.params["feature"], model.pvalues["feature"]
    ci = model.conf_int().loc["feature"]
    # A log-odds this large is separation wearing a coefficient's clothes.
    if not np.isfinite(coef) or abs(coef) > 10:
        return (np.nan,) * 4
    return coef, p, ci[0], ci[1]


def binary_cell_ok(learn, val):
    """Both classes, on both sides, in enough numbers to fit and to score."""
    for frame in (learn, val):
        counts = np.bincount(frame[BINARY].to_numpy(dtype=int), minlength=2)
        if counts.min() < MIN_PER_CLASS:
            return False
    return True


# ----------------------------------------------------------- the tables -----
def sample_table(frames, n_values, summaries, min_rounds_list):
    """The power inventory, written before anything is fitted.

    One row per (N, summary, split, min_rounds): how many long games there were,
    how many of them talked in the first N rounds, and what the final-round outcome
    looks like among those. Cells that will be skipped are recorded here with the
    reason, so a missing result in the other tables is always explained.
    """
    rows = []
    for summary in summaries:
        for min_rounds in min_rounds_list:
            for n in n_values:
                sizes = {}
                for split in ("learn", "val"):
                    df = frames.get((split, summary))
                    if df is None:
                        rows.append({
                            "N": n, "summary": summary, "split": split,
                            "min_rounds": min_rounds, "n_games_eligible": np.nan,
                            "n_games_with_talk": np.nan,
                            "outcome_mean": np.nan, "outcome_sd": np.nan,
                            "frac_at_ceiling": np.nan, "frac_above_0.9": np.nan,
                            "status": "skipped",
                            "reason": f"data/processed/analysis_windows_{split}.csv "
                                      f"has not been built yet (needs steps 9-10)"})
                        sizes[split] = 0
                        continue
                    frame, _, n_eligible = game_frame(df, n, summary, min_rounds)
                    n_talk = 0 if frame is None else len(frame)
                    sizes[split] = n_talk
                    rows.append({
                        "N": n, "summary": summary, "split": split,
                        "min_rounds": min_rounds, "n_games_eligible": n_eligible,
                        "n_games_with_talk": n_talk,
                        "outcome_mean": (np.nan if not n_talk
                                         else frame[CONTINUOUS].mean()),
                        "outcome_sd": (np.nan if not n_talk
                                       else frame[CONTINUOUS].std()),
                        # How much of the outcome is a point mass on the boundary.
                        # Read this before reading any delta R2: it is the other
                        # half of why a continuous fit here is hard.
                        "frac_at_ceiling": (np.nan if not n_talk
                                            else frame[BINARY].mean()),
                        "frac_above_0.9": (np.nan if not n_talk
                                           else (frame[CONTINUOUS] >= 0.9).mean()),
                        "status": "", "reason": ""})
                # The cell is a learn/val pair; if either side is too thin, neither
                # a cross-validated nor a held-out estimate is worth reporting.
                for row in rows[-2:]:
                    if row["status"] == "skipped":
                        continue
                    thin = [s for s, k in sizes.items() if k < MIN_GAMES_CELL]
                    if thin:
                        row["status"] = "skipped"
                        row["reason"] = (
                            "fewer than %d games in %s (%s)"
                            % (MIN_GAMES_CELL, "/".join(thin),
                               ", ".join(f"{s}={sizes[s]}" for s in thin)))
                    else:
                        row["status"] = "ok"
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_WINDOWS / "first_n_sample.csv", index=False)
    return out


def usable(sample, n, summary, min_rounds):
    """Did the sample table clear this cell for modeling?"""
    cell = sample[(sample["N"] == n) & (sample["summary"] == summary)
                  & (sample["min_rounds"] == min_rounds)]
    return len(cell) == 2 and (cell["status"] == "ok").all()


def delta_table(frames, sample, n_values, summaries, min_rounds, kinds, n_folds):
    """What the opening-talk block adds over controls, per specification.

    Each cell is scored twice: ΔR² against the continuous final-round rate, and
    ΔAUC against ``maxed_out``. The ``outcome`` and ``metric`` columns say which,
    and the two metrics' value columns are named apart (``delta_cv_r2`` vs
    ``delta_cv_auc``) so a row can never be read as the wrong one.
    """
    rows = []
    for summary in summaries:
        learn_df, val_df = frames.get(("learn", summary)), frames.get(("val", summary))
        if learn_df is None or val_df is None:
            continue
        for n in n_values:
            if not usable(sample, n, summary, min_rounds):
                print(f"  [min={min_rounds}] N={n:<2} {summary:<14} skipped "
                      f"(see first_n_sample.csv)", flush=True)
                continue
            learn, feats, _ = game_frame(learn_df, n, summary, min_rounds)
            val, _, _ = game_frame(val_df, n, summary, min_rounds)
            binary_ok = binary_cell_ok(learn, val)
            if not binary_ok:
                print(f"  [min={min_rounds}] N={n:<2} {summary:<14} "
                      f"maxed_out not fitted: fewer than {MIN_PER_CLASS} games on "
                      f"one side of the ceiling", flush=True)
            for baseline in BASELINES:
                base = control_cols(learn, baseline)
                full = base + [c for c in feats if c not in set(base)]
                common = {"N": n, "summary": summary, "baseline": baseline,
                          "min_rounds": min_rounds, "n_features": len(feats),
                          "n_games_learn": len(learn), "n_games_heldout": len(val),
                          "frac_at_ceiling_learn": learn[BINARY].mean(),
                          "frac_at_ceiling_heldout": val[BINARY].mean()}
                for kind in kinds:
                    res = modeling.delta_r2(kind, learn, val, base, full,
                                            group_col=None, n_folds=n_folds)
                    rows.append({**common, "outcome": CONTINUOUS,
                                 "metric": "delta_r2", "model_family": kind, **res})
                    print(f"  [min={min_rounds}] N={n:<2} {summary:<14} "
                          f"{baseline:<18} {kind:<14} {CONTINUOUS:<18} "
                          f"dR2 cv {res['delta_cv_r2']:+.3f} "
                          f"[{res['ci_low']:+.3f},{res['ci_high']:+.3f}]  "
                          f"heldout {res.get('delta_heldout_r2', float('nan')):+.3f}  "
                          f"(n={len(learn)}/{len(val)})", flush=True)

                    if not binary_ok:
                        continue
                    # The same block, the same folds, against the ceiling itself.
                    # ``early_contribution`` stays continuous here on purpose: it
                    # is a control describing what the group was already doing, and
                    # dichotomizing a control only throws information away.
                    ares = delta_auc(kind, learn, val, base, full,
                                     n_folds=min(n_folds, CLF_FOLDS))
                    rows.append({**common, "outcome": BINARY,
                                 "metric": "delta_auc", "model_family": kind, **ares})
                    print(f"  [min={min_rounds}] N={n:<2} {summary:<14} "
                          f"{baseline:<18} {kind:<14} {BINARY:<18} "
                          f"dAUC cv {ares['delta_cv_auc']:+.3f} "
                          f"[{ares['auc_ci_low']:+.3f},{ares['auc_ci_high']:+.3f}]  "
                          f"heldout {ares.get('delta_heldout_auc', float('nan')):+.3f}"
                          f"  (base AUC {ares['cv_auc_base']:.3f}"
                          f"/{ares.get('heldout_auc_base', float('nan')):.3f})",
                          flush=True)
    return pd.DataFrame(rows)


def feature_effects(frames, sample, n_values, summaries, min_rounds):
    """One OLS per feature per specification, HC1 - there is nothing to cluster on.

    Features arrive z-scored against the pooled per-round conversation distribution
    and are re-standardized inside ``modeling.one_feature`` against the games in
    this cell, so a coefficient reads as "change in final-round contribution rate
    per SD of variation among the games actually being compared".

    The binary companion outcome is fitted by logistic regression instead, also with
    HC1, so its coefficient is a log-odds per SD - a different unit, flagged in the
    ``coef_units`` column and never mixed into the same comparison.

    FDR is applied within each (N, summary, baseline, outcome): each is a separate
    screen of the same feature list, and pooling them would correct for repetitions
    of one question rather than for many questions.
    """
    fam_of = pd.read_csv(TABLES / "feature_manifest.csv").set_index("feature")["family"]
    rows = []
    for summary in summaries:
        learn_df, val_df = frames.get(("learn", summary)), frames.get(("val", summary))
        if learn_df is None or val_df is None:
            continue
        for n in n_values:
            if not usable(sample, n, summary, min_rounds):
                continue
            learn, feats, _ = game_frame(learn_df, n, summary, min_rounds)
            val, _, _ = game_frame(val_df, n, summary, min_rounds)
            binary_ok = binary_cell_ok(learn, val)
            for baseline in BASELINES:
                ctrl_l = control_cols(learn, baseline)
                ctrl_v = control_cols(val, baseline)
                for outcome in OUTCOMES:
                    if outcome == BINARY and not binary_ok:
                        continue
                    fit = ((lambda d, f, c: modeling.one_feature(d, f, c,
                                                                 group_col=None))
                           if outcome == CONTINUOUS else one_feature_logit)
                    units = ("contribution rate per SD" if outcome == CONTINUOUS
                             else "log-odds per SD")
                    kept = 0
                    for feature in feats:
                        coef, p, lo, hi = fit(learn, feature, ctrl_l)
                        if np.isnan(coef):
                            continue
                        v = fit(val, feature, ctrl_v)
                        base_name = feature.rsplit("__", 1)[0]
                        rows.append({
                            "N": n, "summary": summary, "baseline": baseline,
                            "outcome": outcome, "coef_units": units,
                            "min_rounds": min_rounds, "feature": base_name,
                            "family": fam_of.get(base_name, "Other"),
                            "n_games_learn": len(learn),
                            "n_games_heldout": len(val),
                            "frac_at_ceiling_learn": learn[BINARY].mean(),
                            "coef_learn": coef, "p_learn": p,
                            "ci_low_learn": lo, "ci_high_learn": hi,
                            "coef_val": v[0], "p_val": v[1],
                            "ci_low_val": v[2], "ci_high_val": v[3]})
                        kept += 1
                    print(f"  N={n:<2} {summary:<14} {baseline:<18} "
                          f"{outcome:<18} {kept}/{len(feats)} features estimable "
                          f"(n={len(learn)}/{len(val)})", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_learn"] = np.nan
    for _, grp in out.groupby(["N", "summary", "baseline", "outcome"]):
        out.loc[grp.index, "q_learn"] = multipletests(grp["p_learn"],
                                                      method="fdr_bh")[1]
    # The held-out column is the stronger test: it played no part in selection.
    out["replicates"] = ((out["q_learn"] < 0.05) & (out["p_val"] < 0.05)
                         & (np.sign(out["coef_learn"]) == np.sign(out["coef_val"])))
    return out.sort_values(["summary", "outcome", "baseline", "N", "p_learn"]) \
        .reset_index(drop=True)


# ------------------------------------------------------------------ main ----
def main(args):
    summaries = ([args.summary] if args.summary != "both" else list(SUMMARIES))
    n_values = list(range(1, args.n_max + 1))
    kinds = ("elastic net",) if args.quick else modeling.MODEL_KINDS
    n_folds = QUICK_FOLDS if args.quick else modeling.N_FOLDS

    if args.quick:
        # ``cluster_bootstrap``'s n_boot default is bound at definition time, so
        # setting modeling.N_BOOT would not reach it. Wrapping is the least
        # invasive way to make the smoke test fast without touching modeling.py.
        full_boot = modeling.cluster_bootstrap
        modeling.cluster_bootstrap = (
            lambda groups, stat, n_boot=QUICK_BOOT: full_boot(groups, stat, n_boot))

    # The robustness thresholds are inventoried too, so a skip there is explained
    # by the same table as a skip in the headline.
    min_rounds_list = sorted({args.min_rounds, *ROBUSTNESS_MIN_ROUNDS})
    # N must stay clear of the final round even at the loosest cut.
    if args.n_max >= min(min_rounds_list) - 1:
        raise SystemExit(f"--n-max {args.n_max} is not safely inside a "
                         f"{min(min_rounds_list)}-round game")

    frames = {}
    for summary in summaries:
        for split in ("learn", "val"):
            frames[(split, summary)] = load(split, summary)
        if frames[("learn", summary)] is None or frames[("val", summary)] is None:
            print(f"! summary '{summary}': data/processed/analysis_windows_*.csv is "
                  f"not built yet - it comes from scripts/10_build_windows_table.py, "
                  f"which needs 09_extract_cumulative_features.py to finish. Recording the "
                  f"cells as skipped and carrying on.", flush=True)

    print(f"first-N analysis: N=1..{args.n_max}, min rounds {args.min_rounds} "
          f"(robustness {ROBUSTNESS_MIN_ROUNDS}), summaries {summaries}, "
          f"models {list(kinds)}{' [quick]' if args.quick else ''}", flush=True)

    # 1. The power inventory, first - the sample size has to be readable before
    #    any result is.
    print("\n1. sample inventory", flush=True)
    sample = sample_table(frames, n_values, summaries, min_rounds_list)
    head = sample[(sample["min_rounds"] == args.min_rounds)]
    print(head.to_string(index=False), flush=True)
    n_skipped = int((sample["status"] == "skipped").sum())
    print(f"   wrote first_n_sample.csv ({len(sample)} rows, "
          f"{n_skipped} skipped cells)", flush=True)

    # 2. Headline ΔR².
    print(f"\n2. delta R2 / delta AUC at the {args.min_rounds}-round cut", flush=True)
    deltas = delta_table(frames, sample, n_values, summaries, args.min_rounds,
                         kinds, n_folds)
    deltas.to_csv(TABLES_WINDOWS / "first_n_delta_r2.csv", index=False)
    print(f"   wrote first_n_delta_r2.csv ({len(deltas)} rows)", flush=True)

    # 3. Per-feature effects.
    print("\n3. per-feature effects", flush=True)
    effects = feature_effects(frames, sample, n_values, summaries, args.min_rounds)
    effects.to_csv(TABLES_WINDOWS / "first_n_feature_effects.csv", index=False)
    print(f"   wrote first_n_feature_effects.csv ({len(effects)} rows)", flush=True)

    # 4. Does the answer depend on where the "long game" line is drawn?
    print("\n4. robustness to the minimum-game-length cut (both outcomes)", flush=True)
    parts = [deltas] if len(deltas) else []
    for min_rounds in ROBUSTNESS_MIN_ROUNDS:
        if min_rounds == args.min_rounds and len(deltas):
            continue
        parts.append(delta_table(frames, sample, n_values, summaries, min_rounds,
                                 kinds, n_folds))
    robust = (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    robust = robust[robust["min_rounds"].isin(ROBUSTNESS_MIN_ROUNDS)] \
        if len(robust) else robust
    robust.to_csv(TABLES_WINDOWS / "first_n_robustness.csv", index=False)
    print(f"   wrote first_n_robustness.csv ({len(robust)} rows)", flush=True)

    summarize(sample, deltas, effects, args)


def _span(ok, split):
    """min-max games with talk for one split, as a string."""
    sub = ok[ok["split"] == split]["n_games_with_talk"]
    return "-" if sub.empty else f"{sub.min():.0f}-{sub.max():.0f}"


def summarize(sample, deltas, effects, args):
    """State the size of the screen next to its yield, so a null reads as a null."""
    print("\n" + "=" * 72, flush=True)
    print("first-N summary", flush=True)
    ok = sample[(sample["status"] == "ok") & (sample["min_rounds"] == args.min_rounds)]
    if len(ok):
        print(f"  games per cell at the {args.min_rounds}-round cut: "
              f"learn {_span(ok, 'learn')}, held out {_span(ok, 'val')}", flush=True)
    print(f"  cells skipped for size: "
          f"{int((sample['status'] == 'skipped').sum()) // 2} of "
          f"{len(sample) // 2}", flush=True)

    if len(ok):
        ceil = ok[ok["split"] == "learn"]["frac_at_ceiling"]
        print(f"  at the ceiling (final-round rate >= {CEILING}): "
              f"{ceil.min():.1%}-{ceil.max():.1%} of learning games. Half the "
              f"outcome is a boundary atom; read the delta R2 accordingly.",
              flush=True)

    for outcome, delta_col, lo_col, hi_col, heldout_col, label in (
            (CONTINUOUS, "delta_cv_r2", "ci_low", "ci_high",
             "delta_heldout_r2", "delta R2"),
            (BINARY, "delta_cv_auc", "auc_ci_low", "auc_ci_high",
             "delta_heldout_auc", "delta AUC")):
        sub = deltas[deltas["outcome"] == outcome] if len(deltas) else deltas
        if not len(sub) or delta_col not in sub.columns:
            print(f"  {label}: not fitted", flush=True)
            continue
        pos = sub[(sub[delta_col] > 0) & (sub[lo_col] > 0)]
        held = sub.get(heldout_col, pd.Series(dtype=float))
        print(f"  {label} ({outcome}): {len(sub)} specifications; "
              f"{len(pos)} with a cross-validated interval excluding zero; "
              f"{int((held > 0).sum())} positive on held-out games", flush=True)
        best = sub.loc[sub[delta_col].idxmax()]
        print(f"    largest cross-validated {label}: {best[delta_col]:+.3f} "
              f"(N={best['N']}, {best['summary']}, {best['baseline']}, "
              f"{best['model_family']}), held out "
              f"{best.get(heldout_col, float('nan')):+.3f}", flush=True)

    total_rep = 0
    for outcome in OUTCOMES:
        sub = (effects[effects["outcome"] == outcome] if len(effects)
               else effects)
        if not len(sub):
            print(f"  per-feature screen ({outcome}): not fitted", flush=True)
            continue
        n_screened = len(sub)
        n_p05 = int((sub["p_learn"] < 0.05).sum())
        n_fdr = int((sub["q_learn"] < 0.05).sum())
        n_rep = int(sub["replicates"].sum())
        total_rep += n_rep
        print(f"  per-feature screen ({outcome}, "
              f"{sub['coef_units'].iloc[0]}):", flush=True)
        print(f"    features screened: {n_screened} "
              f"({sub['feature'].nunique()} distinct features across "
              f"{sub.groupby(['N', 'summary', 'baseline']).ngroups} screens)",
              flush=True)
        print(f"    expected significant by chance at p<.05: "
              f"{0.05 * n_screened:.1f}; observed: {n_p05}", flush=True)
        print(f"    clearing FDR q<.05 on the learning split: {n_fdr}", flush=True)
        print(f"    also replicating held out (p<.05, same sign): {n_rep}",
              flush=True)
        for row in sub[sub["replicates"]].head(6).itertuples():
            print(f"       N={row.N} {row.summary}/{row.baseline}: {row.feature} "
                  f"{row.coef_learn:+.3f} (q={row.q_learn:.3f}) / held out "
                  f"{row.coef_val:+.3f} (p={row.p_val:.3f})", flush=True)
    if len(effects) and total_rep == 0:
        print("  -> nothing survives, under either outcome. Read this as a null: "
              "with these sample sizes the analysis could only have detected a "
              "large effect.", flush=True)
    print("=" * 72, flush=True)
    print(f"tables written to {TABLES_WINDOWS}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Does a group's first-N-round talk predict its final round?")
    ap.add_argument("--n-max", type=int, default=DEFAULT_N_MAX,
                    help="largest N (first-N rounds) to test; default 10")
    ap.add_argument("--min-rounds", type=int, default=DEFAULT_MIN_ROUNDS,
                    help="minimum rounds played for a game to enter; default 15")
    ap.add_argument("--summary", choices=list(SUMMARIES) + ["both"],
                    default="mean_of_rounds",
                    help="how to summarize the first N rounds of talk")
    ap.add_argument("--quick", action="store_true",
                    help="fewer folds and bootstraps, elastic net only (smoke test)")
    main(ap.parse_args())
