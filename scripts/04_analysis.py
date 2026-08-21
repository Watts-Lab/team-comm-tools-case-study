"""Step 4 - answer the research question.

    Which conversation features predict greater contribution in groups that
    communicate, versus groups that do not?

The question has three parts, and this script answers them in order:

  A. Does having a channel matter at all?  (chat vs. no-chat groups)
  B. Among groups that talked, which conversation features move contribution,
     holding the rules of the game fixed?
  C. Does conversation content buy predictive power that the game's design
     parameters do not already provide - and does it hold up out of sample?

Everything is fit on the learning split and then checked on a held-out split
that played no part in any modeling decision.

Outputs (into outputs/tables/): channel_effect.csv, feature_effects.csv,
model_comparison.csv, selected_coefficients.csv

Run:  python scripts/04_analysis.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from config import CONFIG_COLS, DATA_PROCESSED, SEED, TABLES

OUTCOME = "contribution_penultimate"
N_FOLDS = 10
N_REPEATS = 3     # repeats of the fold split, to damp fold-assignment noise
N_BOOT = 2000

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- helpers ---
def load(split):
    df = pd.read_csv(DATA_PROCESSED / f"analysis_{split}.csv")
    # Booleans arrive from CSV as True/False strings in some columns.
    for col in df.columns:
        if df[col].dtype == object:
            lowered = df[col].astype(str).str.lower()
            if lowered.isin({"true", "false", "nan"}).all():
                df[col] = lowered.map({"true": 1.0, "false": 0.0})
    return df


def feature_columns():
    manifest = pd.read_csv(TABLES / "feature_manifest.csv")
    return manifest.loc[manifest["kept"], "feature"].tolist()


def usable_configs(df):
    """Config columns present and varying within this subset of games."""
    return [c for c in CONFIG_COLS
            if c in df.columns and df[c].nunique(dropna=True) > 1]


def design(df, cols):
    """Float design matrix with median imputation for the few missing cells.

    The explicit float cast matters: several config columns are booleans, and a
    frame mixing bool and float columns lands in statsmodels as dtype object.
    """
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    return X.fillna(X.median())


def talkers(df):
    """Groups with an open channel that actually used it."""
    return df[df["did_communicate"].astype(bool)].copy()


def silent(df):
    """Groups that had no channel at all - the counterfactual baseline."""
    return df[~df["has_chat_channel"].astype(bool)].copy()


# ------------------------------------------------ A. does the channel matter -
def channel_effect(splits):
    """Regress contribution on channel availability, controlling for game rules."""
    rows = []
    for split, df in splits.items():
        configs = usable_configs(df)
        X = pd.concat([df["has_chat_channel"].astype(float).rename("has_chat_channel"),
                       design(df, configs)], axis=1)
        model = sm.OLS(df[OUTCOME], sm.add_constant(X)).fit(cov_type="HC3")
        ci = model.conf_int().loc["has_chat_channel"]
        rows.append({
            "split": split,
            "n_games": len(df),
            "mean_no_channel": df.loc[~df.has_chat_channel.astype(bool), OUTCOME].mean(),
            "mean_channel": df.loc[df.has_chat_channel.astype(bool), OUTCOME].mean(),
            "adj_coef": model.params["has_chat_channel"],
            "ci_low": ci[0], "ci_high": ci[1],
            "p_value": model.pvalues["has_chat_channel"],
            "model_r2": model.rsquared,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "channel_effect.csv", index=False)
    return out


# --------------------------------------- B. which conversation features matter
def feature_effects(learn, val, features):
    """One regression per conversation feature, among groups that talked.

    Each feature is standardized within the split, so the coefficient reads as
    "change in contribution rate per SD of this feature", holding the game's
    design parameters fixed.

    Two multiplicity controls are reported, and they answer different questions.
    ``q_learn`` is a Benjamini-Hochberg correction across all features within the
    learning split - the strict within-sample screen. ``replicates`` is the
    out-of-sample test: the same feature reaching p<.05 with the same sign in a
    split that played no part in selecting it. With ~150 conversations the FDR
    screen has very little power, so replication is the more informative of the
    two, and a feature that clears it while failing FDR is reported as exactly
    that rather than as an established effect.
    """
    learn_t, val_t = talkers(learn), talkers(val)
    configs = usable_configs(learn_t)

    def fit_one(df, feature, config_cols):
        x = pd.to_numeric(df[feature], errors="coerce")
        if x.notna().sum() < 20 or x.std(skipna=True) == 0:
            return np.nan, np.nan, np.nan, np.nan
        z = ((x - x.mean()) / x.std()).fillna(0.0)
        X = sm.add_constant(pd.concat([z.rename("feature"),
                                       design(df, config_cols)], axis=1))
        model = sm.OLS(df[OUTCOME], X).fit(cov_type="HC3")
        ci = model.conf_int().loc["feature"]
        return model.params["feature"], model.pvalues["feature"], ci[0], ci[1]

    val_configs = usable_configs(val_t)
    rows = []
    for feature in features:
        coef, p, lo, hi = fit_one(learn_t, feature, configs)
        if np.isnan(coef):
            continue
        v_coef, v_p, v_lo, v_hi = fit_one(val_t, feature, val_configs)
        rows.append({"feature": feature, "coef_learn": coef, "p_learn": p,
                     "ci_low_learn": lo, "ci_high_learn": hi,
                     "coef_val": v_coef, "p_val": v_p,
                     "ci_low_val": v_lo, "ci_high_val": v_hi})

    out = pd.DataFrame(rows)
    out["q_learn"] = multipletests(out["p_learn"], method="fdr_bh")[1]
    out["replicates"] = (out["p_learn"] < 0.05) & (out["p_val"] < 0.05) & \
                        (np.sign(out["coef_learn"]) == np.sign(out["coef_val"]))
    out = out.sort_values("p_learn").reset_index(drop=True)
    out.to_csv(TABLES / "feature_effects.csv", index=False)
    return out


# ------------------------------------------ C. does conversation add prediction
def model_pipeline(penalized):
    """Standardized regression; penalized when the feature count is large."""
    estimator = (ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=50, cv=5,
                              random_state=SEED, max_iter=10000, n_jobs=-1)
                 if penalized else LinearRegression())
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", estimator)])


def r2(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def bootstrap_r2_ci(y, pred, n_boot=N_BOOT):
    """Percentile CI for R², resampling games with replacement."""
    y, pred = np.asarray(y), np.asarray(pred)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    draws = np.array([r2(y[i], pred[i]) for i in idx])
    return np.percentile(draws, [2.5, 97.5])


def out_of_fold(pipe, X, y):
    """Out-of-fold predictions, averaged over several independent fold splits.

    With ~150 games a single 10-fold split is noisy enough that R² can swing by
    a tenth on the choice of random seed alone. Averaging the out-of-fold
    predictions across repeats damps that without touching any held-out data.
    """
    preds = [cross_val_predict(pipe, X, y,
                               cv=KFold(N_FOLDS, shuffle=True, random_state=SEED + r))
             for r in range(N_REPEATS)]
    return np.mean(preds, axis=0)


def evaluate(name, learn_df, val_df, cols, penalized):
    """Cross-validated fit on learn, then a single honest test on val."""
    X_learn, y_learn = design(learn_df, cols), learn_df[OUTCOME].to_numpy()
    pipe = model_pipeline(penalized)

    oof = out_of_fold(pipe, X_learn, y_learn)
    lo, hi = bootstrap_r2_ci(y_learn, oof)

    pipe.fit(X_learn, y_learn)
    X_val, y_val = design(val_df, cols), val_df[OUTCOME].to_numpy()
    val_pred = pipe.predict(X_val[X_learn.columns])

    row = {"model": name, "n_features": len(cols),
           "n_learn": len(learn_df), "n_val": len(val_df),
           "cv_r2_learn": r2(y_learn, oof), "cv_r2_ci_low": lo, "cv_r2_ci_high": hi,
           "r2_heldout_val": r2(y_val, val_pred)}
    predictions = {"y_learn": y_learn, "oof": oof, "y_val": y_val, "val": val_pred}
    return row, pipe, predictions


def paired_delta_r2(better, baseline, n_boot=N_BOOT):
    """Paired bootstrap of the R² difference between two models on the same games.

    Resampling the same game indices for both models keeps the comparison paired,
    so the interval reflects how much better one model is rather than how much
    each model varies on its own.
    """
    def delta(key_y, key_pred):
        y = better[key_y]
        idx = rng.integers(0, len(y), size=(n_boot, len(y)))
        draws = np.array([r2(y[i], better[key_pred][i]) - r2(y[i], baseline[key_pred][i])
                          for i in idx])
        point = r2(y, better[key_pred]) - r2(y, baseline[key_pred])
        return point, *np.percentile(draws, [2.5, 97.5])

    cv_point, cv_lo, cv_hi = delta("y_learn", "oof")
    val_point, val_lo, val_hi = delta("y_val", "val")
    return {"delta_cv_r2_learn": cv_point, "delta_cv_ci_low": cv_lo,
            "delta_cv_ci_high": cv_hi, "delta_r2_heldout_val": val_point,
            "delta_val_ci_low": val_lo, "delta_val_ci_high": val_hi}


def model_comparison(learn, val, features):
    configs = usable_configs(talkers(learn))
    rows, fitted = [], {}

    specs = [
        # Groups that could not talk: how far do the rules of the game alone go?
        ("no channel: game rules only", silent(learn), silent(val), configs, False),
        # Same model, groups that did talk - the like-for-like baseline.
        ("communicating: game rules only", talkers(learn), talkers(val), configs, False),
        # The question of interest: does what they said add anything?
        ("communicating: game rules + conversation",
         talkers(learn), talkers(val), configs + features, True),
        # Conversation features alone, for reference.
        ("communicating: conversation only",
         talkers(learn), talkers(val), features, True),
    ]
    preds = {}
    for name, l_df, v_df, cols, penalized in specs:
        row, pipe, prediction = evaluate(name, l_df, v_df, cols, penalized)
        rows.append(row)
        fitted[name] = (pipe, cols)
        preds[name] = prediction

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "model_comparison.csv", index=False)

    # The question of interest, stated as one number: what does adding the
    # conversation buy, over the same groups modeled from the game rules alone?
    delta = paired_delta_r2(preds["communicating: game rules + conversation"],
                            preds["communicating: game rules only"])
    pd.DataFrame([delta]).to_csv(TABLES / "delta_r2.csv", index=False)

    # Which conversation features did the penalized model actually keep?
    pipe, cols = fitted["communicating: game rules + conversation"]
    coefs = pd.DataFrame({"feature": cols,
                          "coef_standardized": pipe.named_steps["model"].coef_})
    coefs["is_conversation_feature"] = ~coefs["feature"].isin(configs)
    coefs = coefs[coefs["coef_standardized"] != 0]
    coefs = coefs.loc[coefs["coef_standardized"].abs()
                      .sort_values(ascending=False).index]
    coefs.to_csv(TABLES / "selected_coefficients.csv", index=False)
    return out, coefs, delta


# ------------------------------------------------------------------- main ---
def main():
    learn, val = load("learn"), load("val")
    features = [f for f in feature_columns()
                if f in learn.columns and f in val.columns]
    print(f"learn: {len(learn)} games ({len(talkers(learn))} talked, "
          f"{len(silent(learn))} had no channel)")
    print(f"val:   {len(val)} games ({len(talkers(val))} talked, "
          f"{len(silent(val))} had no channel)")
    print(f"conversation features carried into modeling: {len(features)}\n")

    print("A. Channel effect")
    print(channel_effect({"learn": learn, "val": val}).to_string(index=False), "\n")

    print("B. Conversation features (top 10 by learn-split p-value)")
    effects = feature_effects(learn, val, features)
    cols = ["feature", "coef_learn", "p_learn", "q_learn", "coef_val", "p_val"]
    print(effects.head(10)[cols].to_string(index=False))
    print(f"features passing FDR q<.05 within the learning split: "
          f"{(effects.q_learn < 0.05).sum()}")
    print(f"features replicating out of sample (p<.05 in both splits, same sign): "
          f"{effects.replicates.sum()}\n")

    print("C. Predictive comparison")
    comparison, coefs, delta = model_comparison(learn, val, features)
    print(comparison.to_string(index=False))
    print(f"\nadding conversation to game rules, among groups that talked:")
    print(f"  cross-validated dR2 = {delta['delta_cv_r2_learn']:+.3f} "
          f"[{delta['delta_cv_ci_low']:+.3f}, {delta['delta_cv_ci_high']:+.3f}]")
    print(f"  held-out         dR2 = {delta['delta_r2_heldout_val']:+.3f} "
          f"[{delta['delta_val_ci_low']:+.3f}, {delta['delta_val_ci_high']:+.3f}]")
    print(f"\nnon-zero coefficients retained: {len(coefs)} "
          f"({int(coefs.is_conversation_feature.sum())} conversation features)")
    print("\ntables written to outputs/tables/")


if __name__ == "__main__":
    main()
