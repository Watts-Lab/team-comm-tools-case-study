"""Step 15 - what the absolute stage partition does with games too short for it.

The opening is the first three rounds of a game and the endgame the last three, so
the two definitions overlap in any game of fewer than seven rounds. Step 1 resolves
that overlap in favour of the endgame: a round is "endgame" whenever two or fewer
remain, and "opening" only otherwise. A three-round game is therefore coded
entirely as endgame and contributes no opening round at all.

That convention cannot inflate the opening effect, since the shortest games are
kept out of the opening cell rather than added to it. What it could do is dilute
the endgame into a null - 14-16% of endgame rounds are whole short games rather
than the closing rounds of long ones - and a null produced by dilution is a weaker
thing than a null produced by absence. This step tests that directly, by refitting
each stage on games of at least a given length:

  min_rounds = 0   the published sample
  min_rounds = 7   the shortest length at which all three stages exist and are
                   disjoint, so every included game contributes to all three
  min_rounds = 11  and 16, purging the endgame further, to see whether a masked
                   effect appears once the short games are gone

Everything else matches step 4's `round_stage`: the POST block, the conversing
sample, the elastic net, the same controls and the same game-clustered bootstraps.
Rows below the 150-round floor step 4 imposes are reported with an empty estimate
rather than fitted; short games cannot be given a cell of their own for the same
reason, contributing only about fifty conversing rounds between both stages.

Run:  python scripts/15_short_game_scope.py
"""

import importlib
import sys

import pandas as pd

from config import TABLES

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
A = importlib.import_module("04_analysis")

BLOCK = "post"
MODEL = "elastic net"
MIN_ROUNDS = (0, 7, 11, 16)
STAGES = ("opening", "middle", "endgame")
# Step 4's floor, repeated rather than imported: a cell thinner than this is not
# an estimate of anything, and at these subsample sizes the penalised fit starts
# reporting the noise instead.
MIN_CELL = 150


def game_length(df):
    return df.groupby(A.GROUP)["round_index"].transform("max") + 1


def one_cell(learn, val, stage, min_rounds):
    """Additional R² from the POST block within one stage, on long-enough games."""
    flag = f"has_features_{BLOCK}"
    l = learn[(learn["stage_absolute"] == stage) & learn[flag].astype(bool)
              & (learn["L"] >= min_rounds)]
    v = val[(val["stage_absolute"] == stage) & val[flag].astype(bool)
            & (val["L"] >= min_rounds)]
    row = dict(stage=stage, min_rounds=min_rounds, n_rounds=len(l),
               n_games=l[A.GROUP].nunique(), n_rounds_heldout=len(v),
               n_games_heldout=v[A.GROUP].nunique())
    if len(l) < MIN_CELL or l[A.GROUP].nunique() < A.N_FOLDS:
        return {**row, "skip_reason": f"fewer than {MIN_CELL} rounds"}

    base_cols = A.controls(l) + [A.CHANNEL]
    feat_cols = base_cols + A.block_features(l, BLOCK, with_indicator=False)
    y, y_val = l[A.OUTCOME].to_numpy(), v[A.OUTCOME].to_numpy()
    base, full = A.out_of_fold(MODEL, l, base_cols), A.out_of_fold(MODEL, l, feat_cols)
    _, base_val = A.fit_predict_heldout(MODEL, l, v, base_cols)
    _, full_val = A.fit_predict_heldout(MODEL, l, v, feat_cols)
    lo, hi = A.cluster_bootstrap(
        l[A.GROUP],
        lambda i, f=full, b=base: A.r2(y[i], f[i]) - A.r2(y[i], b[i]))
    lo_v, hi_v = A.cluster_bootstrap(
        v[A.GROUP],
        lambda i, f=full_val, b=base_val: A.r2(y_val[i], f[i]) - A.r2(y_val[i], b[i]))
    return {**row,
            "delta_cv_r2": A.r2(y, full) - A.r2(y, base), "ci_low": lo, "ci_high": hi,
            "delta_heldout_r2": A.r2(y_val, full_val) - A.r2(y_val, base_val),
            "ci_low_heldout": lo_v, "ci_high_heldout": hi_v, "skip_reason": ""}


def composition(learn, val):
    """How much of each stage comes from games too short to have all three."""
    rows = []
    for name, df in (("learn", learn), ("val", val)):
        for stage in STAGES:
            sub = df[df["stage_absolute"] == stage]
            rows.append(dict(split=name, stage=stage, n_rounds=len(sub),
                             n_games=sub[A.GROUP].nunique(),
                             n_rounds_short_games=int((sub["L"] < 7).sum()),
                             pct_rounds_short_games=100 * (sub["L"] < 7).mean()))
    return pd.DataFrame(rows)


def main():
    learn, val = A.load("learn"), A.load("val")
    for df in (learn, val):
        df["L"] = game_length(df)

    comp = composition(learn, val)
    print("stage composition, all channel-open and silent rounds:")
    print(comp.round(1).to_string(index=False), "\n")

    out = pd.DataFrame([one_cell(learn, val, stage, m)
                        for m in MIN_ROUNDS for stage in STAGES])
    print(f"{BLOCK.upper()} block, conversing rounds, {MODEL}:")
    print(out.round(3).to_string(index=False))

    path = TABLES / "short_game_scope.csv"
    out.to_csv(path, index=False)
    comp.to_csv(TABLES / "short_game_composition.csv", index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
