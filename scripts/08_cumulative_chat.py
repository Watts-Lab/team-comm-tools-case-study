"""Step 8 - build the *cumulative* conversation for each game-round.

The case study's original design bounds the talk that precedes one contribution
decision two ways, and both are narrow by construction:

    pre    the round's own contribution phase (this round, before the outcome)
    post   the previous round's outcome and summary phases (last round, after it)

Neither can see anything a group said earlier in the game. A group that spent
round 2 agreeing on a plan and then said "same as before" in round 9 looks, to the
``post`` block, like a group that said nothing of substance. This step adds the
third bound:

    cumulative    every message the group has spoken so far, from the start of the game
           up to and including the talk that immediately precedes this decision

Concretely, the cumulative conversation for target round r in game g is every
``pre``/``post`` message whose ``target_round`` is at most r. It is a strict
superset of that round's ``pre`` and ``post`` blocks, so it is never entered into a
model alongside them - it is an alternative bound on the same decision, not an
extra one.

Messages are duplicated across cumulative conversations by construction (a round-2
message appears in the cumulative conversation of every later round), which is why
this is written to its own chat table and featurized separately: it must not
contaminate the per-round feature files the published analysis reads.

Outputs (per split, into data/processed/):
  * ``chat_cumulative_{split}.csv`` - one row per (message, target round) pair, in the
    four-column format the toolkit expects, keyed by a ``_cum`` conversation id.

Run:  python scripts/08_cumulative_chat.py
"""

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, SPLITS

# A cumulative conversation grows without bound in a 30-round game, and several
# toolkit features (discursive diversity, the mimicry family) are quadratic in the
# number of messages. Nothing is truncated - the point of the block is that it sees
# everything - but the cost is reported up front so a long run is not a surprise.


def build(split):
    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv", low_memory=False)

    # ``window`` rows are the pre/post messages duplicated under a third id by
    # step 1; taking them here would double every message.
    base = chat[chat["block"].isin(["pre", "post"])].copy()

    rounds = pd.read_csv(DATA_PROCESSED / f"rounds_{split}.csv", low_memory=False)
    # Only build a cumulative conversation for decisions the analysis actually
    # models, i.e. game-rounds that survived step 1's minimum-length filter.
    targets = rounds[["gameId", "round_index"]].rename(
        columns={"round_index": "target_round"})
    targets["gameId"] = targets["gameId"].astype(str)
    base["gameId"] = base["gameId"].astype(str)

    # Cross-join within game, then keep the lower triangle: message.target_round
    # <= decision.target_round. Done per game so the intermediate never holds the
    # full cross product of the split.
    pieces = []
    for game_id, msgs in base.groupby("gameId", sort=False):
        game_targets = targets.loc[targets["gameId"] == game_id, "target_round"]
        if game_targets.empty:
            continue
        for r in np.sort(game_targets.unique()):
            take = msgs[msgs["target_round"] <= r]
            if take.empty:
                continue
            pieces.append(take.assign(block="cumulative",
                                      cumulative_target_round=r,
                                      conv_id=f"{game_id}_r{int(r)}_cumulative"))

    cum = pd.concat(pieces, ignore_index=True)
    # Order matters: every sequential feature (forward flow, moving mimicry) reads
    # the frame in row order, so a cumulative conversation has to be in time order.
    cum = cum.sort_values(["conv_id", "timestamp"]).reset_index(drop=True)

    out = DATA_PROCESSED / f"chat_cumulative_{split}.csv"
    cum.to_csv(out, index=False)

    lengths = cum.groupby("conv_id").size()
    print(f"[{split}] {len(base)} distinct messages -> {len(cum)} message-rows "
          f"across {cum.conv_id.nunique()} cumulative conversations")
    print(f"         length: median {int(lengths.median())}, "
          f"90th pct {int(lengths.quantile(0.9))}, max {int(lengths.max())}")
    print(f"         wrote {out.name}")


if __name__ == "__main__":
    for s in SPLITS:
        build(s)
