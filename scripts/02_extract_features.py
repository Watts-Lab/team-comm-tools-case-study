"""Step 2 - run the Team Communication Toolkit over the prepared chat data.

This is the part of the case study that the toolkit exists for: from a four-column
chat table to 100+ conversation features, in one call per split.

Outputs (per split, under outputs/features/output/):
  * ``chat/{split}_chat_level.csv``  - one row per message
  * ``user/{split}_user_level.csv``  - one row per speaker per conversation
  * ``conv/{split}_conv_level.csv``  - one row per game-round conversation
    (the grain the analysis joins on)

The learning split is written with the toolkit's redundancy reduction applied; the
held-out split keeps every column, and step 3 selects the learning split's surviving
columns from it. That way both splits carry exactly the same features and the
selection never sees held-out data.

The ``output/{level}/`` nesting is imposed by the toolkit: it rewrites whatever
paths you hand it into that layout, so the paths below mirror the convention
rather than fight it. Its logs go to ``outputs/logs/``.

Run:  python scripts/02_extract_features.py [--force]
"""

import argparse
import os

import pandas as pd
from team_comm_tools import FeatureBuilder

from config import DATA_PROCESSED, FEATURES, OUTPUTS, ROOT, SPLITS, VECTOR_CACHE

# FeatureBuilder writes its log tree to ``./{output_file_base}/logs/``, relative to
# the working directory. Naming the base "outputs" and running from the repo root
# puts those logs in outputs/logs/ alongside everything else the pipeline produces,
# instead of creating a second top-level output/ folder next to outputs/.
LOG_BASE = OUTPUTS.name

# --- redundancy reduction, new in team_comm_tools 0.1.8 ----------------------
# The toolkit emits families of near-duplicate columns - politeness and
# receptiveness overlap heavily, and every chat-level feature appears again as a
# conversation mean - so it can group features correlated above a threshold and
# keep one representative per group. Using it replaces the hand-rolled correlation
# screen this case study used to carry.
#
# Two configuration choices are deliberate:
#
#   treat_zero_as_na=False. The toolkit's default is True, which is the better
#   choice for *estimating* correlations between sparse features (co-absence is
#   not evidence of similarity). But the same frame it modifies is the one written
#   to disk, so the default also rewrites every zero in the output as NA. On a
#   60-conversation sample that turned 7,642 zeros into 20,745 NAs. Here zero is
#   meaningful - "this conversation contained no greetings" is data, not a missing
#   value - and the downstream analysis reads NA as "no conversation happened".
#
#   Reduction runs on the learning split only. Which columns survive is a
#   feature-selection decision, and every such decision in this study is made on
#   the learning split and then applied unchanged to held-out data. Letting each
#   split reduce independently would give them different feature sets.
CORR_THRESH = 0.9
MIN_NA_RATIO = 0.3
MIN_ZERO_RATIO = 0.9

# Features that need sentence embeddings, so they are opt-in in the toolkit.
# All four describe how a conversation moves rather than what any single message
# says, which is exactly the kind of group-level signal this study is after.
CUSTOM_FEATURES = [
    "(BERT) Mimicry",
    "Moving Mimicry",
    "Forward Flow",
    "Discursive Diversity",
]


def vector_directory():
    """Prepare the embedding cache and return it as a trailing-slash string.

    The toolkit builds its sub-paths by string concatenation, so the trailing
    slash is required or it will create sibling folders like ``vector_cachesentence``.
    """
    for sub in ("sentence", "sentiment"):
        (VECTOR_CACHE / sub).mkdir(parents=True, exist_ok=True)
    return str(VECTOR_CACHE) + "/"


def output_paths(split):
    """Where the toolkit will actually write, given the paths we pass it.

    FeatureBuilder normalizes any output path to ``<dir>/output/<level>/<name>.csv``,
    so these are the paths downstream steps read from.
    """
    return {level: FEATURES / "output" / level / f"{split}_{level}_level.csv"
            for level in ("chat", "user", "conv")}


def featurize(split, force=False):
    reduce_here = split == "learn"
    paths = output_paths(split)
    if not force and all(p.exists() for p in paths.values()):
        print(f"[{split}] outputs already exist; pass --force to regenerate")
        return

    # FeatureBuilder resolves its log directory against the working directory, so
    # anchor to the repo root and the logs land in outputs/logs/ however the
    # script was invoked.
    os.chdir(ROOT)

    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv")

    # The toolkit accepts either a datetime or a numeric timestamp; epoch
    # milliseconds is unambiguous and matches the default `timestamp_unit`.
    ts = pd.to_datetime(chat["timestamp"], utc=True, errors="coerce")
    chat = chat.loc[ts.notna()].copy()
    chat["timestamp"] = (ts.loc[ts.notna()].astype("int64") // 10**6)

    print(f"[{split}] featurizing {len(chat)} messages from "
          f"{chat.conv_id.nunique()} round-conversations"
          + ("  (reducing redundant columns)" if reduce_here else "  (all columns)"))

    FeatureBuilder(
        input_df=chat,
        # One conversation == one game-round. Features therefore describe what a
        # particular group said during a particular round, which is the grain the
        # analysis needs; TCT drops conversations with fewer than two speakers.
        conversation_id_col="conv_id",
        speaker_id_col="playerId",
        message_col="text",
        timestamp_col="timestamp",
        vector_directory=vector_directory(),
        output_file_base=LOG_BASE,        # log tree goes under outputs/, see above
        output_file_path_chat_level=str(paths["chat"]),
        output_file_path_user_level=str(paths["user"]),
        output_file_path_conv_level=str(paths["conv"]),
        custom_features=CUSTOM_FEATURES,
        turns=False,                      # each message stays its own row
        drop_redundant_columns=reduce_here,
        corr_thresh=CORR_THRESH,
        min_na_ratio=MIN_NA_RATIO,
        min_zero_ratio=MIN_ZERO_RATIO,
        treat_zero_as_na=False,           # see the note above; zero is data here
    ).featurize()

    conv = pd.read_csv(paths["conv"])
    print(f"[{split}] conversation-level output: {conv.shape[0]} rows x "
          f"{conv.shape[1]} columns -> {paths['conv'].name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-run even if outputs already exist")
    ap.add_argument("--split", choices=SPLITS, help="run a single split")
    args = ap.parse_args()

    for split in ([args.split] if args.split else SPLITS):
        featurize(split, force=args.force)
