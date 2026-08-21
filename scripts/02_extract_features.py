"""Step 2 - run the Team Communication Toolkit over the prepared chat data.

This is the part of the case study that the toolkit exists for: from a four-column
chat table to 100+ conversation features, in one call per split.

Outputs (per split, under outputs/features/output/):
  * ``chat/{split}_chat_level.csv``  - one row per message
  * ``user/{split}_user_level.csv``  - one row per speaker per conversation
  * ``conv/{split}_conv_level.csv``  - one row per conversation (the unit we analyze)

The ``output/{level}/`` nesting is imposed by the toolkit: it rewrites whatever
paths you hand it into that layout, so the paths below mirror the convention
rather than fight it.

Run:  python scripts/02_extract_features.py [--force]
"""

import argparse

import pandas as pd
from team_comm_tools import FeatureBuilder

from config import DATA_PROCESSED, FEATURES, SPLITS, VECTOR_CACHE

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
    paths = output_paths(split)
    if not force and all(p.exists() for p in paths.values()):
        print(f"[{split}] outputs already exist; pass --force to regenerate")
        return

    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv")

    # The toolkit accepts either a datetime or a numeric timestamp; epoch
    # milliseconds is unambiguous and matches the default `timestamp_unit`.
    ts = pd.to_datetime(chat["timestamp"], utc=True, errors="coerce")
    chat = chat.loc[ts.notna()].copy()
    chat["timestamp"] = (ts.loc[ts.notna()].astype("int64") // 10**6)

    print(f"[{split}] featurizing {len(chat)} messages from "
          f"{chat.gameId.nunique()} conversations")

    FeatureBuilder(
        input_df=chat,
        conversation_id_col="gameId",     # one conversation == one game
        speaker_id_col="playerId",
        message_col="text",
        timestamp_col="timestamp",
        vector_directory=vector_directory(),
        output_file_path_chat_level=str(paths["chat"]),
        output_file_path_user_level=str(paths["user"]),
        output_file_path_conv_level=str(paths["conv"]),
        custom_features=CUSTOM_FEATURES,
        turns=False,                      # each message stays its own row
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
