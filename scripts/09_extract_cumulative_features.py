"""Step 9 - run the toolkit over the cumulative conversations from step 8.

This is step 2 again, pointed at a different chat table and with one deliberate
difference: **redundancy reduction is off**.

Which features survive reduction is a feature-selection decision, and every such
decision in this study was already made once, on the learning split's per-round
conversations, and is recorded in ``outputs/tables/feature_manifest.csv``. Letting
the cumulative block reduce independently would give it a different feature set
from the ``pre`` and ``post`` blocks, and the whole point of the block comparison
is that the three are measured with the same ruler. So every column is computed
here and step 10 selects the manifest's columns out of it.

Cost. The cumulative block is ~265,000 message-rows against the per-round blocks'
~25,000, because a message reappears in every later round's cumulative window.
Several features are also quadratic in conversation length and the longest
cumulative conversation runs to 496 messages. Expect this to take hours, not
minutes. Sentence embeddings are cached by message text in
``outputs/vector_cache/``, so the repeated text is embedded once; the per-message
classifiers are not cached and are the bulk of the time.

Outputs (per split, under outputs/features_cumulative/output/):
  * ``conv/{split}_conv_level.csv`` - one row per cumulative conversation

Run:  python scripts/09_extract_cumulative_features.py [--split learn] [--force]
"""

import argparse
import os

import pandas as pd
from team_comm_tools import FeatureBuilder

from config import DATA_PROCESSED, FEATURES_CUMULATIVE, OUTPUTS, ROOT, SPLITS, VECTOR_CACHE

LOG_BASE = OUTPUTS.name

# Identical to step 2's list, so the cumulative block is described by the same
# constructs as the per-round blocks.
CUSTOM_FEATURES = [
    "(BERT) Mimicry",
    "Moving Mimicry",
    "Forward Flow",
    "Discursive Diversity",
]


def vector_directory():
    for sub in ("sentence", "sentiment"):
        (VECTOR_CACHE / sub).mkdir(parents=True, exist_ok=True)
    return str(VECTOR_CACHE) + "/"


def output_paths(split):
    return {level: FEATURES_CUMULATIVE / "output" / level / f"{split}_{level}_level.csv"
            for level in ("chat", "user", "conv")}


def featurize(split, force=False):
    paths = output_paths(split)
    if not force and paths["conv"].exists():
        print(f"[{split}] cumulative conv output already exists; pass --force")
        return

    os.chdir(ROOT)
    chat = pd.read_csv(DATA_PROCESSED / f"chat_cumulative_{split}.csv", low_memory=False)

    ts = pd.to_datetime(chat["timestamp"], utc=True, errors="coerce")
    chat = chat.loc[ts.notna()].copy()
    chat["timestamp"] = (ts.loc[ts.notna()].astype("int64") // 10**6)

    print(f"[{split}] featurizing {len(chat)} message-rows from "
          f"{chat.conv_id.nunique()} cumulative conversations "
          f"(all columns; selection comes from the manifest)", flush=True)

    FeatureBuilder(
        input_df=chat,
        conversation_id_col="conv_id",
        speaker_id_col="playerId",
        message_col="text",
        timestamp_col="timestamp",
        vector_directory=vector_directory(),
        output_file_base=LOG_BASE,
        output_file_path_chat_level=str(paths["chat"]),
        output_file_path_user_level=str(paths["user"]),
        output_file_path_conv_level=str(paths["conv"]),
        custom_features=CUSTOM_FEATURES,
        turns=False,
        drop_redundant_columns=False,   # see the module docstring
        treat_zero_as_na=False,
    ).featurize()

    conv = pd.read_csv(paths["conv"], low_memory=False)
    print(f"[{split}] cumulative conversation-level output: {conv.shape[0]} rows x "
          f"{conv.shape[1]} columns -> {paths['conv']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", choices=SPLITS)
    args = ap.parse_args()
    for split in ([args.split] if args.split else SPLITS):
        featurize(split, force=args.force)
