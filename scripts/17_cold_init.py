"""Step 17 - time FeatureBuilder initialization on an empty vector cache.

The runtime table (step 16) reads initialization time out of the run log, and the
per-round calls in that log started in about a second because an earlier run had
already left SBERT sentence vectors and RoBERTa sentiment scores in
``outputs/vector_cache/`` for the same input. The cumulative calls, whose larger
input did not match what was cached, paid the full regeneration. Reported side by
side those two are not the same measurement.

This script measures the missing half: it clears the cached vectors for a split
and times constructing a ``FeatureBuilder`` with exactly the arguments step 2
uses, stopping before ``featurize()`` so that only initialization is timed. The
cleared cache files are moved to ``outputs/vector_cache_backup/`` rather than
deleted, so a run can be undone.

Output:
  * ``outputs/tables/runtime_cold_init.csv`` - one row per split, appended to

Run:  python scripts/17_cold_init.py --split learn
"""

import argparse
import importlib.util
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DATA_PROCESSED, OUTPUTS, ROOT, SPLITS, TABLES, VECTOR_CACHE

# Step 2 holds the FeatureBuilder arguments; import it by path because its module
# name starts with a digit.
_spec = importlib.util.spec_from_file_location(
    "extract_features", Path(__file__).resolve().parent / "02_extract_features.py")
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)

CACHE_BACKUP = OUTPUTS / "vector_cache_backup"
OUT = TABLES / "runtime_cold_init.csv"


def clear_cache(split):
    """Move this split's cached vectors aside and report what was moved.

    The toolkit names each cache file after the chat-level output file, so the two
    caches a split uses are ``sentence/chats/{split}_chat_level.csv`` and
    ``sentiment/chats/{split}_chat_level.csv``.
    """
    name = f"{split}_chat_level.csv"
    moved = []
    for sub in ("sentence", "sentiment"):
        src = VECTOR_CACHE / sub / "chats" / name
        if src.exists():
            dst = CACHE_BACKUP / sub / "chats" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(f"{sub}/{name}")
    return moved


def time_init(split):
    os.chdir(ROOT)

    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv")
    ts = pd.to_datetime(chat["timestamp"], utc=True, errors="coerce")
    chat = chat.loc[ts.notna()].copy()
    chat["timestamp"] = (ts.loc[ts.notna()].astype("int64") // 10**6)

    paths = extract.output_paths(split)
    moved = clear_cache(split)
    print(f"[{split}] cleared {len(moved)} cache file(s): {', '.join(moved) or 'none'}")
    print(f"[{split}] initializing on {len(chat)} message rows "
          f"({chat.conv_id.nunique()} conversations)")

    started = datetime.now()
    clock = time.perf_counter()
    extract.FeatureBuilder(
        input_df=chat,
        conversation_id_col="conv_id",
        speaker_id_col="playerId",
        message_col="text",
        timestamp_col="timestamp",
        vector_directory=extract.vector_directory(),
        output_file_base=extract.LOG_BASE,
        output_file_path_chat_level=str(paths["chat"]),
        output_file_path_user_level=str(paths["user"]),
        output_file_path_conv_level=str(paths["conv"]),
        custom_features=extract.CUSTOM_FEATURES,
        turns=False,
        drop_redundant_columns=(split == "learn"),
        corr_thresh=extract.CORR_THRESH,
        min_na_ratio=extract.MIN_NA_RATIO,
        min_zero_ratio=extract.MIN_ZERO_RATIO,
        treat_zero_as_na=False,
    )
    seconds = time.perf_counter() - clock

    row = {
        "windows": "Pre, Post and Window",
        "split": split,
        "started": started.isoformat(sep=" ", timespec="seconds"),
        "message_rows": len(chat),
        "conversations": int(chat.conv_id.nunique()),
        "cold_init_seconds": round(seconds, 2),
    }
    frame = pd.DataFrame([row])
    if OUT.exists():
        prior = pd.read_csv(OUT)
        prior = prior[prior["split"] != split]
        frame = pd.concat([prior, frame], ignore_index=True)
    frame.to_csv(OUT, index=False)
    print(f"[{split}] cold initialization took {seconds:.2f} seconds -> {OUT.name}")
    return seconds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, help="run a single split")
    args = ap.parse_args()
    for split in ([args.split] if args.split else SPLITS):
        time_init(split)
