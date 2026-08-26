"""Step 7 - show which words a LIWC-style lexicon feature actually counted.

A column like `relative_lexical_wordcount` reports how many words in a message
matched a category, but not which ones. The toolkit ships the categories as
regular expressions, so the matches can be recovered exactly rather than guessed.

Run:  python scripts/07_lexicon_words.py relative [--top 40]
      python scripts/07_lexicon_words.py --list
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from config import DATA_PROCESSED, TABLES

LEXICONS = (Path(__file__).resolve().parents[2] / ".venv")  # placeholder, resolved below
BLOCK, STAGE = "post", "opening"


def load_lexicons():
    """The toolkit's compiled lexicon dictionary, wherever it is installed."""
    import team_comm_tools
    path = (Path(team_comm_tools.__file__).parent / "features" / "assets"
            / "lexicons_dict.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def opening_messages(split="learn"):
    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv")
    rounds = pd.read_csv(DATA_PROCESSED / f"rounds_{split}.csv")
    opening = set(rounds.loc[rounds["stage_absolute"] == STAGE, f"conv_id_{BLOCK}"])
    return chat[(chat["block"] == BLOCK) & (chat["conv_id"].isin(opening))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("category", nargs="?", help="lexicon category, e.g. relative")
    ap.add_argument("--list", action="store_true", help="list every category")
    ap.add_argument("--top", type=int, default=40, help="how many words to show")
    args = ap.parse_args()

    lexicons = load_lexicons()
    if args.list or not args.category:
        print(f"{len(lexicons)} categories:")
        for name in sorted(lexicons):
            print(f"  {name}")
        raise SystemExit(0)

    key = args.category if args.category in lexicons else None
    if key is None:                      # tolerate the LIWC spelling, e.g. "relativ"
        matches = [k for k in lexicons if k.startswith(args.category[:6])]
        if not matches:
            raise SystemExit(f"no category matching {args.category!r}; "
                             f"use --list to see them all")
        key = matches[0]

    pattern = re.compile(lexicons[key], flags=re.IGNORECASE)
    messages = opening_messages()
    counts, examples = Counter(), {}
    for text in messages["text"].astype(str):
        for hit in pattern.findall(text.lower()):
            word = hit if isinstance(hit, str) else hit[0]
            counts[word] += 1
            examples.setdefault(word, text)

    total = sum(counts.values())
    print(f"category '{key}' matched {total} words across {len(messages)} messages "
          f"in {messages.conv_id.nunique()} opening-round conversations\n")
    print(f"{'word':<16}{'count':>7}   example message")
    print("-" * 78)
    for word, n in counts.most_common(args.top):
        print(f"{word:<16}{n:>7}   {examples[word][:48]}")

    out = pd.DataFrame({"word": list(counts), "count": list(counts.values())})
    out = out.sort_values("count", ascending=False)
    out.to_csv(TABLES / f"lexicon_matches_{key}.csv", index=False)
    print(f"\nfull list written to {TABLES / f'lexicon_matches_{key}.csv'}")
