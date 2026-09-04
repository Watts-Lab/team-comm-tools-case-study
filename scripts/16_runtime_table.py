"""Step 16 - the FeatureBuilder runtime table, parsed from the real run log.

The paper reports how long the extraction actually took. That number has to come
from the log the runs wrote, not from memory, so this script parses
``outputs/logs/feature_builder.log`` and emits the table the paper includes.

The log records one block per ``FeatureBuilder`` call, delimited by

    === Start Initializing FeatureBuilder for <file> ===
    === Initialization Complete for <file> ===
    Data file has <n> lines (chats), <n> unique speakers, <n> unique conversations.
      - <feature>: <n> seconds.            (one per extraction step)
    === Featurization Completed for <file> in <n> seconds! ===

Two costs are reported separately because they behave differently:

  initialization   loading the input and, on a cache miss, regenerating the SBERT
                   sentence vectors and the RoBERTa sentiment scores. This is the
                   expensive-but-cached half: it is seconds when the vector cache
                   matches the input and minutes when it does not.
  featurization    the feature extraction itself, which is paid on every run.

The log's per-round calls found a warm cache and so initialized in about a second,
while its cumulative calls did not and paid the full regeneration. Reporting those
two together would compare a cache hit against a cache miss, so the table uses the
cold initialization times step 17 measured on a cleared cache
(``outputs/tables/runtime_cold_init.csv``) wherever the log recorded a hit. The
cached times the log did record are kept in ``runtime.csv`` and printed below.

Which chat table each call featurized is not in the log, so it is recovered by
matching the logged conversation count against the processed chat tables rather
than by assuming the runs happened in a particular order.

Outputs:
  * ``outputs/tables/runtime.csv``       - one row per FeatureBuilder call
  * ``outputs/tables/runtime_steps.csv`` - per-feature costs within each call
  * ``outputs/tables/runtime_table.tex`` - the table body the paper inputs

Run:  python scripts/16_runtime_table.py
"""

import re
from datetime import datetime

import pandas as pd

from config import DATA_PROCESSED, OUTPUTS, SPLITS, TABLES

LOG = OUTPUTS / "logs" / "feature_builder.log"
COLD_INIT = TABLES / "runtime_cold_init.csv"

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
RE_INIT_START = re.compile(TS + r" INFO === Start Initializing FeatureBuilder for (\S+) ===")
RE_INIT_DONE = re.compile(TS + r" INFO === Initialization Complete for (\S+) ===")
RE_DATA = re.compile(r"Data file has (\d+) lines \(chats\), (\d+) unique speakers, "
                     r"(\d+) unique conversations\.")
RE_DONE = re.compile(r"=== Featurization Completed for (\S+) in ([\d.]+) seconds! ===")
RE_STEP = re.compile(r"INFO   - (\w+): ([\d.]+) seconds\.")
RE_LEVEL = re.compile(r"INFO --- (Chat|User|Conversation) Level Features ---")
RE_REGEN = re.compile(r"(Sentence vectors|BERT vectors) regeneration completed in ([\d.]+) seconds\.")


def stamp(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S,%f")


def parse(path):
    """One record per FeatureBuilder call, in log order."""
    runs, cur = [], None
    for line in path.read_text().splitlines():
        m = RE_INIT_START.search(line)
        if m:
            cur = {"input_file": m.group(2), "init_start": stamp(m.group(1)),
                   "regen_seconds": 0.0, "steps": [], "level": None}
            runs.append(cur)
            continue
        if cur is None:
            continue

        m = RE_REGEN.search(line)
        if m:
            cur["regen_seconds"] += float(m.group(2))
        m = RE_INIT_DONE.search(line)
        if m:
            cur["init_seconds"] = (stamp(m.group(1)) - cur["init_start"]).total_seconds()
        m = RE_DATA.search(line)
        if m:
            cur.update(message_rows=int(m.group(1)), speakers=int(m.group(2)),
                       conversations=int(m.group(3)))
        m = RE_LEVEL.search(line)
        if m:
            cur["level"] = m.group(1)
        m = RE_STEP.search(line)
        if m:
            cur["steps"].append({"level": cur["level"], "step": m.group(1),
                                 "seconds": float(m.group(2))})
        m = RE_DONE.search(line)
        if m:
            cur["featurize_seconds"] = float(m.group(2))
            cur = None
    return [r for r in runs if "featurize_seconds" in r]


def conversation_counts():
    """{n_conversations: (block label, split)} from the processed chat tables.

    The log names only ``learn.csv``/``val.csv``, which both extractions share, so
    the conversation count is what distinguishes a per-round call from a cumulative
    one.
    """
    index = {}
    for split in SPLITS:
        per_round = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv",
                                low_memory=False, usecols=["conv_id", "block"])
        # A message belongs to its Pre or Post window and again to Window, so the
        # distinct-message count is the pre/post half of the table.
        distinct = int(per_round["block"].isin(["pre", "post"]).sum())
        index[per_round["conv_id"].nunique()] = ("Pre, Post and Window", split, distinct)
        cumulative = DATA_PROCESSED / f"chat_cumulative_{split}.csv"
        if cumulative.exists():
            cum = pd.read_csv(cumulative, low_memory=False, usecols=["conv_id"])
            index[cum["conv_id"].nunique()] = ("Cumulative", split, distinct)
    return index


def mmss(seconds):
    """m:ss, rolling over to h:mm:ss once a duration passes the hour."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def cold_init(table):
    """Give every call an initialization time paid on an empty vector cache.

    A call whose log block shows a vector regeneration already missed the cache, so
    its logged time is the cold one. A call that hit the cache takes its time from
    the measurement step 17 wrote for the same window and split; without that
    measurement the cell is left empty and the table cannot be built.
    """
    measured = {}
    if COLD_INIT.exists():
        for _, r in pd.read_csv(COLD_INIT).iterrows():
            measured[(r.windows, r.split)] = float(r.cold_init_seconds)

    hit = table["vector_regen_seconds"] == 0
    cold = pd.Series([measured.get((w, s)) for w, s in
                      zip(table["windows"], table["split"])],
                     index=table.index, dtype="float64")
    table["cached_init_seconds"] = table["logged_init_seconds"].where(hit)
    table["init_seconds"] = table["logged_init_seconds"].mask(hit, cold)

    missing = table[table["init_seconds"].isna()]
    if len(missing):
        raise SystemExit(
            f"{len(missing)} call(s) hit the vector cache and have no cold-init "
            f"measurement. Run scripts/17_cold_init.py first.")

    table["init_seconds"] = table["init_seconds"].round(2)
    table["total_seconds"] = (table["init_seconds"]
                              + table["featurize_seconds"]).round(2)
    return table


def build():
    if not LOG.exists():
        raise SystemExit(f"No run log at {LOG} - nothing to report.")

    runs = parse(LOG)
    index = conversation_counts()

    rows, steps = [], []
    for run in runs:
        label, split, distinct = index.get(run["conversations"],
                                           ("unmatched", "?", 0))
        rows.append({
            "windows": label,
            "split": split,
            "input_file": run["input_file"],
            "started": run["init_start"].isoformat(sep=" ", timespec="seconds"),
            "message_rows": run["message_rows"],
            "distinct_messages": distinct,
            "conversations": run["conversations"],
            "speakers": run["speakers"],
            "logged_init_seconds": round(run["init_seconds"], 2),
            "vector_regen_seconds": round(run["regen_seconds"], 2),
            "featurize_seconds": run["featurize_seconds"],
        })
        for step in run["steps"]:
            steps.append({"windows": label, "split": split, **step})

    table = cold_init(pd.DataFrame(rows))
    unmatched = table[table["windows"] == "unmatched"]
    if len(unmatched):
        print(f"  ! {len(unmatched)} run(s) did not match a chat table on "
              f"conversation count; reported as 'unmatched'")

    # Per-round calls first, then cumulative, learning split before validation.
    order = {"Pre, Post and Window": 0, "Cumulative": 1, "unmatched": 2}
    table = (table.assign(_o=table["windows"].map(order),
                          _s=table["split"].map({"learn": 0, "val": 1}))
             .sort_values(["_o", "_s"]).drop(columns=["_o", "_s"]))

    table.to_csv(TABLES / "runtime.csv", index=False)
    pd.DataFrame(steps).to_csv(TABLES / "runtime_steps.csv", index=False)

    write_tex(table, pd.DataFrame(steps))

    print(f"[runtime] {len(table)} FeatureBuilder call(s) from {LOG.name}")
    for _, r in table.iterrows():
        cached = ("" if pd.isna(r.cached_init_seconds)
                  else f" (cached {mmss(r.cached_init_seconds)})")
        print(f"  {r.windows:<21} {r.split:<6} {r.message_rows:>7,} rows "
              f"({r.distinct_messages:>6,} distinct)  {r.conversations:>5,} convs  "
              f"init {mmss(r.init_seconds):>6}{cached}  "
              f"featurize {mmss(r.featurize_seconds):>6}")
    print(f"         wrote runtime.csv, runtime_steps.csv, runtime_table.tex")
    return table


SPLIT_NAME = {"learn": "Learning", "val": "Validation"}


def write_tex(table, steps):
    """The table body the paper \\input{}s, so the numbers cannot drift from the log."""
    lines = [
        "% Generated by scripts/16_runtime_table.py from outputs/logs/feature_builder.log.",
        "% Do not edit by hand; re-run the script instead.",
        "\\begin{tabular}{p{3.6cm}p{1.4cm}rrrrrr}",
        "\\toprule",
        "\\textbf{Conversation Boundary} & \\textbf{Split} & \\textbf{\\# Messages} & "
        "\\textbf{Message Rows} & \\textbf{Convs.} & \\textbf{Init.} & "
        "\\textbf{Featurize} & \\textbf{Total} \\\\",
        "\\midrule",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r.windows} & {SPLIT_NAME.get(r.split, r.split)} & "
            f"{r.distinct_messages:,} & {r.message_rows:,} & "
            f"{r.conversations:,} & {mmss(r.init_seconds)} & "
            f"{mmss(r.featurize_seconds)} & {mmss(r.total_seconds)} \\\\")
    lines += ["\\midrule",
              f"\\textbf{{All calls}} & & & & & & & "
              f"\\textbf{{{mmss(table.total_seconds.sum())}}} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (TABLES / "runtime_table.tex").write_text("\n".join(lines) + "\n")

    # The facts the surrounding prose quotes, so those can be checked too. Which
    # step dominates is not the same in every call, so it is reported per call
    # rather than pooled.
    if len(steps):
        print("         slowest step per call:")
        for (windows, split), g in steps.groupby(["windows", "split"]):
            worst = g.loc[g["seconds"].idxmax()]
            print(f"           {windows:<21} {split:<6} {worst.step} "
                  f"({worst.seconds:.0f}s, {worst.seconds / g.seconds.sum():.0%} "
                  f"of featurization)")


if __name__ == "__main__":
    build()
