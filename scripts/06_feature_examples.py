"""Step 6 - write the conversations behind each reported feature, for reading.

A feature name describes a construct; it does not tell you what that construct
looked like in this dataset. This writes one plain-text file per feature holding
the actual conversations that score highest and lowest on it, so the construct can
be judged from the messages rather than from its name.

Scope matches the finding the case study reports: talk in the previous round,
after revealing its outcome, in the first three rounds of a game. Features are the
ones the figures report - the strongest predictors within that cell.

Run:  python scripts/06_feature_examples.py [--n 15] [--features 14]

Writes outputs/examples/<rank>_<feature>.txt, one per feature.
"""

import argparse
import re
import textwrap

import pandas as pd

from config import DATA_PROCESSED, OUTPUTS, TABLES

BLOCK = "post"
STAGE = "opening"
EXAMPLES = OUTPUTS / "examples"

# How a column reaches the conversation level, spelled out for the file header.
AGG_ROUTE = {
    "mean": "averaged over", "max": "the largest value across",
    "min": "the smallest value across", "stdev": "the spread across",
    "sum": "summed over", "gini": "",
}


def describe_column(column):
    """Plain-English account of what one column is measuring."""
    parts, name = [], column
    while True:
        head = name.split("_", 1)[0]
        if head in AGG_ROUTE and "_" in name and head != "gini":
            rest = name.split("_", 1)[1]
            if rest.startswith("user_"):
                parts.append(f"{AGG_ROUTE[head]} speakers, of")
                name = rest[len("user_"):]
            else:
                parts.append(f"{AGG_ROUTE[head]} the messages, of")
                name = rest
        else:
            break
    construct = name.replace("_", " ")
    return (" ".join(parts) + " " + construct).strip() if parts else construct


def load(split="learn"):
    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv")
    rounds = pd.read_csv(DATA_PROCESSED / f"rounds_{split}.csv")
    analysis = pd.read_csv(DATA_PROCESSED / f"analysis_{split}.csv", low_memory=False)

    described = analysis[analysis[f"has_features_{BLOCK}"].astype(bool)
                         & (analysis["stage_absolute"] == STAGE)]
    messages = (chat[chat["block"] == BLOCK]
                .sort_values(["conv_id", "timestamp"])
                .groupby("conv_id")
                .agg(messages=("text", list),
                     speakers=("avatar", list),
                     round_index=("round_index", "first")))
    return described, messages


def render(conv, value, contribution, width=78):
    """One conversation as indented, speaker-labelled lines.

    Speakers are named by the animal avatar they played under, which is what the
    participants themselves saw and used to address each other - several
    conversations only make sense with it ("sloth, we can maximize our earnings if
    you put in all 20").
    """
    labels = []
    fallback = {}
    for speaker in conv.speakers:
        name = str(speaker).strip()
        if not name or name.lower() == "nan":
            name = fallback.setdefault(speaker, f"speaker {len(fallback) + 1}")
        labels.append(name)
    pad = max((len(label) for label in labels), default=0)

    out = [f"  round {int(conv.round_index)} | {len(conv.messages)} messages | "
           f"{len(set(labels))} speakers | feature z = {value:+.2f} | "
           f"next-round contribution {contribution:.2f}"]
    for label, message in zip(labels, conv.messages):
        body = textwrap.wrap(str(message), width - pad - 8) or [""]
        out.append(f"    {label:>{pad}}: {body[0]}")
        out.extend(" " * (pad + 6) + line for line in body[1:])
    return "\n".join(out)


def write_feature(rank, feature, stats, described, messages, n, width=78):
    column = f"{feature}__{BLOCK}"
    frame = described[[f"conv_id_{BLOCK}", column, "contribution_rate"]].rename(
        columns={f"conv_id_{BLOCK}": "conv_id", column: "value"})
    frame = frame.merge(messages, on="conv_id").dropna(subset=["value"])

    slug = re.sub(r"[^a-z0-9]+", "_", feature.lower()).strip("_")
    path = EXAMPLES / f"{rank:02d}_{slug}.txt"
    lines = [
        "=" * width,
        f"{feature}",
        "=" * width,
        "",
        f"measures : {describe_column(feature)}",
        f"scope    : talk in the previous round, after revealing its outcome,",
        f"           in the first three rounds of a game",
        f"sample   : {len(frame)} conversations",
        "",
        f"effect on the next round's contribution, per SD of this feature:",
        f"           learning split  {stats.coef_learn:+.4f}  (p = {stats.p_learn:.4f})",
        f"           held-out split  "
        + (f"{stats.coef_val:+.4f}  (p = {stats.p_val:.4f})"
           if pd.notna(stats.coef_val) else "not estimable"),
        "",
        "Conversations are ordered by the feature; the highest and lowest scoring",
        f"{n} are shown. Speaker labels are per conversation, not per person.",
        "",
    ]
    for label, subset in ((f"HIGHEST {n}", frame.nlargest(n, "value")),
                          (f"LOWEST {n}", frame.nsmallest(n, "value"))):
        lines += ["-" * width, label, "-" * width, ""]
        for row in subset.itertuples():
            lines += [render(row, row.value, row.contribution_rate, width), ""]
    path.write_text("\n".join(lines))
    return path, len(frame)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="conversations per extreme")
    ap.add_argument("--features", type=int, default=14,
                    help="how many of the strongest features to write")
    args = ap.parse_args()

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for stale in EXAMPLES.glob("*.txt"):
        stale.unlink()

    effects = pd.read_csv(TABLES / "stage_feature_effects.csv")
    cell = effects[(effects.block == BLOCK) & (effects.stage == STAGE)]
    top = cell.nsmallest(args.features, "p_learn")

    described, messages = load()
    print(f"{len(described)} opening-round conversations; writing {len(top)} features")
    for rank, stats in enumerate(top.itertuples(), start=1):
        path, n_convs = write_feature(rank, stats.feature, stats, described,
                                      messages, args.n)
        print(f"  {rank:2d}. {stats.feature:<46} -> {path.name}")
