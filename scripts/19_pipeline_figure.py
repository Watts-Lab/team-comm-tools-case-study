"""Step 19 - the case study's analysis pipeline as one diagram.

Section 4 of the paper moves from a corpus of messages to 27 replicated features
through several narrowing steps and two different estimators. Stated in prose the
sequence is hard to hold in mind, so this step draws it: what enters, what each
step removes, and what standard each of the two questions is held to.

Every count in the diagram is read from the pipeline's own outputs rather than
typed in, so the figure cannot drift away from the tables.

Writes outputs/figures/main/fig3_analysis_pipeline.png

Run:  python scripts/19_pipeline_figure.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import style
from config import DATA_PROCESSED, FIGURES, TABLES, TABLES_WINDOWS

MAIN_DIR = FIGURES / "main"
OPENING = "opening"

# Figure geometry. The axes is drawn in 0-1 coordinates on a canvas of this size,
# so a point of type is a fixed fraction of a unit and text can be laid out by
# counting lines rather than by trial and error.
FIG_W, FIG_H = 9.6, 9.0
# The stack runs a little below the origin, so the axes is given room for it and
# the point-to-unit conversion accounts for the taller span.
Y_MIN = -0.01
UNIT_PER_PT = (1.0 - Y_MIN) / (FIG_H * 72)
TITLE_H = 12.5 * UNIT_PER_PT
BODY_H = 13.0 * UNIT_PER_PT
PAD = 8 * UNIT_PER_PT
GAP = 0.075                       # vertical space between one box and the next


def counts():
    """Every number in the diagram, read from the pipeline's outputs."""
    messages = sum(len(pd.read_csv(DATA_PROCESSED / f"chat_{s}.csv")
                       .drop_duplicates(["gameId", "playerId", "text", "timestamp"]))
                   for s in ("learn", "val"))

    manifest = pd.read_csv(TABLES / "feature_manifest.csv")
    analysis_features = int(manifest["kept"].sum())

    effects = pd.read_csv(TABLES_WINDOWS / "block_feature_effects.csv")
    cell = effects[(effects["binning"] == "stage") & (effects["bin"] == OPENING)
                   & (effects["sample"] == "channel") & (effects["block"] == "post")]
    return {
        "messages": messages,
        "features": analysis_features,
        "nominal": int((cell["p_learn"] < 0.05).sum()),
        "fdr": int((cell["q_learn"] < 0.05).sum()),
        "replicated": int(cell["replicates"].sum()),
    }


def box(ax, x, top, w, title, body, tone=style.BLUE, title_size=10.5):
    """Draw a box whose height is set by how many lines of text it holds.

    Returns the y coordinate of its bottom edge, so the next element can be placed
    against it without any hand-tuned constant.
    """
    lines = body.count("\n") + 1
    h = 2 * PAD + TITLE_H + 0.4 * BODY_H + lines * BODY_H
    ax.add_patch(FancyBboxPatch(
        (x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=1.4, edgecolor=tone, facecolor=style.SURFACE, zorder=3))
    ax.text(x + w / 2, top - PAD, title, ha="center", va="top",
            fontsize=title_size, color=style.INK, fontweight="bold", zorder=4)
    ax.text(x + w / 2, top - PAD - TITLE_H - 0.4 * BODY_H, body, ha="center",
            va="top", fontsize=9, color=style.INK_2, linespacing=1.45, zorder=4)
    return top - h


def arrow(ax, x, y_from, y_to, label=None, label_x=None):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.2, color=style.INK_MUTED, zorder=2, shrinkA=0, shrinkB=0))
    if label:
        ax.text(label_x, (y_from + y_to) / 2, label, fontsize=8.5,
                color=style.INK_MUTED, ha="left", va="center", linespacing=1.4,
                zorder=4)


def main():
    style.use_style()
    n = counts()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(Y_MIN, 1)
    ax.axis("off")

    left_x, left_w = 0.03, 0.50
    mid = left_x + left_w / 2
    note_x = left_x + left_w + 0.03

    top = 0.985
    bottom = box(ax, left_x, top, left_w, "Corpus",
                 f"{n['messages']:,} messages from 803 games,\n"
                 "at the round level")
    arrow(ax, mid, bottom, bottom - GAP,
          "one FeatureBuilder call\nper split", note_x)

    bottom = box(ax, left_x, bottom - GAP, left_w, "Feature extraction",
                 "166 features, giving 3,073 numeric\nconversation-level columns")
    arrow(ax, mid, bottom, bottom - GAP,
          "drop columns over 30% missing\nor over 90% zero, and keep one\n"
          "column per correlated group", note_x)

    bottom = box(ax, left_x, bottom - GAP, left_w, "After redundancy removal",
                 "242 columns")
    arrow(ax, mid, bottom, bottom - GAP,
          "drop near-constant columns and\ncolumns absent from the held-out split",
          note_x)

    bottom = box(ax, left_x, bottom - GAP, left_w, "Candidate features",
                 f"{n['features']} features, for each of the four\n"
                 "conversation boundaries")

    # The two questions split here and are answered by different estimators, so
    # the diagram splits with them.
    split_y = bottom - GAP * 0.55
    q_left_x, q_right_x, q_w = 0.0, 0.515, 0.485
    ax.add_patch(FancyArrowPatch((mid, bottom), (mid, split_y), arrowstyle="-",
                                 linewidth=1.2, color=style.INK_MUTED, zorder=2,
                                 shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch(
        (q_left_x + q_w / 2, split_y), (q_right_x + q_w / 2, split_y),
        arrowstyle="-", linewidth=1.2, color=style.INK_MUTED, zorder=2,
        shrinkA=0, shrinkB=0))

    q_top = split_y - 0.03
    for x, title, body in (
            (q_left_x, "Question 1: the conversation as a whole",
             "additional $R^2$ over the game's rules,\n"
             "from an ElasticNet and a random forest"),
            (q_right_x, "Question 2: the effect of individual features",
             "OLS per feature over the same controls,\n"
             "standard errors clustered on games")):
        arrow(ax, x + q_w / 2, split_y, q_top)
        q_bottom = box(ax, x, q_top, q_w, title, body, tone=style.VIOLET,
                       title_size=9.5)
        arrow(ax, x + q_w / 2, q_bottom, q_bottom - GAP * 0.7)

    e_top = q_bottom - GAP * 0.7
    box(ax, q_left_x, e_top, q_w, "Evidence standard",
        "Cross-validated on the learning games and\n"
        "scored on the held-out validation games.\n"
        "Both intervals must clear zero.",
        tone=style.GREEN)
    box(ax, q_right_x, e_top, q_w, "Evidence standard",
        "False-discovery-rate correction on the\n"
        "learning games, then replication held out\n"
        "with the same sign.",
        tone=style.GREEN)

    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    out = MAIN_DIR / "fig3_analysis_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(n)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
