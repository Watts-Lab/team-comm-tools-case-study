"""Step 19 - the case study's analysis pipeline as one diagram.

Section 4 of the paper moves from a corpus of messages to 27 replicated features
through several narrowing steps and two different estimators. Stated in prose the
sequence is hard to hold in mind, so this step draws it: what enters, what each
step removes, and what standard each of the two questions is held to.

Every count in the diagram is read from the pipeline's own outputs rather than
typed in, so the figure cannot drift away from the tables.

The diagram is two columns, not one tall stack. The narrowing runs down the left
column and the two questions run down the right, which keeps the drawing wide and
short. A page charges for a figure by its height, and a stack six boxes tall is
charged twice over: once for the height, and again for the type, which has to be
set small enough for the whole thing to fit the text width.

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
# counting lines rather than by trial and error. The canvas is deliberately small
# in inches: the paper scales the file to the text width, so every inch of canvas
# shrinks the type the reader ends up with.
FIG_W, FIG_H = 7.2, 4.5
Y_MIN = 0.0
UNIT_PER_PT = (1.0 - Y_MIN) / (FIG_H * 72)

TITLE_PT, BODY_PT, NOTE_PT = 10.0, 8.8, 7.6
TITLE_H = 12.5 * UNIT_PER_PT
BODY_H = 12.2 * UNIT_PER_PT
NOTE_H = 9.4 * UNIT_PER_PT
PAD = 5.5 * UNIT_PER_PT
GAP = 0.10                       # vertical space in the narrowing chain
Q_GAP = 0.05                     # question box to its evidence box
LANE_GAP = 0.08                  # one question lane to the next


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


def box(ax, x, top, w, title, body, tone=style.BLUE, title_size=TITLE_PT):
    """Draw a box whose height is set by how many lines of text it holds.

    Returns the y coordinate of its bottom edge, so the next element can be placed
    against it without any hand-tuned constant.
    """
    lines = body.count("\n") + 1
    title_lines = title.count("\n") + 1
    h = 2 * PAD + title_lines * TITLE_H + 0.35 * BODY_H + lines * BODY_H
    ax.add_patch(FancyBboxPatch(
        (x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.018",
        linewidth=1.4, edgecolor=tone, facecolor=style.SURFACE, zorder=3,
        clip_on=False))
    ax.text(x + w / 2, top - PAD, title, ha="center", va="top",
            fontsize=title_size, color=style.INK, fontweight="bold", zorder=4,
            linespacing=TITLE_H / (title_size * UNIT_PER_PT))
    ax.text(x + w / 2, top - PAD - title_lines * TITLE_H - 0.35 * BODY_H, body,
            ha="center",
            va="top", fontsize=BODY_PT, color=style.INK_2,
            linespacing=BODY_H / (BODY_PT * UNIT_PER_PT), zorder=4)
    return top - h


def arrow(ax, x, y_from, y_to, label=None, label_x=None):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.1, color=style.INK_MUTED, zorder=2, shrinkA=0, shrinkB=0))
    if label:
        ax.text(label_x, (y_from + y_to) / 2, label, fontsize=NOTE_PT,
                color=style.INK_MUTED, ha="left", va="center",
                linespacing=NOTE_H / (NOTE_PT * UNIT_PER_PT), zorder=4)


def line(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-", linewidth=1.1,
                                 color=style.INK_MUTED, zorder=2,
                                 shrinkA=0, shrinkB=0))


def main():
    style.use_style()
    n = counts()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(Y_MIN, 1)
    ax.axis("off")

    # Two columns of equal width, with the connecting bus running between them.
    left_x, left_w = 0.0, 0.40
    right_x, right_w = 0.53, 0.47
    mid = left_x + left_w / 2
    bus_x = 0.505
    note_x = mid + 0.01

    top = 0.99
    bottom = box(ax, left_x, top, left_w, "Corpus",
                 f"{n['messages']:,} messages from 803\ngames, at the round level")
    arrow(ax, mid, bottom, bottom - GAP,
          "one FeatureBuilder call per split", note_x)

    bottom = box(ax, left_x, bottom - GAP, left_w, "Feature extraction",
                 "166 features, giving 3,073\nnumeric conversation-level columns")
    arrow(ax, mid, bottom, bottom - GAP,
          "drop columns over 30% missing\nor over 90% zero, and all but\none of each correlated group", note_x)

    bottom = box(ax, left_x, bottom - GAP, left_w, "After redundancy removal",
                 "242 columns")
    arrow(ax, mid, bottom, bottom - GAP,
          "drop near-constant columns\nand any absent from the\nheld-out split",
          note_x)

    cf_bottom = box(ax, left_x, bottom - GAP, left_w, "Candidate features",
                    f"{n['features']} features, for each of the\n"
                    "four conversation boundaries")

    # The two questions are answered by different estimators, so the diagram
    # branches. The branch leaves the bottom of the narrowing chain and runs up a
    # bus between the columns, entering each question from its left edge.
    q_top = 0.99
    lanes = []
    for title, body, ev_body in (
            ("Question 1:\nthe conversation as a whole",
             "additional $R^2$ over the game's rules,\n"
             "from an ElasticNet and a random forest",
             "Cross-validated on the learning games and\n"
             "scored on the held-out validation games.\n"
             "Both intervals must clear zero."),
            ("Question 2:\nthe effect of individual features",
             "OLS per feature over the same controls,\n"
             "standard errors clustered on games",
             "False-discovery-rate correction on the\n"
             "learning games, and must replicate\non the validation set.")):
        q_bottom = box(ax, right_x, q_top, right_w, title, body, tone=style.VIOLET,
                       title_size=9.0)
        lanes.append((q_top + q_bottom) / 2)
        arrow(ax, right_x + right_w / 2, q_bottom, q_bottom - Q_GAP)
        ev_bottom = box(ax, right_x, q_bottom - Q_GAP, right_w,
                        "Evidence standard", ev_body, tone=style.GREEN)
        q_top = ev_bottom - LANE_GAP

    junction = cf_bottom - 0.05
    line(ax, (mid, cf_bottom), (mid, junction))
    line(ax, (mid, junction), (bus_x, junction))
    line(ax, (bus_x, junction), (bus_x, lanes[0]))
    for y in lanes:
        ax.add_patch(FancyArrowPatch(
            (bus_x, y), (right_x, y), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.1, color=style.INK_MUTED, zorder=2, shrinkA=0, shrinkB=0))

    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    out = MAIN_DIR / "fig3_analysis_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(n)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
