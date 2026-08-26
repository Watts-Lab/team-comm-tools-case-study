"""Step 5 - draw the figures the case study reports.

  fig1  when in a game talk matters, under both definitions of a game stage
  fig2  which features predict contribution in the opening rounds
  fig3  which family of features carries the effect, in the opening rounds
  fig4  what makes opening talk different from later talk
  fig5  whether the same features predict later in the game

The analyses these leave out - the channel effect, and whether the gain comes from
speaking at all rather than from content - are still computed by 04_analysis.py and
still in outputs/tables/. They are left out of the figure set because they answer a
different question than this case study asks.

Every figure follows the rules documented at the top of scripts/style.py. In
particular: no explanatory prose in the figure, legend under the subtitle, nothing
overlapping. Explanation belongs in the README.

Run:  python scripts/05_figures.py   (after 04_analysis.py)
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, FIGURES, TABLES
from style import (AQUA, COLOR_CHANNEL, COLOR_LEARN, COLOR_NO_CHANNEL, COLOR_VAL,
                   GREEN, GRID, INK, INK_2, INK_MUTED, MAGENTA, ORANGE, VIOLET,
                   YELLOW, axis_direction, bold_label, header, panel_title, use_style)

use_style()

MODEL_COLORS = {"elastic net": COLOR_CHANNEL, "random forest": ORANGE}

# Absolute staging is primary; see the README for the definition.
STAGING = "absolute"
STAGE_ORDER = ["opening", "middle", "endgame"]
STAGE_COLORS = {"opening": COLOR_CHANNEL, "middle": AQUA, "endgame": ORANGE}

# The two talk blocks, both predicting the same round's contribution but drawn from
# different moments. PRE is this round's own deliberation, before anyone sees how it
# turns out; POST is the previous round's reaction, after its result was revealed.
# The stages above are a different axis entirely - they locate the *round* within its
# game - so the labels name the round explicitly to keep the two apart.
BLOCK_LABEL = {"pre": "Talk in this round, before its outcome",
               "post": "Talk in the previous round, after its outcome"}
BLOCK_SHORT = {"pre": "this round, pre-outcome", "post": "previous round, post-outcome"}
# Two-line form for axis ticks, where the single-line labels do not fit.
BLOCK_WRAPPED = {"pre": "Talk in this round,\nbefore its outcome",
                 "post": "Talk in the previous round,\nafter its outcome"}
BLOCK_COLORS = {"pre": COLOR_CHANNEL, "post": ORANGE}

FAMILY_SUFFIXES = {"_lexical_wordcount": " (LIWC)", "_politeness_convokit": " (politeness)",
                   "_receptiveness_yeomans": " (receptiveness)", "_bert": " (BERT)",
                   "_chats": "", "_conversation": ""}
AGG_WORDS = {"mean": "avg", "max": "max", "min": "min", "stdev": "SD", "sum": "total",
             "gini": "gini"}


def add_split_legend(ax):
    """Proxy handles so the marker shapes are read off the legend, not from prose."""
    ax.plot([], [], marker="o", linestyle="none", color=INK_MUTED, markersize=7,
            markeredgecolor="white", label="Learning split")
    ax.plot([], [], marker="D", linestyle="none", color=INK_MUTED, markersize=6,
            alpha=0.5, markeredgecolor="white", label="Held-out split")


def pretty(name):
    """Human-readable label: aggregation prefix in brackets, construct in words."""
    for old, new in FAMILY_SUFFIXES.items():
        name = name.replace(old, new)
    prefix = []
    while True:
        head = name.split("_", 1)[0]
        if head in AGG_WORDS and "_" in name:
            rest = name.split("_", 1)[1]
            if rest.startswith("user_"):
                prefix.append(f"{AGG_WORDS[head]} of speaker")
                name = rest[len("user_"):]
            else:
                prefix.append(AGG_WORDS[head])
                name = rest
        else:
            break
    label = name.replace("_", " ").strip()
    return f"{label} [{' '.join(prefix)}]" if prefix else label


# ------------------------------------------------------------------ fig 3 ---
def fig_family_importance():
    """Which family of features carries the opening-round effect."""
    fam = pd.read_csv(TABLES / "family_importance_opening.csv")
    n_convs = int(fam["n_conversations"].iloc[0])
    order = (fam[fam.model_family == "elastic net"]
             .sort_values("drop_in_cv_r2")["feature_family"].tolist())

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(order) + 2.8))
    height = 0.36
    for i, model in enumerate(MODEL_COLORS):
        sub = fam[fam.model_family == model].set_index("feature_family").reindex(order)
        y = np.arange(len(order)) + (i - 0.5) * height
        ax.barh(y, sub["drop_in_cv_r2"], height=height * 0.9,
                color=MODEL_COLORS[model], label=model, zorder=3)
        ax.scatter(sub["drop_in_heldout_r2"], y, s=32, color=INK, zorder=5,
                   edgecolor="white", linewidth=1.2,
                   label="Held-out split" if i == 0 else None)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    counts = (fam[fam.model_family == "elastic net"]
              .set_index("feature_family")["n_features"])
    ax.set_yticks(np.arange(len(order)),
                  [bold_label(f) + f"  ({counts[f]})" for f in order], fontsize=9.5)
    ax.set_xlabel("Change in R² when the family is removed")
    ax.grid(axis="y", visible=False)
    axis_direction(ax, "family was hurting accuracy", "family was helping accuracy")
    header(fig, ax, "Sentiment carries the opening-round effect",
           f"Talk in the previous round, after its outcome, in the first three "
           f"rounds ({n_convs:,} conversations). Feature count per family in "
           "parentheses. Families overlap, so the changes do not sum.",
           legend_from=ax, ncol=3)
    fig.savefig(FIGURES / "fig3_opening_families.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5 ---
# Families, in fixed order, for the one cell worth looking at closely.
FAMILY_COLORS = {"Sentiment & emotion": COLOR_CHANNEL, "Semantic dynamics": ORANGE,
                 "Lexical (LIWC)": AQUA, "Volume & form": YELLOW,
                 "Participation & timing": MAGENTA, "Receptiveness": GREEN,
                 "Questions & repair": VIOLET}


def fig_opening_reaction(top_n=14):
    """A closer look at the one stage and block where content predicts.

    Every other cell of the stage analysis is a null. This one - what a group says
    after seeing the result of an opening round - is where the toolkit's features
    carry signal, so it gets its own figure rather than one line on a summary chart.
    """
    eff = pd.read_csv(TABLES / "stage_feature_effects.csv")
    cell = eff[(eff.block == "post") & (eff.stage == "opening")]
    n_convs = int(cell["n_game_rounds"].iloc[0])
    # Grouped by family so the colours read as blocks rather than confetti, and
    # ordered within a family by strength of evidence.
    top = cell.nsmallest(top_n, "p_learn")
    family_rank = {fam: i for i, fam in enumerate(FAMILY_COLORS)}
    top = (top.assign(_fam=top["family"].map(family_rank))
           .sort_values(["_fam", "p_learn"]).iloc[::-1].reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(9.4, 0.42 * len(top) + 3.2))
    y = np.arange(len(top))
    seen = set()
    for yi, row in zip(y, top.itertuples()):
        color = FAMILY_COLORS.get(row.family, INK_MUTED)
        label = row.family if row.family not in seen else None
        seen.add(row.family)
        ax.hlines(yi, row.ci_low_learn, row.ci_high_learn, color=color, alpha=0.35,
                  linewidth=2.5, zorder=3)
        ax.scatter(row.coef_learn, yi, s=52, color=color, zorder=4,
                   edgecolor="white", linewidth=1.2, label=label)
        if not np.isnan(row.coef_val):
            ax.scatter(row.coef_val, yi, s=32, color=color, zorder=5, marker="D",
                       alpha=0.45, edgecolor="white", linewidth=1.0)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    ax.set_yticks(y, [bold_label(pretty(f)) for f in top["feature"]], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in contribution rate per SD of the feature, with 95% CI")
    ax.margins(y=0.04)
    add_split_legend(ax)
    header(fig, ax, "What predicts contribution in the first three rounds",
           f"Talk in the previous round, after its outcome ({n_convs:,} conversations).",
           legend_from=ax, ncol=4)
    fig.savefig(FIGURES / "fig2_opening_features.png")
    plt.close(fig)

# ------------------------------------------------------------------ fig 7 ---
def fig_stage_profile(top_n=10, block="post"):
    """How opening talk differs from talk later in the game."""
    prof = pd.read_csv(TABLES / "stage_profile.csv")
    stages = [c for c in STAGE_ORDER if c in prof.columns]
    sub = prof[prof["block"] == block].head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(sub) + 3.0))
    y = np.arange(len(sub))
    ax.hlines(y, sub[stages].min(axis=1), sub[stages].max(axis=1),
              color=GRID, linewidth=2.5, zorder=2)
    for stage in stages:
        ax.scatter(sub[stage], y, s=52, color=STAGE_COLORS[stage],
                   edgecolor="white", linewidth=1.2, zorder=4, label=stage)
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=3)
    ax.set_yticks(y, [bold_label(pretty(f)) for f in sub["feature"]], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Stage mean, in SDs from the average conversation")
    ax.margins(y=0.04)
    header(fig, ax, "What gets said early differs from what gets said later",
           "Talk in the previous round, after its outcome. Colour marks where the "
           "round being predicted sits in its game. Features shown are those that "
           "shift most between the first and last three rounds.",
           legend_from=ax, ncol=3)
    fig.savefig(FIGURES / "fig4_opening_talk_differs.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 8 ---
def fig_stage_feature_effects(top_n=8, block="post"):
    """Whether the features that predict early also predict later."""
    eff = pd.read_csv(TABLES / "stage_feature_effects.csv")
    eff = eff[eff.block == block]
    stages = [s for s in STAGE_ORDER if s in set(eff["stage"])]

    # The features that matter at the opening, followed across the rest of the game.
    opening = eff[(eff.stage == "opening")].nsmallest(top_n, "p_learn")
    picks = opening["feature"].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(8.8, 0.44 * len(picks) + 3.0))
    y = np.arange(len(picks))
    for yi, feature in zip(y, picks):
        by_stage = eff[eff.feature == feature].set_index("stage")
        vals = [by_stage.loc[st, "coef_learn"] for st in stages if st in by_stage.index]
        ax.hlines(yi, min(vals), max(vals), color=GRID, linewidth=2.5, zorder=2)
        for stage in stages:
            if stage not in by_stage.index:
                continue
            r = by_stage.loc[stage]
            sig = r["p_learn"] < 0.05
            ax.scatter(r["coef_learn"], yi, s=58 if sig else 34,
                       color=STAGE_COLORS[stage], zorder=4,
                       edgecolor="white" if sig else STAGE_COLORS[stage],
                       linewidth=1.3, alpha=1.0 if sig else 0.40,
                       label=stage if yi == y[0] else None)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=3)
    ax.set_yticks(y, [bold_label(pretty(f)) for f in picks], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in contribution rate per SD of the feature")
    ax.margins(y=0.05)
    header(fig, ax, "The features that predict early do not predict later",
           f"The {top_n} strongest opening-round features, re-estimated within each "
           "stage. Solid markers are significant within that stage; faded are not.",
           legend_from=ax, ncol=3)
    fig.savefig(FIGURES / "fig5_effects_across_stages.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4 ---
# The two ways of saying where a round sits in its game. Games run from 3 to 30
# rounds, so a fixed round number and a fraction of the game are far from
# interchangeable: the first third of a 30-round game runs to round 9, while the
# opening is over by round 2.
# Both schemes describe the same three positions, so the axis reads the same way in
# both rows and only the definition changes. The row header carries the definition.
STAGE_TICKS = ["beginning", "middle", "end"]
STAGE_SCHEMES = {
    "absolute": (["opening", "middle", "endgame"], "By round number",
                 "beginning is the first three rounds, end is the last three"),
    "relative": (["early", "middle", "late"], "By thirds",
                 "beginning, middle and end are each one third of the rounds played"),
}


def fig_staging_comparison():
    """Whether the stage pattern depends on how a stage is defined."""
    stage = pd.read_csv(TABLES / "round_stage.csv")

    # The gap has to clear the top row's two-line tick labels *and* the bottom
    # row's header, which sits above its axes.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8), sharey=True,
                             gridspec_kw={"wspace": 0.10, "hspace": 0.52})
    for row, (scheme, (order, scheme_name, definition)) in enumerate(
            STAGE_SCHEMES.items()):
        description = (rf"$\bf{{{scheme_name.replace(' ', chr(92) + ' ')}}}$"
                       f":  {definition}")
        present = [s for s in order if s in set(stage.loc[stage.staging == scheme,
                                                          "stage"])]
        for col, block in enumerate(BLOCK_LABEL):
            ax = axes[row, col]
            x = np.arange(len(present))
            for model, color in MODEL_COLORS.items():
                sub = (stage[(stage.staging == scheme)
                             & (stage.model_family == model)
                             & (stage.block == block)]
                       .set_index("stage").reindex(present))
                ax.plot(x, sub["delta_cv_r2_content"], color=color, marker="o",
                        markersize=6.5, markeredgecolor="white", markeredgewidth=1.3,
                        label=model, zorder=4)
                ax.fill_between(x, sub["ci_low"], sub["ci_high"], color=color,
                                alpha=0.13, linewidth=0, zorder=2)
            ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=3)
            n = (stage[(stage.staging == scheme) & (stage.model_family == "elastic net")
                       & (stage.block == block)]
                 .set_index("stage").reindex(present)["n_conversations"])
            ax.set_xticks(x, [f"{tick}\n({int(n[s]):,} convs)"
                              for tick, s in zip(STAGE_TICKS, present)], fontsize=9)
            ax.margins(x=0.14)
            if row == 0:
                panel_title(ax, BLOCK_LABEL[block])
        axes[row, 0].set_ylabel("Added variance explained (ΔR²)")
        # Row header sits above the panel titles in the top row, and directly above
        # the axes in the bottom row where there are none.
        axes[row, 0].annotate(description, xy=(0, 1), xycoords="axes fraction",
                              xytext=(0, 38 if row == 0 else 16),
                              textcoords="offset points", ha="left", va="bottom",
                              fontsize=11, color=INK, annotation_clip=False)

    header(fig, axes, "The pattern holds only under one definition of a game stage",
           "Conversation features added to a model of the game's rules, fitted "
           "separately within each stage. 10-fold cross-validation holding out whole "
           "games; bands are game-clustered bootstrap 95% intervals.",
           legend_from=axes[0, 0], ncol=2, panel_titles=True, extra_top=0.34)
    fig.savefig(FIGURES / "fig1_when_talk_matters.png")
    plt.close(fig)


FIGURES_TO_DRAW = [fig_staging_comparison, fig_opening_reaction,
                   fig_family_importance, fig_stage_profile,
                   fig_stage_feature_effects]

if __name__ == "__main__":
    failures = []
    for draw in FIGURES_TO_DRAW:
        try:
            draw()
        except Exception as exc:  # noqa: BLE001 - reported, then summarised below
            failures.append((draw.__name__, exc))
            print(f"  ! {draw.__name__} failed: {type(exc).__name__}: {exc}")
    drawn = len(FIGURES_TO_DRAW) - len(failures)
    print(f"{drawn}/{len(FIGURES_TO_DRAW)} figures written to {FIGURES}")
    if failures:
        raise SystemExit(f"{len(failures)} figure(s) failed")
