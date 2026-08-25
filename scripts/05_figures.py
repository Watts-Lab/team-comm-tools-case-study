"""Step 5 - draw the figures the case study reports.

  fig1  the channel effect: contribution over the course of a game
  fig2  the decomposition: the channel, then whether talk happened, then its content
  fig3  which family of toolkit features carries any content term
  fig4  when in a game talk matters, under both definitions of a game stage
  fig5  a closer look at the one stage where content predicts
  fig6  which individual features move contribution, and whether they replicate
  fig7  what is actually said at each stage of a game
  fig8  whether the same features predict contribution at every stage

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


# ------------------------------------------------------------------ fig 1 ---
def fig_channel_effect():
    rounds = pd.concat([pd.read_csv(DATA_PROCESSED / f"rounds_{s}.csv") for s in
                        ("learn", "val")])
    rounds["has_chat_channel"] = rounds["has_chat_channel"].astype(bool)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for has_channel, color, label in [(True, COLOR_CHANNEL, "Channel open"),
                                      (False, COLOR_NO_CHANNEL, "No channel")]:
        sub = rounds[rounds.has_chat_channel == has_channel]
        stats = sub.groupby("round_index")["contribution_rate"].agg(
            ["mean", "sem", "count"])
        stats = stats[stats["count"] >= 20]
        x = stats.index.to_numpy()
        ax.fill_between(x, stats["mean"] - 1.96 * stats["sem"],
                        stats["mean"] + 1.96 * stats["sem"],
                        color=color, alpha=0.14, linewidth=0)
        ax.plot(x, stats["mean"], color=color, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean contribution rate")
    ax.set_xlim(left=0)
    header(fig, ax, "Groups that can talk contribute more, and keep contributing",
           "Share of endowment contributed, averaged across groups. Bands are 95% CIs.",
           legend_from=ax, ncol=2)
    fig.savefig(FIGURES / "fig1_channel_effect.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2 ---
def fig_decomposition():
    """Channel, then whether each kind of talk happened, then what it contained.

    These three terms form one chain: the channel is added to a model of the game's
    rules, the speech indicators to that, and the content features to that. Showing
    them together is what makes the comparison legible - splitting the talk terms
    into a second figure hides that they are subdivisions of the same quantity.
    """
    decomp = pd.read_csv(TABLES / "variance_decomposition.csv")
    sv = pd.read_csv(TABLES / "speech_vs_content.csv")

    def bold_first(head, tail):
        return rf"$\bf{{{head.replace(' ', chr(92) + ' ')}}}$" + f"\n{tail}"

    rows = [(bold_first("Having a channel", "channel open vs. closed"),
             decomp[decomp.step == "channel"])]
    for block in ("pre", "post"):
        for component, name in [("spoke at all", "Spoke at all"),
                                ("what was said", "What was said")]:
            rows.append((bold_first(name, BLOCK_SHORT[block]),
                         sv[(sv.block == block) & (sv.component == component)]))
    rows = rows[::-1]                       # first term at the top of the chart
    labels = [label for label, _ in rows]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    height = 0.34
    for i, model in enumerate(MODEL_COLORS):
        deltas, los, his, held = [], [], [], []
        for _, table in rows:
            r = table[table.model_family == model].iloc[0]
            deltas.append(r["delta_cv_r2"])
            los.append(r["ci_low"])
            his.append(r["ci_high"])
            held.append(r["delta_r2_heldout"])
        y = np.arange(len(rows)) + (i - 0.5) * height
        deltas = np.array(deltas)
        ax.barh(y, deltas, height=height * 0.9, color=MODEL_COLORS[model],
                label=model, zorder=3)
        ax.errorbar(deltas, y,
                    xerr=[np.clip(deltas - np.array(los), 0, None),
                          np.clip(np.array(his) - deltas, 0, None)],
                    fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
        ax.scatter(held, y, s=34, color=INK, zorder=5, edgecolor="white",
                   linewidth=1.2, label="Held-out split" if i == 0 else None)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    ax.axhline(len(rows) - 1.5, color=GRID, linewidth=1.4, zorder=2)
    ax.axhline(len(rows) - 3.5, color=GRID, linewidth=1.4, zorder=2)
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=9.5)
    ax.set_xlabel("Added variance explained (ΔR²)")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.10)
    header(fig, ax, "Whether a group spoke matters more than what it said",
           "The channel is added to a model of the game's rules. Within each kind of "
           "talk, whether the group spoke is added to that, and what it said on top "
           "of that again. 10-fold cross-validation holding out whole games; "
           "intervals are 95% game-clustered bootstraps.",
           legend_from=ax, ncol=3)
    fig.savefig(FIGURES / "fig2_decomposition.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3 ---
def fig_family_importance():
    fam = pd.read_csv(TABLES / "family_importance.csv")
    order = (fam[fam.model_family == "elastic net"]
             .sort_values("drop_in_cv_r2")["feature_family"].tolist())

    fig, ax = plt.subplots(figsize=(8, 0.46 * len(order) + 2.6))
    height = 0.36
    for i, model in enumerate(MODEL_COLORS):
        sub = fam[fam.model_family == model].set_index("feature_family").reindex(order)
        y = np.arange(len(order)) + (i - 0.5) * height
        ax.barh(y, sub["drop_in_cv_r2"], height=height * 0.9,
                color=MODEL_COLORS[model], label=model, zorder=3)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    counts = (fam[fam.model_family == "elastic net"]
              .set_index("feature_family")["n_features"])
    ax.set_yticks(np.arange(len(order)),
                  [bold_label(f) + f"  ({counts[f]})" for f in order], fontsize=9.5)
    ax.set_xlabel("Change in R² when the family is removed")
    ax.grid(axis="y", visible=False)
    axis_direction(ax, "family was hurting accuracy", "family was helping accuracy")
    header(fig, ax, "No family of features carries the model on its own",
           "Feature count per family in parentheses. Families overlap, so the "
           "changes do not sum. 10-fold cross-validation holding out whole games.",
           legend_from=ax, ncol=2)
    fig.savefig(FIGURES / "fig3_family_importance.png")
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
    fig.savefig(FIGURES / "fig5_opening_reaction.png")
    plt.close(fig)

# ------------------------------------------------------------------ fig 6 ---
def fig_feature_effects(per_block=8):
    eff = pd.read_csv(TABLES / "feature_effects.csv")
    # A handful of features are constant in the held-out split, so no held-out
    # estimate exists for them. This figure exists to compare the two splits, and a
    # row with a missing marker cannot be read, so they are left out.
    eff = eff[eff["coef_val"].notna()]
    # Grouped by block so the two kinds of talk can be told apart at a glance.
    picks = [eff[eff.block == b].nsmallest(per_block, "p_learn") for b in ("pre", "post")]
    eff = pd.concat(picks)
    order = eff.iloc[::-1].reset_index(drop=True)      # bottom-up
    split_at = (order["block"] == "post").sum() - 0.5  # divider between the blocks

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 3.0))
    y = np.arange(len(order))
    for block, color in BLOCK_COLORS.items():
        mask = (order["block"] == block).to_numpy()
        if not mask.any():
            continue
        ax.hlines(y[mask], order.loc[mask, "ci_low_learn"],
                  order.loc[mask, "ci_high_learn"], color=color, alpha=0.30,
                  linewidth=2.5, zorder=3)
        ax.scatter(order.loc[mask, "coef_learn"], y[mask], s=46, color=color,
                   zorder=4, edgecolor="white", linewidth=1.2,
                   label=BLOCK_LABEL[block])
        # No legend entry: a coloured diamond would read as a third category, and
        # the subtitle already says what circles and diamonds are.
        ax.scatter(order.loc[mask, "coef_val"], y[mask], s=30, color=color,
                   zorder=5, marker="D", alpha=0.45, edgecolor="white",
                   linewidth=1.0)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    ax.axhline(split_at, color=GRID, linewidth=1.4, zorder=2)
    ax.set_yticks(y, [bold_label(pretty(f)) for f in order["feature"]],
                  fontsize=8.5)
    ax.set_xlabel("Change in contribution rate per SD of the feature, with 95% CI")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.03)
    add_split_legend(ax)
    header(fig, ax, "Which conversation features predict contribution",
           f"Strongest {per_block} features of each kind of talk.",
           legend_from=ax, ncol=4)
    fig.savefig(FIGURES / "fig6_feature_effects.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 7 ---
def fig_stage_profile(top_n=12):
    prof = pd.read_csv(TABLES / "stage_profile.csv")
    stages = [c for c in STAGE_ORDER if c in prof.columns]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.36 * top_n + 3.0), sharex=True,
                             gridspec_kw={"wspace": 0.62})
    for ax, block in zip(axes, BLOCK_LABEL):
        sub = prof[prof["block"] == block].head(top_n).iloc[::-1]
        y = np.arange(len(sub))
        ax.hlines(y, sub[stages].min(axis=1), sub[stages].max(axis=1),
                  color=GRID, linewidth=2.5, zorder=2)
        for stage in stages:
            ax.scatter(sub[stage], y, s=48, color=STAGE_COLORS[stage],
                       edgecolor="white", linewidth=1.2, zorder=4,
                       label=stage if block == "pre" else None)
        ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=3)
        ax.set_yticks(y, [bold_label(pretty(f)) for f in sub["feature"]],
                      fontsize=8.5)
        ax.grid(axis="y", visible=False)
        panel_title(ax, BLOCK_LABEL[block])
        ax.set_xlabel("Stage mean (SDs from the average conversation)")

    header(fig, axes, "What gets said changes over the course of a game",
           "Colour marks where the round being predicted sits in its game. Features "
           "shown shift most between opening and endgame.",
           legend_from=axes[0], ncol=3, panel_titles=True)
    fig.savefig(FIGURES / "fig7_stage_profile.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 8 ---
def fig_stage_feature_effects(per_block=7):
    """Whether a feature's effect on contribution holds across stages.

    Grouped by which kind of talk the feature describes, because colour is already
    carrying the stage and a reader cannot track two categorical encodings at once.
    """
    eff = pd.read_csv(TABLES / "stage_feature_effects.csv")
    stages = [s for s in STAGE_ORDER if s in set(eff["stage"])]

    # Ranked within each block by the largest effect the feature reaches in any
    # stage, so the features with the most to say sit at the top of their group.
    strength = (eff.assign(size=eff["coef_learn"].abs())
                .groupby(["block", "feature"])["size"].max().reset_index())
    picks = pd.concat([strength[strength.block == b].nlargest(per_block, "size")
                       for b in ("pre", "post")])
    picks = picks.iloc[::-1].reset_index(drop=True)      # bottom-up for barh
    split_at = (picks["block"] == "post").sum() - 0.5

    fig, ax = plt.subplots(figsize=(9.4, 0.42 * len(picks) + 3.2))
    y = np.arange(len(picks))
    labels = []
    for yi, row in zip(y, picks.itertuples()):
        by_stage = eff[(eff.block == row.block)
                       & (eff.feature == row.feature)].set_index("stage")
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
                       label=stage if yi == y[-1] else None)
        labels.append(bold_label(pretty(row.feature)))

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=3)
    ax.axhline(split_at, color=GRID, linewidth=1.4, zorder=2)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in contribution rate per SD of the feature")
    ax.margins(y=0.04)

    # Which block each group belongs to, outside the data area.
    for block, rows in picks.groupby("block").groups.items():
        ax.annotate(BLOCK_LABEL[block], xy=(1.008, np.mean(list(rows))),
                    xycoords=("axes fraction", "data"), rotation=270,
                    ha="left", va="center", fontsize=9.5, fontweight="semibold",
                    color=BLOCK_COLORS[block], annotation_clip=False)

    header(fig, ax, "The same feature predicts differently at different stages",
           f"Strongest {per_block} features of each kind, ranked by the largest "
           "effect they reach in any stage. Solid markers are significant within "
           "that stage; faded are not.", legend_from=ax, ncol=3)
    fig.savefig(FIGURES / "fig8_stage_feature_effects.png")
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
                 "beginning is rounds 0-2, end is the last 3 rounds"),
    "relative": (["early", "middle", "late"], "By thirds",
                 "beginning is the first third of the rounds played"),
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
    fig.savefig(FIGURES / "fig4_staging_comparison.png")
    plt.close(fig)


FIGURES_TO_DRAW = [fig_channel_effect, fig_decomposition, fig_family_importance,
                   fig_staging_comparison, fig_opening_reaction, fig_feature_effects,
                   fig_stage_profile, fig_stage_feature_effects]

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
