"""Step 5 - draw the figures the case study reports.

  fig1  the channel effect: contribution over the course of a game
  fig2  the decomposition: momentum, channel, and each block of talk
  fig3  which kind of talk carries any content term
  fig4  when in a game each block of talk matters
  fig5  which individual features move contribution, and whether they replicate

Run:  python scripts/05_figures.py   (after 04_analysis.py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, FIGURES, TABLES
from style import (AQUA, COLOR_CHANNEL, COLOR_LEARN, COLOR_NO_CHANNEL, COLOR_VAL,
                   GRID, INK, INK_2, INK_MUTED, ORANGE, caption, title, use_style)

use_style()

# One colour per model family, held fixed across every figure that shows both.
MODEL_COLORS = {"elastic net": COLOR_CHANNEL, "random forest": ORANGE}

# Toolkit column names are precise but long; shorten them for axis labels only.
LABEL_FIXES = {"_lexical_wordcount": " (LIWC)", "_politeness_convokit": " (politeness)",
               "_receptiveness_yeomans": " (receptiveness)", "_bert": " (BERT)",
               "sum_": "total ", "mean_": "", "stdev_": "SD of ",
               "_chats": "", "_conversation": "", "_": " "}


def pretty(name):
    for old, new in LABEL_FIXES.items():
        name = name.replace(old, new)
    return name.strip()


# ------------------------------------------------------------------ fig 1 ---
def fig_channel_effect():
    """Mean contribution by round, groups with a channel vs. groups without."""
    rounds = pd.concat([pd.read_csv(DATA_PROCESSED / f"rounds_{s}.csv").assign(split=s)
                        for s in ("learn", "val")])
    rounds["has_chat_channel"] = rounds["has_chat_channel"].astype(bool)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for has_channel, color, label in [(True, COLOR_CHANNEL, "Channel open"),
                                      (False, COLOR_NO_CHANNEL, "No channel")]:
        sub = rounds[rounds.has_chat_channel == has_channel]
        stats = sub.groupby("round_index")["contribution_rate"].agg(["mean", "sem", "count"])
        stats = stats[stats["count"] >= 20]      # keep rounds still well populated
        x = stats.index.to_numpy()
        ax.fill_between(x, stats["mean"] - 1.96 * stats["sem"],
                        stats["mean"] + 1.96 * stats["sem"],
                        color=color, alpha=0.14, linewidth=0)
        ax.plot(x, stats["mean"], color=color, label=label)
        ax.annotate(label, (x[-1], stats["mean"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", fontweight="semibold")

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean contribution rate")
    ax.set_xlim(left=0)
    ax.margins(x=0.12)
    title(ax, "Groups that can talk contribute more, and keep contributing",
          "Mean per-player contribution as a share of endowment, with 95% CIs")
    caption(fig, "All game-rounds, both splits. Rounds shown while at least 20 games "
                 "remain in play.")
    fig.savefig(FIGURES / "fig1_channel_effect.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2 ---
def fig_decomposition():
    """ΔR² from adding the channel, then from adding what was said in it."""
    decomp = pd.read_csv(TABLES / "variance_decomposition.csv")
    comparison = pd.read_csv(TABLES / "model_comparison.csv")

    steps = ["channel", "momentum", "deliberation (PRE)", "reaction (POST)",
             "both talk blocks"]
    labels = ["Having a channel\n(mere access)",
              "What they contributed\nlast round (momentum)",
              "What they said while\ndeciding (PRE)",
              "What they said about\nthe last result (POST)",
              "Both blocks of talk\ntogether"]
    fams = list(MODEL_COLORS)
    fig, ax = plt.subplots(figsize=(8, 5.6))

    height = 0.34
    for i, fam in enumerate(fams):
        sub = decomp[decomp.model_family == fam].set_index("step").loc[steps]
        y = np.arange(len(steps)) + (i - 0.5) * height
        ax.barh(y, sub["delta_cv_r2"], height=height * 0.9,
                color=MODEL_COLORS[fam], label=fam, zorder=3)
        ax.errorbar(sub["delta_cv_r2"], y,
                    xerr=[sub["delta_cv_r2"] - sub["ci_low"],
                          sub["ci_high"] - sub["delta_cv_r2"]],
                    fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
        ax.scatter(sub["delta_r2_heldout"], y, s=34, color=INK, zorder=5,
                   edgecolor="white", linewidth=1.2,
                   label="Held-out split" if i == 0 else None)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    ax.set_yticks(np.arange(len(steps)), labels, fontsize=9)
    ax.set_xlabel("ΔR²  (added out-of-fold variance explained in the next round's contribution)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    full = comparison[comparison.model == "M5 + both blocks"]
    totals = ", ".join(f"{r.model_family} R²={r.cv_r2_learn:.2f}"
                       for r in full.itertuples())
    title(ax, "Momentum dominates, the channel matters, the content barely registers",
          f"Each block measured against the model it should be judged against. "
          f"Full model {totals}.")
    caption(fig, "Bars: cross-validated ΔR² on the learning split, games held out whole; "
                 "error bars are game-clustered bootstrap 95% CIs. Dots: ΔR² on the "
                 "held-out split. Channel then momentum are sequential; the two talk "
                 "blocks are each measured against the momentum model, so they do not "
                 "compete for being entered first. The channel is measured before "
                 "momentum because it was randomized and momentum is one of the things "
                 "it changes.")
    fig.savefig(FIGURES / "fig2_decomposition.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3 ---
def fig_family_importance():
    """How much R² each toolkit feature family carries that the others do not."""
    fam = pd.read_csv(TABLES / "family_importance.csv")
    order = (fam[fam.model_family == "elastic net"]
             .sort_values("drop_in_cv_r2")["feature_family"].tolist())

    fig, ax = plt.subplots(figsize=(7.6, 0.46 * len(order) + 2.4))
    height = 0.36
    for i, model in enumerate(MODEL_COLORS):
        sub = fam[fam.model_family == model].set_index("feature_family").loc[order]
        y = np.arange(len(order)) + (i - 0.5) * height
        ax.barh(y, sub["drop_in_cv_r2"], height=height * 0.9,
                color=MODEL_COLORS[model], label=model, zorder=3)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    counts = (fam[fam.model_family == "elastic net"]
              .set_index("feature_family")["n_features"])
    ax.set_yticks(np.arange(len(order)),
                  [f"{f}  ({counts[f]})" for f in order], fontsize=9)
    ax.set_xlabel("Drop in R² when this family is removed from the full model")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    title(ax, "Which kind of talk is doing the work",
          "Leave-one-family-out: what each family adds that no other family already covers")
    caption(fig, "Feature counts in parentheses. Families overlap, so the drops do not "
                 "sum to the content term. A negative drop means the family was costing "
                 "the model accuracy.")
    fig.savefig(FIGURES / "fig3_family_importance.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4 ---
def fig_round_stage():
    """Each talk block's contribution, recomputed within thirds of a game."""
    stage = pd.read_csv(TABLES / "round_stage.csv")
    order = ["early", "middle", "late"]
    block_titles = {"pre": "Deliberation (PRE)\nsaid while deciding",
                    "post": "Reaction (POST)\nsaid about the last result"}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    x = np.arange(len(order))
    for ax, block in zip(axes, block_titles):
        for model, color in MODEL_COLORS.items():
            sub = (stage[(stage.model_family == model) & (stage.block == block)]
                   .set_index("stage").loc[order])
            ax.plot(x, sub["delta_cv_r2_content"], color=color, marker="o",
                    markersize=7, markeredgecolor="white", markeredgewidth=1.4,
                    label=model, zorder=4)
            ax.fill_between(x, sub["ci_low"], sub["ci_high"], color=color,
                            alpha=0.13, linewidth=0, zorder=2)
        ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=3)
        n = (stage[(stage.model_family == "elastic net") & (stage.block == block)]
             .set_index("stage").loc[order, "n_game_rounds"])
        ax.set_xticks(x, [f"{s.capitalize()}\n({n[s]:,})" for s in order], fontsize=9)
        ax.set_title(block_titles[block], fontsize=10, color=INK_2,
                     fontweight="normal")
        ax.set_xlabel("Where the round sits in the game")

    axes[0].set_ylabel("ΔR² from that block of talk")
    axes[1].legend(loc="best")
    fig.suptitle("When in a game does what you say matter?", x=0.0, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    caption(fig, "Bands are game-clustered 95% CIs. Each third is fit and "
                 "cross-validated separately, so the models see far fewer game-rounds "
                 "than the full sample (counts in parentheses) and intervals widen.")
    fig.savefig(FIGURES / "fig4_round_stage.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5 ---
def fig_feature_effects(top_n=16):
    """Standardized effect of each conversation feature, learn vs. held-out."""
    eff = pd.read_csv(TABLES / "feature_effects.csv").head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 0.38 * len(eff) + 2.4))
    y = np.arange(len(eff))

    ax.hlines(y, eff["ci_low_learn"], eff["ci_high_learn"],
              color=COLOR_LEARN, alpha=0.35, linewidth=2.5, zorder=3)
    ax.scatter(eff["coef_learn"], y, s=44, color=COLOR_LEARN, zorder=4,
               edgecolor="white", linewidth=1.2, label="Learning split")
    ax.scatter(eff["coef_val"], y, s=30, color=COLOR_VAL, zorder=5, marker="D",
               edgecolor="white", linewidth=1.2, label="Held-out split")

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    labels = [f"{pretty(f)} [{b}]" + (" *" if r else "")
              for f, b, r in zip(eff["feature"], eff["block"], eff["replicates"])]
    ax.set_yticks(y, labels, fontsize=9)
    for tick, fam in zip(ax.get_yticklabels(), eff["family"]):
        tick.set_color(INK if fam != "Other" else INK_MUTED)
    ax.set_xlabel("Change in contribution rate per SD of the feature\n"
                  "(controlling for game design and round timing)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    n_rep = int(pd.read_csv(TABLES / "feature_effects.csv")["replicates"].sum())
    title(ax, "Which conversation features predict contribution",
          f"Top {len(eff)} by learning-split p-value. [pre] = said while deciding, "
          f"[post] = said about the last result")
    caption(fig, "Bars are 95% CIs on the learning split, clustered by game. Controls "
                 "include game design, round timing, and last round's contribution. "
                 f"* marks the {n_rep} feature-block combination(s) that clear FDR "
                 "q<.05 on the learning split and hold up on held-out data.")
    fig.savefig(FIGURES / "fig5_feature_effects.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_channel_effect()
    fig_decomposition()
    fig_family_importance()
    fig_round_stage()
    fig_feature_effects()
    print(f"figures written to {FIGURES}")
