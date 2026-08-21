"""Step 5 - draw the four figures the case study reports.

  fig1  the channel effect: contribution over the course of a game
  fig2  what conversation adds to prediction, in and out of sample
  fig3  which conversation features move contribution, and whether they replicate
  fig4  the top replicating features, one scatter each

Run:  python scripts/05_figures.py   (after 04_analysis.py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, FIGURES, TABLES
from style import (COLOR_CHANNEL, COLOR_LEARN, COLOR_NO_CHANNEL, COLOR_VAL,
                   GRID, INK, INK_2, INK_MUTED, caption, title, use_style)

use_style()

# Toolkit column names are precise but long; shorten them for axis labels only.
LABEL_FIXES = {"_lexical_wordcount": " (LIWC)", "_politeness_convokit": " (politeness)",
               "_receptiveness_yeomans": " (receptiveness)", "_bert": " (BERT)",
               "sum_": "total ", "average_": "mean ", "stdev_": "SD of ",
               "_chats": "", "_": " "}


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
        # Only plot rounds still supported by a reasonable number of games.
        stats = stats[stats["count"] >= 20]
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
    caption(fig, "All games, both splits. Rounds shown while at least 20 games remain in play.")
    fig.savefig(FIGURES / "fig1_channel_effect.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2 ---
def fig_model_comparison():
    """Cross-validated and held-out R² for each nested model."""
    comp = pd.read_csv(TABLES / "model_comparison.csv")
    comp = comp.iloc[::-1]  # first spec at the top of the chart

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(comp))
    colors = [COLOR_NO_CHANNEL if "no channel" in m else COLOR_CHANNEL
              for m in comp["model"]]

    ax.barh(y, comp["cv_r2_learn"], height=0.5, color=colors, zorder=3)
    ax.errorbar(comp["cv_r2_learn"], y,
                xerr=[comp["cv_r2_learn"] - comp["cv_r2_ci_low"],
                      comp["cv_r2_ci_high"] - comp["cv_r2_learn"]],
                fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
    ax.scatter(comp["r2_heldout_val"], y, s=42, color=INK, zorder=5,
               edgecolor="white", linewidth=1.5, label="Held-out split")

    for yi, row in zip(y, comp.itertuples()):
        ax.annotate(f"{row.cv_r2_learn:.2f}", (max(row.cv_r2_learn, 0), yi),
                    xytext=(4, 10), textcoords="offset points",
                    fontsize=9, color=INK_2)

    ax.axvline(0, color=GRID, linewidth=1, zorder=2)
    ax.set_yticks(y, [m.replace(": ", ":\n") for m in comp["model"]], fontsize=9)
    ax.set_xlabel("R²  (variance in penultimate-round contribution explained)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")

    delta = pd.read_csv(TABLES / "delta_r2.csv").iloc[0]
    title(ax, "How much does the conversation add to what the rules already tell us?",
          "Bars: repeated 10-fold cross-validated R² on the learning split, with bootstrap "
          "95% CIs. Dots: R² on the held-out split.")
    ax.annotate(
        "adding conversation to game rules:\n"
        f"ΔR² = {delta.delta_cv_r2_learn:+.2f} "
        f"[{delta.delta_cv_ci_low:+.2f}, {delta.delta_cv_ci_high:+.2f}] cross-validated\n"
        f"ΔR² = {delta.delta_r2_heldout_val:+.2f} "
        f"[{delta.delta_val_ci_low:+.2f}, {delta.delta_val_ci_high:+.2f}] held out",
        xy=(0.99, 0.06), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8.5, color=INK_2)
    caption(fig, "Orange: groups with no channel. Blue: groups that talked. "
                 "A negative R² means the model predicts worse than the mean.")
    fig.savefig(FIGURES / "fig2_model_comparison.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3 ---
def fig_feature_effects(top_n=14):
    """Standardized effect of each conversation feature, learn vs. held-out."""
    eff = pd.read_csv(TABLES / "feature_effects.csv").head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.8, 0.38 * len(eff) + 2.2))
    y = np.arange(len(eff))

    ax.hlines(y, eff["ci_low_learn"], eff["ci_high_learn"],
              color=COLOR_LEARN, alpha=0.35, linewidth=2.5, zorder=3)
    ax.scatter(eff["coef_learn"], y, s=44, color=COLOR_LEARN, zorder=4,
               edgecolor="white", linewidth=1.2, label="Learning split")
    ax.scatter(eff["coef_val"], y + 0.0, s=30, color=COLOR_VAL, zorder=5,
               marker="D", edgecolor="white", linewidth=1.2, label="Held-out split")

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=2)
    labels = [f"{pretty(f)} *" if r else pretty(f)
              for f, r in zip(eff["feature"], eff["replicates"])]
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Change in contribution rate per SD of the feature\n"
                  "(controlling for the game's design parameters)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    title(ax, "Which conversation features predict contribution",
          f"Top {len(eff)} features by learning-split p-value, among groups that talked")
    n_rep = int(pd.read_csv(TABLES / "feature_effects.csv")["replicates"].sum())
    caption(fig, "Bars are 95% CIs on the learning split. * marks the "
                 f"{n_rep} feature(s) reaching p<.05 with the same sign in both splits. "
                 "No feature clears an FDR correction within the learning split alone, "
                 "which is what ~150 conversations and 136 candidate features buys you.")
    fig.savefig(FIGURES / "fig3_feature_effects.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4 ---
def fig_top_feature_scatters(n_panels=3):
    """The strongest features, drawn as raw game-level scatters."""
    eff = pd.read_csv(TABLES / "feature_effects.csv")
    # Rank by learning-split evidence, which is the only ranking available before
    # the held-out split is opened; the panels then show which ones held up.
    chosen = eff.head(n_panels)

    learn = pd.read_csv(DATA_PROCESSED / "analysis_learn.csv")
    learn = learn[learn["did_communicate"].astype(bool)]

    fig, axes = plt.subplots(1, len(chosen), figsize=(3.4 * len(chosen), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, chosen.itertuples()):
        x = pd.to_numeric(learn[row.feature], errors="coerce")
        y = learn["contribution_penultimate"]
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=22, color=COLOR_CHANNEL, alpha=0.55,
                   edgecolor="white", linewidth=0.6, zorder=3)
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.6, zorder=4)
        ax.set_xlabel(pretty(row.feature), fontsize=9)
        verdict = "replicates out of sample" if row.replicates else "does not replicate"
        ax.set_title(f"r = {np.corrcoef(x[ok], y[ok])[0, 1]:.2f}   ·   {verdict}",
                     fontsize=8.5, color=INK_2, fontweight="normal")

    axes[0].set_ylabel("Contribution rate\n(penultimate round)")
    fig.suptitle("The strongest learning-split signals, and which one survived",
                 x=0.0, ha="left", fontsize=12, fontweight="semibold", color=INK)
    caption(fig, "Learning split, groups that talked. Lines are unadjusted OLS fits; "
                 "the coefficients in fig3 additionally control for game design.")
    fig.savefig(FIGURES / "fig4_top_features.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_channel_effect()
    fig_model_comparison()
    fig_feature_effects()
    fig_top_feature_scatters()
    print(f"figures written to {FIGURES}")
