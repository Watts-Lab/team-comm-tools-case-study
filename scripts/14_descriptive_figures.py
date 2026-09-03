"""Step 14 - descriptive figures for the case study's corpus section.

Two figures, both about the data rather than about any model:

  fig0_communication_effect.png  what having a channel, and using it, is worth
  fig0_corpus.png                how long games run, when talk happens, and how
                                 often a group with a channel says nothing

These belong before the modeling results: the first establishes that the effect
the case study is trying to decompose is present in this corpus at all, and the
second shows how much conversation there is to work with and where it sits.

The three-way split in the first figure is the one that matters. A group with no
channel could not talk; a group with a channel that never sent a message chose not
to. Collapsing those two into "no conversation" would attribute a behaviour to a
treatment.

Run:  python scripts/14_descriptive_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import style
from config import DATA_PROCESSED, FIGURES, SEED, SPLITS, TABLES

FIG_DIR = FIGURES / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_LABEL = {"learn": "Learning games", "val": "Validation games"}
STAGE_ORDER = ["opening", "middle", "endgame"]
STAGE_LABEL = {"opening": "opening\n(first 3 rounds)", "middle": "middle",
               "endgame": "endgame\n(last 3 rounds)"}

# The four marks in the first figure. The first is the control; the second pools
# every group that was given a channel, whether or not it used one, which is the
# comparison the experiment actually randomized; the last two split that pool by
# what the group did with the channel.
GROUPS = [("no channel", "no chat\nchannel", style.INK_MUTED),
          ("channel", "channel present\n(pooled)", style.VIOLET),
          ("silent", "channel,\nnever used", style.ORANGE),
          ("spoke", "channel,\nused", style.BLUE)]

# Each of the three channel states is tested against the same control group.
CONTRASTS = [("channel", 0, 1), ("silent", 0, 2), ("spoke", 0, 3)]
# Tested and reported, but not drawn: a bracket between two marks that are both
# already bracketed to the control would cross the other two.
EXTRA_CONTRASTS = [("silent", "spoke")]

# The game-length histogram overlaps two series, so it needs transparency; the
# other two panels do not, but they use the same value anyway or a bar in one
# panel would read as a different colour from the same bar in another.
BAR_ALPHA = 0.62

# Contribution is a share of the endowment; it is reported throughout the case
# study in percent of the endowment, and differences between groups in percentage
# points, so the game means are scaled by 100 as soon as they are read.
PP = 100

rng = np.random.default_rng(SEED)


def load(split):
    rounds = pd.read_csv(DATA_PROCESSED / f"rounds_{split}.csv", low_memory=False)
    chat = pd.read_csv(DATA_PROCESSED / f"chat_{split}.csv", low_memory=False)
    rounds["gameId"] = rounds["gameId"].astype(str)
    chat["gameId"] = chat["gameId"].astype(str)
    return rounds, chat


def game_table(rounds, chat):
    """One row per game: how much it contributed, and whether it talked."""
    spoke = set(chat["gameId"].unique())
    g = rounds.groupby("gameId").agg(
        contribution=("contribution_rate", "mean"),
        rounds=("round_index", "nunique"),
        channel=("has_chat_channel", "first")).reset_index()
    g["channel"] = g["channel"].astype(bool)
    g["state"] = np.where(~g["channel"], "no channel",
                          np.where(g["gameId"].isin(spoke), "spoke", "silent"))
    return g


def values_for(g, state):
    """Contributions of the games in one of the four marks.

    "channel" is not a state a game is in but the union of the two states that
    had a channel, so it is selected on the channel flag rather than on `state`.
    """
    mask = g["channel"] if state == "channel" else g["state"] == state
    return PP * g.loc[mask, "contribution"].to_numpy(dtype=float)


def contrast(a, b):
    """Welch's t-test, Cohen's d and a rank test for one group against control.

    Welch rather than Student because the groups are of very unequal size, and the
    rank test alongside it because the smallest group here is a few dozen games.
    """
    t, p = stats.ttest_ind(b, a, equal_var=False)
    df = stats.ttest_ind(b, a, equal_var=False).df
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                         + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    _, p_mw = stats.mannwhitneyu(b, a, alternative="two-sided")
    return dict(n_control=len(a), n_group=len(b),
                mean_control=a.mean(), mean_group=b.mean(),
                diff=b.mean() - a.mean(), t=t, df=df, p=p,
                cohens_d=(b.mean() - a.mean()) / pooled_sd, p_mannwhitney=p_mw)


def p_text(p):
    return ("$p$ < 0.001" if p < 0.001
            else f"$p$ = {p:.3f}".replace("0.", ".", 1))


def bracket(ax, i, j, y, label):
    """Comparison bar spanning two marks, labelled with the test result."""
    drop = 0.7
    ax.plot([i, i, j, j], [y - drop, y, y, y - drop], color=style.INK_MUTED,
            linewidth=1.0, zorder=2, clip_on=False)
    ax.annotate(label, xy=((i + j) / 2, y), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.2, color=style.INK_2, zorder=5, annotation_clip=False)


def boot_ci(values, n_boot=2000):
    """Percentile interval for a mean, resampling games."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    draws = [values[rng.integers(0, len(values), len(values))].mean()
             for _ in range(n_boot)]
    return tuple(np.percentile(draws, [2.5, 97.5]))


# ------------------------------------------------- 1. communication effect --
def fig_communication_effect():
    """Mean contribution by whether a group could talk, and whether it did."""
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(12.2, 5.8), sharey=True)
    axes = np.atleast_1d(axes)

    # The two panels share a y-axis, so where the comparison brackets sit has to
    # be decided from both panels at once: a level chosen inside one panel lands
    # on the data, or on the panel title, in the other.
    panels = {}
    for split in SPLITS:
        g = game_table(*load(split))
        marks = [(state, values_for(g, state)) for state, _, _ in GROUPS]
        panels[split] = (g, marks, [boot_ci(v) for _, v in marks])

    ceiling = max(hi for _, _, cis in panels.values() for _, hi in cis)
    step, headroom = 3.0, 3.4
    top = ceiling + step * len(CONTRASTS) + headroom

    rows = []
    for ax, split in zip(axes, SPLITS):
        g, marks, cis = panels[split]
        for i, ((state, values), (lo, hi)) in enumerate(zip(marks, cis)):
            colour = GROUPS[i][2]
            ax.plot([i, i], [lo, hi], color=colour, linewidth=2.2,
                    solid_capstyle="butt", zorder=3)
            ax.plot([i], [values.mean()], "o", markersize=9, color=colour, zorder=4)
        # The group size goes in the tick label rather than beside the mark: an
        # annotation pinned to the end of an interval lands on the axis whenever
        # that interval happens to be a long one.
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f"{label}\nn={len(values)}"
                            for (_, label, _), (_, values) in zip(GROUPS, marks)],
                           fontsize=8.5)
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        ax.set_ylim(top=top)
        style.panel_title(ax, f"{SPLIT_LABEL[split]} (n = {len(g)} games)")

        control = values_for(g, "no channel")
        for level, (state, i, j) in enumerate(CONTRASTS):
            res = contrast(control, values_for(g, state))
            rows.append(dict(split=split, comparison=f"{state} vs no channel", **res))
            bracket(ax, i, j, ceiling + step * (level + 1),
                    f"$\\Delta$ = {res['diff']:+.1f} pp, {p_text(res['p'])}")
        for base, state in EXTRA_CONTRASTS:
            rows.append(dict(split=split, comparison=f"{state} vs {base}",
                             **contrast(values_for(g, base), values_for(g, state))))

    axes[0].set_ylabel("game's mean contribution\n(percent of endowment)")

    style.header(
        fig, axes,
        "The Communication Effect in Alsobay et al. (2026)",
        subtitle="Brackets give the difference in means, in percentage points of "
                 "the endowment, against the games with no chat channel. The "
                 "$p$-value comes from a Welch's $t$-test.",
        panel_titles=True, headline_weight="bold", headline_size=14)
    out = FIG_DIR / "fig0_communication_effect.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    tests = pd.DataFrame(rows)
    tests_out = TABLES / "channel_effect_tests.csv"
    tests.to_csv(tests_out, index=False)
    print(f"wrote {tests_out}")
    print(tests.to_string(index=False))


# -------------------------------------------------------- 2. the corpus ----
def fig_corpus():
    """How long games run, when messages are sent, and how often nobody speaks."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))

    medians = {}
    for split, colour in zip(SPLITS, (style.BLUE, style.AQUA)):
        rounds, chat = load(split)
        g = game_table(rounds, chat)

        # (a) game length
        axes[0].hist(g["rounds"], bins=np.arange(2.5, 31.5, 1), color=colour,
                     alpha=BAR_ALPHA, label=SPLIT_LABEL[split])
        medians[split] = float(np.median(g["rounds"]))

        # (b) where the messages are. Messages are attributed to the stage of the
        # round they predict, not the round they were spoken in, so that this
        # matches the partition the models use.
        stage_of = rounds.set_index(["gameId", "round_index"])["stage_absolute"]
        msgs = chat[chat["block"].isin(["pre", "post"])]
        msgs = msgs.join(stage_of.rename("stage"), on=["gameId", "target_round"])
        counts = msgs["stage"].value_counts().reindex(STAGE_ORDER).fillna(0)
        x = np.arange(len(STAGE_ORDER)) + (0.2 if split == "val" else -0.2)
        axes[1].bar(x, counts.to_numpy(), width=0.36, color=colour, alpha=BAR_ALPHA,
                    label=SPLIT_LABEL[split])

        # (c) among rounds where a channel existed, how often nobody spoke
        spoke_ids = set(chat.loc[chat["block"] == "window", "conv_id"])
        rounds["spoke"] = rounds["conv_id_window"].astype(str).isin(spoke_ids)
        chan = rounds[rounds["has_chat_channel"].astype(bool)]
        silent = (chan.groupby("stage_absolute")["spoke"]
                  .apply(lambda s: 1 - s.mean()).reindex(STAGE_ORDER))
        axes[2].bar(x, silent.to_numpy() * 100, width=0.36, color=colour,
                    alpha=BAR_ALPHA, label=SPLIT_LABEL[split])

    # The two medians are a round apart, so their labels are pushed to opposite
    # sides of their own line rather than centred, where they would overlap.
    for split, colour, ha, dx in (("learn", style.BLUE, "left", 4),
                                  ("val", style.AQUA, "right", -4)):
        m = medians[split]
        axes[0].axvline(m, color=colour, linestyle=(0, (4, 2)), linewidth=1.6,
                        zorder=4)
        axes[0].annotate(f"median {m:.0f}", xy=(m, 1), xytext=(dx, -6),
                         xycoords=("data", "axes fraction"),
                         textcoords="offset points", ha=ha, va="top",
                         fontsize=8.5, color=colour, zorder=5)

    axes[0].set_xlabel("rounds in the game")
    axes[0].set_ylabel("games")
    style.panel_title(axes[0], "Game length")

    for ax in axes[1:]:
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels([STAGE_LABEL[s] for s in STAGE_ORDER], fontsize=8.5)
        ax.set_xlabel("position of the round in its game")
    axes[1].set_ylabel("messages")
    style.panel_title(axes[1], "Temporal distribution of messages")
    axes[2].set_ylabel("percent of rounds with no message")
    style.panel_title(axes[2], "Rates of silence (0 messages sent in round)")

    # No headline: the three panel titles already say what each panel is, and a
    # single sentence spanning three unrelated distributions would only summarize
    # one of them.
    style.header(fig, axes, "", legend_from=axes[0], ncol=2, panel_titles=True)
    out = FIG_DIR / "fig0_corpus.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    style.use_style()
    fig_communication_effect()
    fig_corpus()
    print(f"descriptive figures written to {FIG_DIR}")
