"""Step 13 - the window comparison as figures.

Four ways of bounding the talk that precedes one contribution decision, read side
by side. Every figure here answers the same question - *does this result depend on
how you draw the window?* - so each one puts the four blocks on a shared axis
rather than giving each its own chart.

Figures (into outputs/figures/windows/):

  fig_w1_blocks_across_game.png   what conversation content adds, by block and by
                                  position in the game, with the marginal fits and
                                  the common-subsample fits stacked so the two
                                  operationalizations can be compared directly.
  fig_w2_replicating_features.png how many features survive FDR on the learning
                                  split *and* replicate held out, per block per
                                  bin. This is the most robust evidence in the
                                  study and it is a count, so it gets its own chart.
  fig_w3_paired_contrasts.png     block-versus-block differences estimated on one
                                  common subsample, where a difference has an
                                  interval that means something.

Run:  python scripts/13_window_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import style
from style import pretty
from config import BLOCK_MEANING, DATA_PROCESSED, FIGURES, TABLES_WINDOWS

# Main figures carry the narrative; appendix figures are the same analyses run on
# the two wider windows, plus the robustness work. They are kept apart so the
# reader is never asked to work out which is which.
# The outcome is a share of the endowment; every effect on it is reported in
# percentage points of the endowment, so plotted coefficients are scaled by 100.
PP = 100

MAIN_DIR = FIGURES / "main"
APPENDIX_DIR = FIGURES / "appendix"
for _d in (MAIN_DIR, APPENDIX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Pre and Post are the two halves of one gap and are the narrative; Window is their
# union and Cumulative their superset, so both are supplementary.
MAIN_BLOCKS = ("pre", "post")
APPENDIX_BLOCKS = ("window", "cumulative")

# Row set and row order for the feature figures come from one named window, so a
# main figure and its appendix counterpart show the same features in the same
# order and can be read side by side. Post is the window the published study uses
# and the only one whose opening effect replicates cleanly.
RANK_BLOCK = "post"

# Which rounds a cell was fitted on; see SAMPLES in 11_window_compare.py. The
# figures lead with `channel`, which conditions only on the randomized
# availability of a chat channel. `talkers` conditions on whether the group chose
# to speak, which is a post-treatment behaviour, and is reported as a robustness
# check rather than as the headline.
MAIN_SAMPLE = "channel"
SAMPLE_NOTE = {
    "channel": "all rounds with a chat channel open",
    "talkers": "only rounds in which someone spoke",
}

# Fixed slots, so a block is the same colour in every figure of the set.
def _semantic_groupings():
    """Map each toolkit column to the semantic grouping the toolkit itself assigns.

    The case study's own FEATURE_FAMILIES are a regex convenience invented here;
    these are the categories the Team Communication Toolkit documents its features
    under, so a reader can carry a grouping from the figure back to the docs. The
    dictionary keys features by their base column, and the analysis reports
    aggregated forms of those columns (``max_``, ``stdev_user_min_``, and so on),
    so lookup is by longest matching suffix.

    A handful of features - the LIWC lexicons - are documented under more than one
    grouping. A mark can only take one colour, so the first listed wins, and the
    caption says so.
    """
    from team_comm_tools.feature_dict import feature_dict
    mapping = {}
    for spec in feature_dict.values():
        grouping = spec["semantic_grouping"]
        if isinstance(grouping, list):
            grouping = grouping[0]
        if grouping in ("N/A", None):
            continue
        for column in spec["columns"]:
            mapping[column] = grouping
    return mapping


SEMANTIC_GROUPING = _semantic_groupings()
# Longest first, so `gini_coefficient_sum_num_messages` resolves to Equality
# rather than to the `num_messages` it happens to end with.
_GROUPING_KEYS = sorted(SEMANTIC_GROUPING, key=len, reverse=True)


def semantic_grouping_of(feature):
    for column in _GROUPING_KEYS:
        if feature == column or feature.endswith("_" + column):
            return SEMANTIC_GROUPING[column]
    return "Other"


# The toolkit's categories, in a fixed order with fixed colours.
# Three tones of clearly different lightness, so the three states are still
# distinguishable printed in black and white: dark blue, mid red, light grey.
BETTER, WORSE, NEUTRAL = style.BLUE_DARK, "#d0342c", style.GRAY_LIGHT
# Figure 2 encodes something different from figure 1 - the direction of a feature's
# effect, not whether the block helped prediction - so it gets its own pair of
# colours rather than reusing blue and red for a second meaning.
MORE, LESS = style.GREEN, style.ORANGE

GROUPING_COLORS = {
    "Content": style.BLUE,
    "Emotion": style.ORANGE,
    "Engagement": style.GREEN,
    "Equality": style.MAGENTA,
    "Pace": style.VIOLET,
    "Quantity": style.YELLOW,
    "Variance": style.AQUA,
    "Other": style.INK_MUTED,
}

# Stage names, in reading order, matching the analysis tables.
STAGE_ORDER = ["opening", "middle", "endgame"]

BLOCK_COLOR = {"pre": style.BLUE, "post": style.ORANGE,
               "window": style.AQUA, "cumulative": style.VIOLET}
BLOCK_ORDER = ["pre", "post", "window", "cumulative"]
ALL_BLOCKS = tuple(BLOCK_ORDER)

# The blocks are defined relative to one decision, so the legend has to say what
# each one is or the figure is unreadable to anyone who has not read the README.
# Short enough to sit as a panel title without colliding with its neighbour.
BLOCK_SHORT = {
    "pre": "this round, before the outcome",
    "post": "last round, after the outcome",
    "window": "post + pre together",
    "cumulative": "everything so far this game",
}

BLOCK_LABEL = {
    "pre": "pre - this round, before the outcome",
    "post": "post - last round, after the outcome",
    "window": "window - the whole gap (post + pre)",
    "cumulative": "cumulative - everything so far",
}


def read(name):
    path = TABLES_WINDOWS / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing - run scripts/11_window_compare.py")
    return pd.read_csv(path)


# Bin labels have to be put back in reading order explicitly. They arrive as
# strings and the tables are written cell-by-cell, so relying on order of
# appearance silently puts "rounds 11+" between "round 3" and "rounds 4-6".
STAGE_SORT = {"opening": 0, "middle": 1, "endgame": 2}


def _bin_key(label):
    if label in STAGE_SORT:
        return STAGE_SORT[label]
    digits = "".join(c if c.isdigit() else " " for c in label).split()
    return int(digits[0]) if digits else -1


def bin_order(df, binning):
    """Bins in reading order: by stage, or by the round the label starts at."""
    return sorted(set(df.loc[df["binning"] == binning, "bin"]), key=_bin_key)


def robust_limits(*arrays, pad=0.12, floor=0.05):
    """Axis limits that the bulk of the intervals fit inside, always spanning zero.

    A handful of cells - a small common subsample fitted with 151 features - return
    intervals an order of magnitude wider than everything else. Scaling to them
    would flatten every real effect to an invisible dot, so the axis is set from the
    central mass and the few longer intervals run off the end.
    """
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if not len(values):
        return -floor, floor
    lo, hi = np.nanpercentile(values, [4, 96])
    lo, hi = min(lo, -floor), max(hi, floor)
    span = hi - lo
    return lo - pad * span, hi + pad * span


# =============================================================== core set ===
# Three figures, mirroring the three the case study already reports, each widened
# from two conversation windows to four:
#
#   c1  when in a game talk predicts contribution        (the paper's fig1)
#   c2  which features carry it in the opening           (the paper's fig2)
#   c3  whether those features keep predicting later     (the paper's fig3)
#
# All three read the *marginal* estimates - each window fitted on the rounds that
# had that window's talk - because those have the most games behind them and they
# reproduce the published result. The paired estimates, which fit every window on
# one common subsample, answer a narrower question ("is one window better than
# another on identical rounds?") at a third of the sample, and belong in the
# appendix rather than in the figure that establishes when talk matters.


def legend_estimate(ax):
    """Both encodings: marker shape for the estimate, ink for what it means.

    Sign carries meaning here that a reader cannot infer from the axis alone. A
    negative value is not an effect in the other direction - it means the
    conversation features made out-of-sample prediction *worse* than the controls
    on their own, which is what an overfitting block looks like. Colouring the two
    halves of the plane differently says so without a sentence.
    """
    ax.plot([], [], "o", color=style.INK, markerfacecolor=style.INK,
            markersize=6, linestyle="none",
            label="cross-validated on the learning games")
    ax.plot([], [], "o", color=style.INK, markerfacecolor="none",
            markersize=6, markeredgewidth=1.6, linestyle="none",
            label="scored on held-out games")
    ax.plot([], [], "-", color=BETTER, linewidth=3.2,
            label="improves prediction")
    ax.plot([], [], "-", color=WORSE, linewidth=3.2,
            label="makes prediction worse")
    ax.plot([], [], "-", color=NEUTRAL, linewidth=3.2,
            label="interval includes zero")


CAPTION_C1 = """Variance in a group's mean contribution explained by its \
conversation *in addition to* what the game's parameters already explain. The \
baseline model contains the game's randomized design parameters (group size, \
multiplier, punishment and reward rules), the round's position in its game, and \
indicators for whether the group spoke at all; the value plotted is how much the \
151 conversation features add on top of that. Above zero the features improve \
prediction; below zero they make it worse than the baseline alone, which is what \
an overfitting block of features looks like rather than an effect in the opposite \
direction. The counts under each label are the learning-split game-rounds and \
games behind the cross-validated mark; the held-out mark beside it is scored on \
the corresponding validation games. Opening is the first three rounds of a game, \
Endgame the last three.

Each mark carries a 95% percentile interval from 2,000 bootstrap resamples of \
whole games: the learning games for the cross-validated mark, the validation \
games for the held-out one. A cell is treated as evidence only where both \
estimates clear zero in the same direction; a held-out mark that clears zero on \
its own, as at Middle/Post and Endgame/Pre here, is not.
"""


def sample_line(sample):
    return (f"Fitted on {SAMPLE_NOTE[sample]}."
            + (" Rounds in which nobody spoke are retained, carrying a neutral "
               "fill, and indicators for whether anyone spoke are in both the "
               "baseline and the full model, so the quantity plotted is content "
               "net of speech." if sample == "channel" else
               " This conditions on a post-treatment behaviour and is reported as "
               "a robustness check."))


def fig_c1_when(kind="elastic net", controls="rules+timing",
                blocks=MAIN_BLOCKS, sample=MAIN_SAMPLE, out_dir=None, stem=None,
                variation=None):
    """When in a game does conversation predict contribution, for each window."""
    table = read("block_delta_r2")
    sub = table[(table["binning"] == "stage") & (table["model_family"] == kind)
                & (table["controls"] == controls) & (table["sample"] == sample)]
    if sub.empty:
        print("no marginal rows for this slice; skipping c1")
        return
    stages = [st for st in STAGE_ORDER if st in set(sub["bin"])]
    ylim = robust_limits(sub["ci_low"], sub["ci_high"], sub["delta_cv_r2"],
                         sub["ci_low_heldout"], sub["ci_high_heldout"])

    out_dir = out_dir or MAIN_DIR
    stem = stem or f"fig1_when_talk_matters_{kind.replace(' ', '_')}"
    fig, axes = plt.subplots(1, len(blocks), sharey=True,
                             figsize=(3.6 * len(blocks) + 0.6, 5.2))
    axes = np.atleast_1d(axes)
    DODGE = 0.10

    for ax, block in zip(axes, blocks):
        b = sub[sub["block"] == block].set_index("bin")
        for i, stage in enumerate(stages):
            if stage not in b.index:
                continue
            r = b.loc[stage]
            for dx, lo_c, hi_c, mid_c, filled in (
                    (-DODGE, "ci_low", "ci_high", "delta_cv_r2", True),
                    (+DODGE, "ci_low_heldout", "ci_high_heldout",
                     "delta_heldout_r2", False)):
                if r[lo_c] > 0:
                    colour = BETTER
                elif r[hi_c] < 0:
                    colour = WORSE
                else:
                    colour = NEUTRAL
                ax.plot([i + dx, i + dx], [r[lo_c], r[hi_c]], color=colour,
                        linewidth=1.8, solid_capstyle="butt", zorder=3)
                ax.plot([i + dx], [r[mid_c]], "o", markersize=6, color=colour,
                        markerfacecolor=colour if filled else "none",
                        markeredgewidth=1.6, zorder=4)
        ax.axhline(0, color=style.INK_MUTED, linewidth=1, zorder=2)
        # Above the line the features are earning their place; below it they are
        # costing accuracy. Tinting the halves makes that readable at a glance.
        ax.axhspan(0, ylim[1], color=BETTER, alpha=0.05, zorder=0, linewidth=0)
        ax.axhspan(ylim[0], 0, color=WORSE, alpha=0.05, zorder=0, linewidth=0)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(
            [f"{st.capitalize()}\n{int(b.loc[st, 'n_rounds']):,} rounds"
             f"\n{int(b.loc[st, 'n_games'])} games" if st in b.index else
             st.capitalize()
             for st in stages], fontsize=8.5)
        ax.set_xlim(-0.5, len(stages) - 0.5)
        ax.set_ylim(*ylim)
        ax.set_xlabel("position of the round in its game")
        ax.set_title(f"$\\bf{{{block.capitalize()}}}$\n{BLOCK_SHORT[block]}",
                     fontsize=11.5, color=style.INK, fontweight="normal",
                     loc="left", pad=12)
    axes[0].set_ylabel("additional variance explained (R²)")

    legend_estimate(axes[0])
    style.header(
        fig, axes,
        # Two lines. header() draws the headline with its bottom on the reserved
        # band, so extra lines grow upward into the margin rather than down onto
        # the legend, and the tight bounding box takes them in.
        "Conversation predicts contribution only in the Opening rounds,\n"
        "after revealing contribution outcomes",
        subtitle=variation,
        legend_from=axes[0], ncol=3, panel_titles=True, extra_top=0.10,
        headline_weight="bold", headline_size=14, legend_gap=0.18,
        legend_borderpad=0.0)
    (out_dir / f"{stem}_caption.txt").write_text(
        CAPTION_C1 + "\n" + sample_line(sample) + "\n")
    out = out_dir / f"{stem}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


CAPTION_C2 = """Conversation features that predict a group's mean contribution in \
the Opening three rounds of a game, over the game's rules and the round's position. \
Rows are the features that replicate in at least one window - significant after \
false-discovery correction on the learning games, and significant with the same \
sign on held-out games - and every window is shown for every row. Marks are the \
change in contribution rate per standard deviation of the feature, on the learning \
games, with 95% intervals clustered by game. Colour is the semantic grouping the \
Team Communication Toolkit documents the feature under; the LIWC lexicons are \
documented under three groupings at once and are shown under the first. A filled \
mark replicates in validation, an open one does not.
"""

N_FEATURES_SHOWN = 18


def ranked_features(cell, n):
    """The rows for figures 2 and 3: chosen and ordered once, from RANK_BLOCK.

    Both figures use this so they are a matched pair - figure 2 shows these
    features in the opening, figure 3 follows the same features across the game.
    Ranking from one named window rather than from whichever window flatters each
    feature keeps a single meaning on the axis, and keeps the main and appendix
    versions on identical rows so they can be read side by side.
    """
    ranker = cell[(cell["block"] == RANK_BLOCK) & cell["replicates"]]
    if ranker.empty:
        return [], {}
    best = ranker[["feature", "coef_learn"]].copy()
    best["grouping"] = best["feature"].map(semantic_grouping_of)
    best = best.nlargest(n, "coef_learn")
    rank = {g: i for i, g in enumerate(GROUPING_COLORS)}
    best = best.assign(_f=best["grouping"].map(rank)).sort_values(
        ["_f", "coef_learn"], ascending=[False, True])
    return best["feature"].tolist(), dict(zip(best["feature"], best["grouping"]))


def fig_c2_opening_features(bin_label="opening", blocks=MAIN_BLOCKS,
                            sample=MAIN_SAMPLE, out_dir=None, stem=None):
    """Which features carry the opening effect, and whether the window changes them.

    Coloured by family rather than by significance, matching the published fig2:
    the families are the substantive grouping, and a reader who wants to know what
    kind of talk this is needs to see them grouped. Replication is carried by the
    fill instead.
    """
    e = read("block_feature_effects")
    cell = e[(e["binning"] == "stage") & (e["bin"] == bin_label)
             & (e["sample"] == sample)]

    order, family_of = ranked_features(cell, N_FEATURES_SHOWN)
    if not order:
        print(f"nothing replicates in {RANK_BLOCK}; skipping c2")
        return

    out_dir = out_dir or MAIN_DIR
    stem = stem or "fig2_opening_features"
    fig, axes = plt.subplots(1, len(blocks), sharey=True, sharex=True,
                             figsize=(3.5 * len(blocks) + 2.2,
                                      0.36 * len(order) + 3.6))
    axes = np.atleast_1d(axes)
    ys = np.arange(len(order))
    seen = set()

    for ax, block in zip(axes, blocks):
        b = cell[cell["block"] == block].set_index("feature")
        for y, feature in zip(ys, order):
            if feature not in b.index:
                continue
            r = b.loc[feature]
            colour = GROUPING_COLORS.get(family_of[feature], style.INK_MUTED)
            label = None
            if ax is axes[0] and family_of[feature] not in seen:
                label = family_of[feature]
                seen.add(family_of[feature])
            ax.hlines(y, PP * r["ci_low_learn"], PP * r["ci_high_learn"],
                      color=colour, alpha=0.4, linewidth=2.5, zorder=3)
            ax.scatter(PP * r["coef_learn"], y, s=52, zorder=4, label=label,
                       color=colour if r["replicates"] else style.SURFACE,
                       edgecolor=colour, linewidth=1.6)
        ax.axvline(0, color=style.INK_MUTED, linewidth=1, zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels([style.bold_label(pretty(f)) for f in order], fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("change in contribution (percentage points\nof endowment) per SD of the feature")
        ax.set_title(f"$\\bf{{{block.capitalize()}}}$\n{BLOCK_SHORT[block]}",
                     fontsize=11.5, color=style.INK, fontweight="normal",
                     loc="left", pad=12)
    axes[0].margins(y=0.03)
    # The legend fills column by column over two rows, so an odd number of entries
    # splits the replicates/does-not pair across two columns and they land on the
    # diagonal. One blank entry, when needed, keeps them stacked in one column.
    if (len(seen) + 2) % 2:
        axes[0].scatter([], [], s=0, color="none", edgecolor="none", label=" ")
    axes[0].scatter([], [], s=52, color=style.INK_MUTED, edgecolor=style.INK_MUTED,
                    linewidth=1.6, label="replicates in validation")
    axes[0].scatter([], [], s=52, color=style.SURFACE, edgecolor=style.INK_MUTED,
                    linewidth=1.6, label="does not replicate in validation")
    legend_cols = (len(seen) + 2 + ((len(seen) + 2) % 2)) // 2

    style.header(
        fig, axes,
        "Top predictive features in the opening rounds, by conversation window",
        legend_from=axes[0], ncol=legend_cols, panel_titles=True, extra_top=0.34,
        headline_weight="bold", headline_size=14)
    (out_dir / f"{stem}_caption.txt").write_text(
        CAPTION_C2 + "\n" + sample_line(sample) + "\n")
    out = out_dir / f"{stem}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


CAPTION_C3 = """Each row is one of the strongest Opening-round features for that \
window, re-estimated separately within each time period of the game. A feature is \
eligible if it survives false-discovery correction on the learning games in the \
Opening, and the {n_shown} eligible features with the largest Opening coefficients are \
shown, ordered by that coefficient; the panels therefore carry different features, \
and different numbers of them, because the windows found different things. Filled \
marks replicate in validation: significant after correction on the learning games \
and significant with the same sign on held-out games. Opening is the first three \
rounds of a game, Endgame the last three. Horizontal bars are 95% \
intervals from a regression of the outcome on that feature alone plus the game's \
rules and the round's position, with standard errors clustered by game.
"""

N_TRACKS = 8


def fig_c3_across_stages(top_n=N_TRACKS, blocks=MAIN_BLOCKS, sample=MAIN_SAMPLE,
                         out_dir=None, stem=None, variation=None,
                         headline="Opening-round features are no longer "
                                  "predictive in later rounds"):
    """Do the opening features keep predicting once the game is underway?

    A slopegraph per window, in the form the published fig3 uses: one line per
    feature, named at the right-hand end so a reader can see *which* effect
    collapses rather than only that something does.
    """
    e = read("block_feature_effects")
    e = e[(e["binning"] == "stage") & (e["sample"] == sample)]
    stages = [st for st in STAGE_ORDER if st in set(e["stage"] if "stage" in e
                                                    else e["bin"])]
    col = "stage" if "stage" in e.columns else "bin"
    stages = [st for st in STAGE_ORDER if st in set(e[col])]
    x = np.arange(len(stages))

    # The feature names hang off the right of each panel, so the gap between
    # columns has to be wide enough to hold them; at the default they land on the
    # next panel's y axis.
    out_dir = out_dir or MAIN_DIR
    stem = stem or "fig3_effects_across_stages"

    # Each window is ranked by its own strongest opening features, so a panel shows
    # what that window actually found rather than what another window found.
    #
    # Selection is screened by false-discovery correction on the learning games,
    # and the fill still means replication on held-out games. Both matter: an
    # earlier version selected on an uncorrected p<0.05 and filled on the same
    # rule, which drew Pre's chance hits - 17 of 151, about what 151 tests return
    # by luck - as solid marks. Under FDR, Pre has 2 candidates and neither
    # replicates, and the panel says so at a glance.
    # Tracks first, layout second: a window with nothing to show is dropped from
    # the figure rather than drawn as an empty pair of axes, and the caption says
    # which windows were dropped and why.
    by_block = {}
    for block in blocks:
        sub = e[e["block"] == block]
        opening = sub[(sub[col] == "opening") & (sub["q_learn"] < 0.05)]
        order = opening.nlargest(top_n, "coef_learn")["feature"].tolist()
        tracks = []
        for feature in order:
            by_stage = sub[sub["feature"] == feature].set_index(col)
            if not all(st in by_stage.index for st in stages):
                continue
            # Coefficients are on the 0-1 contribution rate; they are plotted in
            # percentage points of the endowment, which is how the case study
            # reports every effect on contribution.
            tracks.append((feature,
                           [PP * by_stage.loc[st, "coef_learn"] for st in stages],
                           [bool(by_stage.loc[st, "replicates"]) for st in stages],
                           [PP * by_stage.loc[st, "ci_low_learn"] for st in stages],
                           [PP * by_stage.loc[st, "ci_high_learn"] for st in stages]))
        if tracks:
            by_block[block] = tracks

    drawn = [b for b in blocks if b in by_block]
    omitted = [b for b in blocks if b not in by_block]
    if not drawn:
        print(f"nothing survives correction in any window; skipping {stem}")
        return

    # One row per window, one column per time period, and one row of the y axis per
    # feature. An earlier version drew each feature as a line across the three time
    # periods, which put eight lines of one colour through one another and left the
    # reader to follow a grey leader line to the name. Giving each feature its own
    # y position separates them by position instead of by hue, so the figure holds
    # up in greyscale and for a reader who cannot distinguish the colours.
    n_rows = max(len(by_block[b]) for b in drawn)
    fig, axes = plt.subplots(
        len(drawn), len(stages), sharey="row", squeeze=False,
        figsize=(11.5, 1.9 + 0.42 * n_rows * len(drawn)),
        gridspec_kw={"wspace": 0.12, "hspace": 0.42})

    for row, block in enumerate(drawn):
        tracks = by_block[block]
        # Strongest at the top, which is the order the Opening column establishes
        # and the other two columns inherit.
        tracks = sorted(tracks, key=lambda t: t[1][0])
        ys = np.arange(len(tracks))
        x_lo = min(min(l) for _, _, _, l, _ in tracks)
        x_hi = max(max(h) for _, _, _, _, h in tracks)
        x_pad = (x_hi - x_lo) * 0.06
        for cell, stage in enumerate(stages):
            ax = axes[row][cell]
            for y, (feature, values, sig, lows, highs) in zip(ys, tracks):
                # One colour for every feature: the sign of the estimate is
                # already on the x axis, so colour would only repeat it.
                colour = MORE
                ax.plot([lows[cell], highs[cell]], [y, y], color=colour,
                        alpha=0.55, linewidth=1.8, solid_capstyle="butt", zorder=3)
                ax.scatter(values[cell], y, s=48, zorder=4,
                           color=colour if sig[cell] else style.SURFACE,
                           edgecolor=colour, linewidth=1.4)
            ax.axvline(0, color=style.INK, linewidth=1.1, zorder=2)
            ax.set_ylim(-0.7, len(tracks) - 0.3)
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
            ax.set_yticks(ys)
            if cell == 0:
                ax.set_yticklabels(
                    [style.bold_label(pretty(f)) for f, _, _, _, _ in tracks],
                    fontsize=8)
            ax.set_title(stage.capitalize(), fontsize=10.5, color=style.INK_2,
                         loc="left", pad=6)
            if row == len(drawn) - 1:
                ax.set_xlabel("change in contribution\n(percentage points) per SD"
                              if cell == 1 else "")
        axes[row][0].annotate(
            f"$\\bf{{{block.capitalize()}}}$: {BLOCK_SHORT[block]}",
            xy=(0, 1.0), xytext=(0, 26), xycoords="axes fraction",
            textcoords="offset points", ha="left", va="bottom",
            fontsize=11.5, color=style.INK, annotation_clip=False)

    axes[0][0].scatter([], [], s=48, color=style.INK_MUTED,
                       edgecolor=style.INK_MUTED, linewidth=1.4,
                       label="replicates in validation")
    axes[0][0].scatter([], [], s=48, color=style.SURFACE,
                       edgecolor=style.INK_MUTED, linewidth=1.4,
                       label="does not replicate in validation")

    style.header(
        fig, axes, headline, subtitle=variation,
        legend_from=axes[0][0], ncol=2, panel_titles=True, extra_top=0.34,
        headline_weight="bold", headline_size=14, legend_borderpad=0.0)
    note = ""
    if omitted:
        names = " and ".join(b.capitalize() for b in omitted)
        note = (f" {names} {'is' if len(omitted) == 1 else 'are'} not shown: not "
                f"one of its features survives false-discovery correction in the "
                f"opening rounds, so there is nothing to follow across the game.")
    (out_dir / f"{stem}_caption.txt").write_text(
        CAPTION_C3.format(n_shown=top_n).rstrip() + note + "\n"
        + sample_line(sample) + "\n")
    out = out_dir / f"{stem}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    style.use_style()

    # Main: the penalized linear model, on the two halves of the gap, fitted on
    # every round where a channel was open. The elastic net is canonical because it
    # is the more conservative of the two families; the `channel` sample is
    # canonical because it conditions only on the randomized treatment.
    #
    # Figure 2 has no model family to choose: it is one OLS per feature, clustered
    # by game, not a fit of the whole feature block.
    fig_c1_when(stem="fig1_when_talk_matters")
    fig_c3_across_stages(stem="fig2_effects_across_stages", headline=None)

    # Appendix, in the order the supplement reports them: the other sample, the
    # other model family, then the two wider windows.
    TALKERS = ("Robustness check: silent game-rounds dropped, "
               "rather than retained through the neutral fill")
    FOREST = "Robustness check: the random forest in place of the ElasticNet"
    WIDE = ("Robustness check: the Window and Cumulative conversation "
            "boundaries in place of Pre and Post")
    fig_c1_when(sample="talkers", out_dir=APPENDIX_DIR, variation=TALKERS,
                stem="figS1_when_talk_matters_talkers_only")
    fig_c3_across_stages(sample="talkers", out_dir=APPENDIX_DIR, variation=TALKERS,
                         stem="figS2_effects_across_stages_talkers_only")
    fig_c1_when(kind="random forest", out_dir=APPENDIX_DIR, variation=FOREST,
                stem="figS3_when_talk_matters_random_forest")
    fig_c1_when(blocks=APPENDIX_BLOCKS, out_dir=APPENDIX_DIR, variation=WIDE,
                stem="figS4_when_talk_matters_wide_windows")
    fig_c3_across_stages(blocks=APPENDIX_BLOCKS, out_dir=APPENDIX_DIR,
                         variation=WIDE,
                         stem="figS5_effects_across_stages_wide_windows")

    # fig_c2_opening_features() is deliberately not part of the figure set: it is a
    # single time slice, and the across-stages figure carries the same coefficients
    # with the time axis attached. It remains callable for the per-feature
    # intervals on their own.
    print(f"main figures -> {MAIN_DIR}\nappendix figures -> {APPENDIX_DIR}")
