"""Shared matplotlib styling so every figure in outputs/figures/ reads as one set.

FIGURE RULES - these are not preferences, and there are no exceptions.

1. A figure must be readable by someone who has never seen this repository and was
   not present for any discussion about it. Nothing in a figure may refer to the
   process that produced it: no notes about why a model was specified one way
   rather than another, no justification of an analysis choice, no allusion to an
   earlier version. Those belong in the README. If a sentence would only make
   sense to someone who watched the analysis being built, it is prohibited.

2. Show only what is strictly necessary to read the chart. Necessary means: what
   the axes are, what the marks are, what the uncertainty is, and what units. A
   fact that is merely interesting, defensive, or explanatory goes in the README.

3. No small grey caption text under the figure. It is unreadable at any realistic
   size and is where extraneous explanation accumulates. Anything genuinely
   required goes in the subtitle, directly under the headline, in legible type.

4. The legend goes directly under the subtitle, above the plot area, laid out
   horizontally. It never floats inside the data area, where it can cover marks.

5. Nothing overlaps. Not title and subtitle, not subtitle and legend, not panel
   titles and the headline, not tick labels and a neighbouring panel.

6. Where a direction on an axis carries meaning that a reader cannot infer from
   the label, mark the direction on the axis itself rather than explaining it in
   prose.

`header()` implements rules 3-5; use it for every figure and do not hand-place
titles, subtitles, or legends.
"""

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots, assigned in fixed order and never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET = "#e87ba4", "#008300", "#4a3aa7"
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e5e4df"

# Roles used throughout the case study: the contrast the question is built on.
COLOR_CHANNEL = BLUE       # groups that could communicate
COLOR_NO_CHANNEL = ORANGE  # groups that could not
COLOR_LEARN = BLUE
COLOR_VAL = AQUA
COLOR_MODEL_LINEAR = BLUE
COLOR_MODEL_FOREST = ORANGE


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
    })


def header(fig, axes, headline, subtitle=None, legend_from=None, ncol=4,
           panel_titles=False, extra_top=0.0, headline_weight="semibold",
           headline_size=12.5, legend_gap=0.0, legend_borderpad=0.4):
    """Stack headline, subtitle and legend above the plot area, never overlapping.

    Each element gets its own horizontal band, measured in inches and converted to
    figure coordinates, so the layout holds at any figure height. Bands are
    allocated bottom-up from the top of the axes: panel titles first (when the
    figure has them), then the legend, then the subtitle, then the headline.

    :param legend_from: axis whose handles supply the legend, or None for no legend.
    :param panel_titles: reserve a band for per-panel titles.
    :param extra_top: further inches to reserve, for figures that stack another
        label above the panel titles (row headers in a grid, for instance).
    :param headline_weight: weight of the headline text; the default keeps the
        published figures exactly as they are.
    :param headline_size: point size of the headline text.
    :param legend_borderpad: padding inside the legend box, in font units.
        Matplotlib's default indents the first handle; pass 0 to sit the handle
        flush with the headline's left edge.
    :param legend_gap: inches to close up between the legend and the headline.
        The legend rises by this much and the header block shrinks with it, so the
        plot area gains the space. Zero leaves the published figures unchanged.
    """
    axes = np.atleast_1d(axes)
    left_ax = axes.flat[0]
    fig_h = fig.get_size_inches()[1]

    handles, labels = [], []
    if legend_from is not None:
        raw_handles, raw_labels = legend_from.get_legend_handles_labels()
        for handle, label in zip(raw_handles, raw_labels):
            if label and not label.startswith("_") and label not in labels:
                handles.append(handle)
                labels.append(label)

    # Matplotlib does not wrap text and `bbox_inches="tight"` grows the canvas to
    # fit whatever it is given, so an unwrapped subtitle silently stretches the
    # saved image to several times its intended width. Wrap to the figure's own
    # width and pay for the extra lines in the reserved band.
    if subtitle:
        subtitle = textwrap.fill(subtitle,
                                 max(60, int(fig.get_size_inches()[0] * 15)))
    sub_lines = subtitle.count("\n") + 1 if subtitle else 0

    H_GAP, H_PANEL, H_LEGEND, H_SUB, H_HEAD = 0.10, 0.46, 0.30, 0.22, 0.40
    # A figure whose panel titles carry the meaning does not need a headline; the
    # band it would occupy is then not reserved, rather than left blank.
    band = H_GAP + (H_HEAD if headline else 0.0)
    band += H_PANEL if panel_titles else 0.0
    # A legend with more entries than columns wraps, and every wrapped row needs
    # its own space or it lands on the subtitle.
    legend_rows = int(np.ceil(len(handles) / ncol)) if handles else 0
    band += H_LEGEND * legend_rows
    band -= legend_gap if handles else 0.0
    band += H_SUB * sub_lines
    band += extra_top

    fig.subplots_adjust(top=1 - band / fig_h)
    x0 = left_ax.get_position().x0
    y = 1 - band / fig_h                      # top edge of the axes, figure coords
    y += extra_top / fig_h
    if panel_titles:
        y += H_PANEL / fig_h
    if handles:
        fig.legend(handles, labels, loc="lower left", ncol=ncol, frameon=False,
                   fontsize=9, handletextpad=0.5, columnspacing=1.6,
                   borderaxespad=0.0, borderpad=legend_borderpad,
                   bbox_to_anchor=(x0, y), bbox_transform=fig.transFigure)
        y += (H_LEGEND * legend_rows - legend_gap) / fig_h
    if subtitle:
        fig.text(x0, y, subtitle, ha="left", va="bottom", fontsize=9.5, color=INK_2,
                 linespacing=1.45)
        y += H_SUB * sub_lines / fig_h
    if headline:
        fig.text(x0, y, headline, ha="left", va="bottom", fontsize=headline_size,
                 fontweight=headline_weight, color=INK)


def axis_direction(ax, low, high, axis="x", pad=-32):
    """Label what each end of an axis means, on the axis itself.

    For quantities where the sign carries the interpretation and a reader cannot
    recover it from the axis label alone.
    """
    kw = dict(xycoords="axes fraction", textcoords="offset points",
              fontsize=8.5, color=INK_MUTED, annotation_clip=False)
    if axis == "x":
        ax.annotate(f"← {low}", xy=(0, 0), xytext=(0, pad), ha="left",
                    va="top", **kw)
        ax.annotate(f"{high} →", xy=(1, 0), xytext=(0, pad), ha="right",
                    va="top", **kw)
    else:
        ax.annotate(f"← {low}", xy=(0, 0), xytext=(pad, 0), ha="right",
                    va="bottom", rotation=90, **kw)
        ax.annotate(f"{high} →", xy=(0, 1), xytext=(pad, 0), ha="right",
                    va="top", rotation=90, **kw)


def panel_title(ax, text):
    """Title for one panel of a multi-panel figure.

    Deliberately heavier and darker than the subtitle: at the same weight the two
    read as one block of text and the panel boundary disappears.
    """
    ax.set_title(text, fontsize=11.5, color=INK, fontweight="semibold", loc="left",
                 pad=12)


FAMILY_SUFFIXES = {"_lexical_wordcount": " (LIWC)", "_politeness_convokit": " (politeness)",
                   "_receptiveness_yeomans": " (receptiveness)", "_bert": " (BERT)",
                   "_chats": "", "_conversation": ""}
# "gini" is deliberately absent: `gini_coefficient_sum_num_chars` is one native
# conversation-level feature, not a gini aggregation of something else, and
# stripping the prefix renamed it to "coefficient sum num chars [gini]".
AGG_WORDS = {"mean": "avg", "max": "max", "min": "min", "stdev": "SD", "sum": "total"}


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


def bold_label(text):
    """Mathtext label with the construct name bold and any qualifier left plain.

    Feature names carry an aggregation qualifier in brackets - "certainty (LIWC)
    [max of speaker min]" - which is needed for precision but swamps the construct
    if both are set the same way. Matplotlib cannot mix weights inside one tick
    label, so the bold half goes through mathtext.
    """
    head, sep, tail = text.partition(" [")
    escaped = head.replace(" ", r"\ ")
    bold = rf"$\bf{{{escaped}}}$"
    return bold + (sep + tail if sep else "")
