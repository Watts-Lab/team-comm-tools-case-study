"""Shared matplotlib styling so every figure in outputs/figures/ reads as one set."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, assigned in fixed order and never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e5e4df"

# Roles used throughout the case study: the contrast the question is built on.
COLOR_CHANNEL = BLUE       # groups that could communicate
COLOR_NO_CHANNEL = ORANGE  # groups that could not
COLOR_LEARN = BLUE
COLOR_VAL = AQUA


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
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


def title(ax, headline, subtitle=None):
    """Headline in ink, optional subtitle beneath it in secondary ink."""
    ax.set_title(headline, color=INK, pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_2, va="bottom", ha="left")


def caption(fig, text):
    fig.text(0.0, -0.04, text, fontsize=8, color=INK_MUTED, ha="left", va="top")
