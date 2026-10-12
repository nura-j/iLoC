"""
plot_style.py
=============
House style for COMPASS paper figures: a warm, colourblind-aware palette on a
white background, with redundant marker/linestyle encodings so figures survive
grayscale printing. Import this module instead of hand-picking colours.

Usage:
    import plot_style as ps
    ps.apply()                      # rcParams: palette cycle, white bg, spines
    fig, ax = plt.subplots()
    ax.plot(x, y); ps.style_lines(ax)
    im = ax.imshow(M, cmap=ps.seq_cmap())          # sequential: white -> blue
    im = ax.imshow(D, cmap=ps.seq_cmap(ps.RED))    # distances: white -> red
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CHARCOAL = "#343A40"
ORANGE = "#E87E37"
SKY_BLUE = "#62B6E6"
TEAL = "#3AA594"
DARK_BLUE = "#203E8C"
RED = "#E7412B"
PURPLE = "#8C4A8A"
GOLD = "#F5C748"  # fills/highlights only — vanishes as a thin line
MAROON = "#660033"  # sequential scales for distances/intensities

# Front-loads the most mutually distinct colours; gold last (fills only).
CYCLE = [CHARCOAL, ORANGE, SKY_BLUE, TEAL, DARK_BLUE, RED, PURPLE, GOLD]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def cycle(n=None):
    """First n palette colours in the recommended order."""
    return CYCLE[: n if n is not None else len(CYCLE)]


def apply():
    """White background, thin dark axes, palette cycle, faint grid."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "text.color": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": CHARCOAL,
        "grid.alpha": 0.12,
        "grid.linewidth": 0.6,
        "font.size": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_lines(ax):
    """Add distinct markers and dashes so colour is never the only signal."""
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(MARKERS[i % len(MARKERS)])
        line.set_linestyle(DASHES[i % len(DASHES)])
        line.set_markersize(5)
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(0.5)


def seq_cmap(color=DARK_BLUE, name=None):
    """Perceptually-monotone sequential map: white -> palette colour."""
    return LinearSegmentedColormap.from_list(
        name or f"seq_{color.strip('#')}", ["#FFFFFF", color])


def heatmap(ax, M, labels, cmap=None, vmin=0.0, vmax=None, fmt="{:.2f}",
            title=None, annot_threshold=0.55, tick_fs=9, annot_fs=8.5,
            title_fs=10.5):
    """Annotated square heatmap with palette styling (no grid, boxed)."""
    cmap = cmap or seq_cmap()
    vmax = vmax if vmax is not None else max(1e-12, float(M.max()))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(CHARCOAL)
        ax.spines[side].set_linewidth(0.8)
    n = len(labels)
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticks(range(n), labels, fontsize=tick_fs)
    for i in range(n):
        for j in range(n):
            frac = (M[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=annot_fs,
                    color="white" if frac > annot_threshold else CHARCOAL)
    if title:
        ax.set_title(title, fontsize=title_fs)
    return im


def save(fig, stem, formats=("pdf", "png")):
    """Save vector + raster next to each other; returns the paths."""
    paths = []
    for ext in formats:
        p = f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
