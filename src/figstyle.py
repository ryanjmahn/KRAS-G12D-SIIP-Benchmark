"""Figure style shared by both phases, so the two papers' figures read as one set.

Palette: the four categorical slots are fixed per tool and never recycled, so a tool keeps its colour in every
figure of either phase. They pass the colourblind-separation and contrast checks on a white surface.

Conventions used throughout: horizontal dot-and-interval rather than bars wherever a confidence interval
exists, a hollow marker for the secondary arm of a pair, grey for a reference level (chance, control, apo),
thin axes with only the grid that the reading needs, and direct labels in place of a legend where there is
room for them.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOOLS = ["fpocket", "P2Rank", "AE-PocketMiner", "pLM-NN"]
COLOR = dict(zip(TOOLS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))   # validated categorical slots 1-4
GREY = "#8a8a85"
INK = "#333333"


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"], "font.size": 8,
                         "axes.linewidth": 0.6, "axes.edgecolor": INK, "savefig.dpi": 400,
                         "savefig.bbox": "tight", "lines.linewidth": 2})


def finish(ax, grid="x"):
    ax.spines[["top", "right"]].set_visible(False)
    getattr(ax, f"{grid}axis").grid(True, color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name, fmts, out):
    out.mkdir(parents=True, exist_ok=True)
    for f in fmts:
        fig.savefig(out / f"{name}.{f}")
    plt.close(fig)


def dot(ax, y, v, lo, hi, color, hollow=False):
    """Horizontal point estimate with its interval. Hollow marks the secondary arm of a pair."""
    ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", ms=6, color=color, mfc="white" if hollow else color,
                mew=1.5, elinewidth=1.2, capsize=0)


def boot_ci(x, stat=np.mean, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(stat(x[idx], axis=1), [2.5, 97.5]))
