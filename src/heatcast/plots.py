"""Shared figure style. Each model keeps the same colour in every figure."""
import matplotlib as mpl
import matplotlib.pyplot as plt

MODEL_COLOURS = {
    "Chronos-2+cov": "#2a78d6",
    "Chronos-2+cov+RH": "#184f95",
    "Chronos-2+cov-FT": "#0d366b",
    "Chronos-2": "#eb6834",
    "Chronos-Bolt": "#1baf7a",
    "Chronos-T5": "#eda100",
    "Quantile-LSTM": "#e87ba4",
    "AR-anomaly": "#008300",
    "Persistence": "#4a3aa7",
    "Climatology": "#e34948",
}
MODEL_STYLE = {"Chronos-2+cov+RH": (0, (4, 2)), "Chronos-2+cov-FT": (0, (1, 1.2))}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def setup():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


def clean(ax, grid_axis="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def line(ax, x, y, model, **kw):
    return ax.plot(x, y, color=MODEL_COLOURS[model], ls=MODEL_STYLE.get(model, "-"), label=model, **kw)


def save(fig, path):
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
