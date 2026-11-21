"""
Shared figure style
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from config import FIGURES_DIR

ACCENT_FILL_COLOR = "#E39B9B"
CHURN_COLORS = {"No": "#5B84B1", "Yes": ACCENT_FILL_COLOR}
MAIN_COLOR = "#5B84B1"
ACCENT_COLOR = "#D97878"
REFERENCE_COLOR = "#555555"
SERIES_COLORS = ["#5B84B1", "#D97878", "#6A9F58"]

_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")


def set_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "figure.dpi": 120,
        "savefig.dpi": 300
    })


def decimal_comma(text: str) -> str:
    parts = text.split("$")
    return "$".join(_DECIMAL_POINT.sub("{,}" if i % 2 else ",", part) for i, part in enumerate(parts))


def number(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def percent(value: float, decimals: int = 1) -> str:
    return f"{100 * value:.{decimals}f}".replace(".", ",") + "%"


def comma_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def percent_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda v, _: percent(v, decimals))


def thousands_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v / 1000:g}".replace(".", ",").replace("-", "−"))


def finish_axes(ax) -> None:
    ax.grid(False)
    sns.despine(ax=ax)
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "linear" and type(axis.get_major_formatter()) is ScalarFormatter:
            axis.set_major_formatter(comma_formatter())


def commas_in_figure(fig) -> None:
    for ax in fig.axes:
        for text in ax.texts:
            text.set_text(decimal_comma(text.get_text()))
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(comma_formatter())
            else:
                old = axis.get_ticklabels()
                new = [decimal_comma(label.get_text()) for label in old]
                if new != [label.get_text() for label in old]:
                    axis.set_ticks(axis.get_ticklocs())
                    axis.set_ticklabels(new, rotation=old[0].get_rotation(), ha=old[0].get_ha())


def save_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)
