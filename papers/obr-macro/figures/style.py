"""Shared matplotlib style for the OBR-emulator working paper figures.

Colorblind-safe Okabe-Ito palette; PDF output for LaTeX inclusion.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"  # model / emulator
ORANGE = "#E69F00"  # OBR EFO (official)
GREEN = "#009E73"
VERMILLION = "#D55E00"
GREY = "#666666"
SKY = "#56B4E9"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def quarters_axis(ax, periods, step=2):
    """Tidy quarterly x-axis labels."""
    ticks = list(range(0, len(periods), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(periods[i]) for i in ticks], rotation=45, ha="right")
