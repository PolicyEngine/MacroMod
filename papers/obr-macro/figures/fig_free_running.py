"""Figure 2: free-running (raw, de-seeded) model vs the OBR EFO.

Shows the honest divergence of the raw model (add-factors off, EFO seed
removed) for real GDP, consumption, and household disposable income.
Uses the same de-seeded solve as the calibration scorecard
(obr_macro.calibration_score.raw_solve).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import BLUE, ORANGE, quarters_axis

from obr_macro.calibration_score import raw_solve, START, END
from obr_macro.data import load_obr_data

HERE = Path(__file__).parent


def main():
    s, _, _, t0, t1 = raw_solve()
    efo = load_obr_data()

    model = s.data.loc[START:END]
    official = efo.loc[START:END]
    periods = list(model.index)

    panels = [
        ("GDPM", "Real GDP"),
        ("CONS", "Consumption"),
        ("HHDI", "Household income"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for ax, (code, label) in zip(axes, panels):
        m = model[code].values / 1000.0
        o = official[code].values / 1000.0
        ax.plot(range(len(periods)), o, color=ORANGE, lw=2.2, label="OBR EFO")
        ax.plot(
            range(len(periods)),
            m,
            color=BLUE,
            lw=1.4,
            ls="--",
            marker="o",
            ms=2.5,
            label="Emulator, free-running",
        )
        mape = 100 * np.mean(np.abs(m - o) / o)
        ax.set_title(f"{label}  (MAPE {mape:.1f}%)")
        ax.set_ylabel("£bn per quarter")
        quarters_axis(ax, periods, step=3)
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "fig_free_running.pdf")
    out = model[[c for c, _ in panels]].join(
        official[[c for c, _ in panels]], lsuffix="_model", rsuffix="_efo"
    )
    out.to_csv(HERE / "fig_free_running_data.csv")
    print("wrote fig_free_running.pdf")


if __name__ == "__main__":
    main()
