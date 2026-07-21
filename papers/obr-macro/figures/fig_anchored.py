"""Figure 1: anchored baseline vs the OBR March 2026 EFO (GDP, consumption).

Runs the anchored solve (add-factors on) over 2025Q1--2027Q4 and plots the
emulator path against the published EFO path. Reproducible:

    ./venv/bin/python fig_anchored.py

with the obr_macro package installed (pip install from the read-only clone of
github.com/PolicyEngine/obr-macroeconomic-model).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import BLUE, ORANGE, quarters_axis

from obr_macro.baseline import build
from obr_macro.data import load_obr_data

HERE = Path(__file__).parent
START, END = "2025Q1", "2027Q4"


def main():
    s = build(anchored=True)
    s.solve(START, END)
    efo = load_obr_data()

    model = s.data.loc[START:END]
    official = efo.loc[START:END]
    periods = list(model.index)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, code, label in [
        (axes[0], "GDPM", "Real GDP ($GDPM$)"),
        (axes[1], "CONS", "Consumption ($CONS$)"),
    ]:
        m = model[code].values / 1000.0
        o = official[code].values / 1000.0
        ax.plot(range(len(periods)), o, color=ORANGE, lw=2.2, label="OBR EFO (Mar 2026)")
        ax.plot(
            range(len(periods)),
            m,
            color=BLUE,
            lw=1.4,
            ls="--",
            marker="o",
            ms=2.5,
            label="Emulator, anchored",
        )
        mape = 100 * np.mean(np.abs(m - o) / o)
        ax.set_title(f"{label}\nMAPE {mape:.2f}\\%" if False else f"{label}  (MAPE {mape:.2f}%)")
        ax.set_ylabel("£bn per quarter, real")
        quarters_axis(ax, periods)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_anchored.pdf")
    # Persist the series for audit
    out = model[["GDPM", "CONS"]].join(
        official[["GDPM", "CONS"]], lsuffix="_model", rsuffix="_efo"
    )
    out.to_csv(HERE / "fig_anchored_data.csv")
    print("wrote fig_anchored.pdf")


if __name__ == "__main__":
    main()
