"""Figure: comparison with official results.

Left panel: grouped bars, PolicyEngine static costing of the basic-rate +1pp
reform against HMRC's published ready reckoner (June 2025 vintage: GBP 6.9bn
in 2026-27, GBP 8.2bn a year in 2028-29).

Right panels: the anchored real-GDP path overlaid on the published OBR
November 2025 EFO path (top) and the quarterly percentage deviation with the
+/-1 per cent CI gate band (bottom), from figures/fig_anchored_data.csv --
the same artifact behind Figure fig_anchored, produced by the anchored model
run in figures/fig_anchored.py.

Okabe-Ito colorblind-safe palette via style.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import BLUE, GREY, ORANGE, VERMILLION

HERE = Path(__file__).parent

# Basic-rate +1pp static costing, GBP bn/yr: ours (PolicyEngine bridge, scored
# endpoints) vs HMRC ready reckoner, June 2025 ODS vintage.
GROUPS = ["2026–27", "2028–29"]
OURS = [6.46, 6.92]  # 2028 value linearly interpolated between 6.46 and 7.38
HMRC = [6.9, 8.2]


def main():
    df = pd.read_csv(HERE / "fig_anchored_data.csv", index_col=0)
    periods = list(df.index)
    gdp_m = df["GDPM_model"].values / 1000.0  # GBP bn per quarter
    gdp_e = df["GDPM_efo"].values / 1000.0
    dev = 100 * (gdp_m - gdp_e) / gdp_e

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[2, 1],
                          hspace=0.12, wspace=0.32)

    # Left: grouped bars over both rows.
    ax = fig.add_subplot(gs[:, 0])
    x = np.arange(len(GROUPS))
    w = 0.34
    ax.bar(x - w / 2, OURS, w, color=BLUE, label="PolicyEngine (ours)")
    ax.bar(x + w / 2, HMRC, w, color=ORANGE, label="HMRC ready reckoner")
    for xi, (o, h) in enumerate(zip(OURS, HMRC)):
        ax.text(xi - w / 2, o + 0.12, f"{o:.2f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, h + 0.12, f"{h:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("£bn per year")
    ax.set_ylim(0, 9.6)
    ax.set_title("Basic rate +1pp: static costing", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")

    # Right top: anchored GDP overlay.
    ax1 = fig.add_subplot(gs[0, 1])
    t = np.arange(len(periods))
    ax1.plot(t, gdp_e, color=ORANGE, lw=1.8, label="OBR EFO (Nov 2025)")
    ax1.plot(t, gdp_m, color=BLUE, lw=1.4, ls="--", label="Anchored emulator")
    ax1.set_ylabel("£bn/qtr")
    ax1.set_title("Anchored real GDP vs published EFO", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.tick_params(labelbottom=False)

    # Right bottom: percentage deviation with the CI gate band.
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.axhspan(-1, 1, color=GREY, alpha=0.15, label="±1% CI gate")
    ax2.fill_between(t, dev, 0, color=VERMILLION, alpha=0.35)
    ax2.plot(t, dev, color=VERMILLION, lw=1.2)
    ax2.axhline(0, color=GREY, lw=0.8)
    ax2.set_ylabel("% dev.")
    ax2.set_ylim(-1.2, 1.2)
    ticks = list(range(0, len(periods), 2))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([periods[i] for i in ticks], rotation=45,
                        ha="right", fontsize=7)
    ax2.legend(fontsize=7, loc="upper left")

    fig.savefig(HERE / "fig_comparison.pdf", bbox_inches="tight")

    pd.DataFrame({"period": periods, "gdp_model_bn": gdp_m,
                  "gdp_efo_bn": gdp_e, "pct_dev": dev}).to_csv(
        HERE / "fig_comparison_data.csv", index=False)
    print("wrote fig_comparison.pdf")
    print("max |dev| GDP:", f"{np.max(np.abs(dev)):.3f}%")


if __name__ == "__main__":
    main()
