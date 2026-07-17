"""Figure: forecast vs outturn, and multipliers vs the literature.

Left panel: quarter-on-quarter real GDP growth, 2025Q2-2026Q1 -- the
November-2025-anchored emulator forecast against subsequently published ONS
outturn (GDP first quarterly estimate, Jan-Mar 2026 bulletin, published
14 May 2026: 2025Q2 +0.1, Q3 +0.2, Q4 +0.2, 2026Q1 +0.6 per cent).
Model path from figures/fig_anchored_data.csv (the anchored solve).

Right panel: the emulator's government-consumption multiplier (~1.0) and the
implied income-tax "multiplier" of the 1p basic-rate reform (second-round GDP
effect / HHDI shock, both in per cent of GDP: 0.09 on impact to 0.25 by
2027Q4, from fig_reform_data.csv and the bridge shock of ~GBP 1.6-1.7bn/qtr)
against published UK/advanced-economy estimates: the interim OBR's June 2010
first-year multipliers by instrument (capital spending 1.0, current spending
0.6, VAT 0.35, income tax/NICs 0.3; obr.uk box "Fiscal multipliers"), the
IMF bunching guidance of first-year multipliers of 0-1 for advanced
economies in normal times (Batini et al. 2014), and Ramey's (2019) preferred
0.6-1.0 range for aggregate government purchases.

Okabe-Ito colorblind-safe palette via style.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import BLUE, GREEN, GREY, ORANGE

HERE = Path(__file__).parent

# ONS outturn, qoq real GDP growth %, latest vintage at time of writing
# (GDP first quarterly estimate, UK: January to March 2026, ONS, 14 May 2026).
OUT_PERIODS = ["2025Q2", "2025Q3", "2025Q4", "2026Q1"]
ONS_OUTTURN = [0.1, 0.2, 0.2, 0.6]

# Multiplier comparison: (label, low, high or None for a point, ours?)
MULTS = [
    ("Emulator: gov. consumption", 1.0, None, True),
    ("Emulator: income tax (implied)", 0.09, 0.25, True),
    ("OBR 2010: capital spending", 1.0, None, False),
    ("OBR 2010: current spending", 0.6, None, False),
    ("OBR 2010: VAT", 0.35, None, False),
    ("OBR 2010: income tax / NICs", 0.3, None, False),
    ("IMF (Batini et al. 2014): AEs, yr 1", 0.0, 1.0, False),
    ("Ramey (2019): purchases", 0.6, 1.0, False),
]


def main():
    df = pd.read_csv(HERE / "fig_anchored_data.csv", index_col=0)
    gdp_m = df["GDPM_model"]
    gdp_e = df["GDPM_efo"]
    model_g = [
        100 * (gdp_m.loc[p] / gdp_m.iloc[list(df.index).index(p) - 1] - 1)
        for p in OUT_PERIODS
    ]
    efo_g = [
        100 * (gdp_e.loc[p] / gdp_e.iloc[list(df.index).index(p) - 1] - 1)
        for p in OUT_PERIODS
    ]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # Left: grouped bars, outturn vs anchored forecast (EFO as reference line).
    x = np.arange(len(OUT_PERIODS))
    w = 0.32
    ax.bar(x - w / 2, model_g, w, color=BLUE, label="Anchored emulator")
    ax.bar(x + w / 2, ONS_OUTTURN, w, color=GREEN, label="ONS outturn")
    ax.plot(x - w / 2, efo_g, ls="none", marker="_", ms=11, mew=1.6,
            color=ORANGE, label="OBR EFO (Nov 2025)")
    for xi, (m, o) in enumerate(zip(model_g, ONS_OUTTURN)):
        ax.text(xi - w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, o + 0.02, f"{o:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(OUT_PERIODS, rotation=30, ha="right")
    ax.set_ylabel("Real GDP growth, % q/q")
    ax.set_ylim(0, 0.95)
    ax.set_title("Forecast vs subsequent outturn", fontsize=9)
    ax.legend(fontsize=6.8, loc="upper left", handlelength=1.4)

    # Right: multiplier dot/range chart.
    ys = np.arange(len(MULTS))[::-1]
    for y, (label, lo, hi, ours) in zip(ys, MULTS):
        c = BLUE if ours else GREY
        if hi is None:
            ax2.plot(lo, y, "o", color=c, ms=5)
        else:
            ax2.plot([lo, hi], [y, y], "-", color=c, lw=2.6,
                     solid_capstyle="round", alpha=0.85)
            ax2.plot([lo, hi], [y, y], "|", color=c, ms=7, mew=1.4)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([m[0] for m in MULTS], fontsize=7.3)
    ax2.set_xlim(-0.05, 1.3)
    ax2.set_xlabel("First-year GDP multiplier")
    ax2.set_title("Multipliers vs the literature", fontsize=9)
    ax2.grid(axis="y", alpha=0)

    fig.tight_layout()
    fig.savefig(HERE / "fig_outturn.pdf", bbox_inches="tight")

    pd.DataFrame(
        {
            "period": OUT_PERIODS,
            "model_qoq_pct": model_g,
            "efo_qoq_pct": efo_g,
            "ons_outturn_qoq_pct": ONS_OUTTURN,
        }
    ).to_csv(HERE / "fig_outturn_data.csv", index=False)
    print("wrote fig_outturn.pdf")
    print("model:", [f"{g:.3f}" for g in model_g])
    print("efo:  ", [f"{g:.3f}" for g in efo_g])


if __name__ == "__main__":
    main()
