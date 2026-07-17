"""Generate the ours-vs-publication comparison figure (fig_comparison.pdf).

Compares the replication's 1-year global FEVD shares for UK GDP and UK CPI
against the values published in Brignone & Piffer (2025, BoE Macro Technical
Paper No. 3):

  - "CI fast config": a FRESH fixed-seed run of the test suite's fast
    configuration (600 NIW posterior draws, sample seed 1, identification
    seed 2, unweighted; identical to tests/conftest.py in the read-only
    repo clone) -- recomputed here, not copied.
  - "Production (weighted)": the repo's production artifact
    (results/summary.md): 10,000 draws, 751 accepted, importance-weighted
    (ESS 350.3) -- 40.9% GDP / 50.1% CPI.
  - "Paper (~)": the approximate shares stated in the publication
    (~40% GDP, ~50% CPI at business-cycle horizons).

Also recomputes the unrestricted oil-price impact response to a world supply
shock (paper: positive, Fig. 2; no numeric IRF values are published or
stored in the repo's validation data, so no IRF overlay is drawn) and writes
all numbers to comparison_numbers.json.

Palette: Okabe-Ito (colorblind-safe), consistent with make_figures.py.
Usage: python make_comparison.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = "/Users/janansadeqian/boe-var-model"
sys.path.insert(0, os.path.join(REPO, "src"))
HERE = os.path.dirname(os.path.abspath(__file__))

import pandas as pd  # noqa: E402

from boe_var import analysis  # noqa: E402
from boe_var.analysis import WORLD_SHOCKS  # noqa: E402
from boe_var.bvar import BVAR  # noqa: E402
from boe_var.data import load_data  # noqa: E402
from boe_var.identification import identify  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

# Okabe-Ito, fixed assignment by run identity (never cycled).
C_FAST = "#0072B2"       # blue      - CI fast config (fresh run)
C_PROD = "#E69F00"       # orange    - production weighted artifact
C_PAPER = "#999999"      # grey      - published paper values

I_CPI, I_GDP = 5, 7
H_1YR = 4

# Fast fixed-seed configuration, identical to tests/conftest.py.
N_DRAWS, SAMPLE_SEED, IDENT_SEED, LAGS = 600, 1, 2, 4

# Production artifact (results/summary.md): 10,000 draws, 751 accepted,
# importance-weighted, ESS 350.3.
PROD = {"gdp": 40.9, "cpi": 50.1}
# Published values (approximate, as stated in the paper).
PAPER = {"gdp": 40.0, "cpi": 50.0}


def main():
    df = load_data()
    df = df.loc[(df.index >= pd.Period("1992Q1", "Q"))
                & (df.index <= pd.Period("2023Q2", "Q"))]
    y = df.to_numpy(dtype=float)
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    dummies = np.column_stack(
        [(df.index == q).astype(float) for q in quarters])

    model = BVAR(y, lags=LAGS, dummies=dummies, lam=0.2, mu=1.0, theta=1.0)
    draws = model.sample_posterior(N_DRAWS, seed=SAMPLE_SEED)
    accepted = identify(draws, rng=np.random.default_rng(IDENT_SEED),
                        compute_weights=False)
    pairs = [(d, B) for d, B, _ in accepted]
    print(f"fast config: accepted {len(pairs)}/{N_DRAWS}")

    bands = analysis.fevd_bands(pairs, horizons=21)
    med = bands["median"]
    med = med / med.sum(axis=1, keepdims=True)
    fast = {
        "gdp": 100.0 * float(med[I_GDP, WORLD_SHOCKS, H_1YR].sum()),
        "cpi": 100.0 * float(med[I_CPI, WORLD_SHOCKS, H_1YR].sum()),
    }
    irf = analysis.irf_bands(pairs, horizons=21)
    oil_impact = float(irf["median"][2, 2, 0])  # oil price, world supply, h=0
    print(f"fast FEVD global shares: GDP {fast['gdp']:.1f}%, "
          f"CPI {fast['cpi']:.1f}%; oil impact {oil_impact:+.2f}")

    # ---- grouped bar chart -------------------------------------------------
    groups = ["UK GDP", "UK CPI"]
    series = [
        (f"Replication, CI fast config ({len(pairs)} accepted / "
         f"{N_DRAWS} draws, unweighted)", C_FAST,
         [fast["gdp"], fast["cpi"]], None),
        ("Replication, production (751 / 10,000 draws, weighted)", C_PROD,
         [PROD["gdp"], PROD["cpi"]], None),
        ("Brignone & Piffer (2025), published (approx.)", C_PAPER,
         [PAPER["gdp"], PAPER["cpi"]], "//"),
    ]
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for k, (label, color, vals, hatch) in enumerate(series):
        pos = x + (k - 1) * (w + 0.02)
        bars = ax.bar(pos, vals, width=w, color=color, hatch=hatch,
                      edgecolor="white", linewidth=0.8, label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5,
                        color="#1a1a2e")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Global share of 1-yr FEV (%)")
    ax.set_ylim(0, 62)
    ax.axhspan(30, 60, color="#0072B2", alpha=0.06, zorder=0)
    ax.text(0.5, 57.2, "calibration band [30, 60]", fontsize=6.5,
            color="#555555", ha="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
              frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_comparison.pdf"))
    plt.close(fig)

    out = {
        "fast_config": {"n_draws": N_DRAWS, "sample_seed": SAMPLE_SEED,
                        "ident_seed": IDENT_SEED, "accepted": len(pairs),
                        "gdp_global_share_pct": fast["gdp"],
                        "cpi_global_share_pct": fast["cpi"],
                        "oil_impact_world_supply": oil_impact},
        "production_artifact": PROD,
        "paper": PAPER,
    }
    with open(os.path.join(HERE, "comparison_numbers.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
