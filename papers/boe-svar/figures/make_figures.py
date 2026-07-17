"""Generate the paper's figures from the boe-var-model replication code.

Runs the read-only clone at REPO with its packaged dataset, fixed seeds, and
writes colorblind-safe (Okabe-Ito palette) PDF figures into this directory:

  fig_irf.pdf   - IRF panel, 3 key shocks x 4 variables, 68/90% bands
  fig_fevd.pdf  - stacked FEVD decomposition for UK GDP and UK CPI
  fig_hd.pdf    - historical decomposition of YoY UK CPI inflation, 2016-2023
  fig_fan.pdf   - fan-chart forecast from the 2024Q2 edge vs ONS outturns

Draw counts (documented in the paper): 3000 NIW posterior draws for the
replication-sample figures and 3000 for the 2024Q2-edge forecast run;
identification keeps one Q per posterior draw (Arias et al. 2018), sign
restrictions by accept/reject. Importance weights are NOT computed here
(compute_weights=False): figures are unweighted, which Section 6/validation
shows moves 1-year FEVD shares by only a few points relative to the
importance-weighted benchmark.

Usage: python make_figures.py  (needs numpy/scipy/pandas/matplotlib)
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

from boe_var import analysis, forecast  # noqa: E402
from boe_var.bvar import BVAR  # noqa: E402
from boe_var.data import load_data  # noqa: E402
from boe_var.identification import identify  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

# Okabe-Ito colorblind-safe palette; hue assigned to shock identity in fixed
# order, never cycled.
OI = {
    "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00",
    "purple": "#CC79A7", "grey": "#999999",
}
SHOCK_COLORS = {
    "World demand": OI["blue"], "World energy": OI["vermilion"],
    "World supply": OI["green"], "UK demand": OI["skyblue"],
    "UK supply": OI["orange"], "UK mon. pol.": OI["purple"],
    "Unidentified": OI["grey"], "Covid": "#000000",
}
LINE = "#1a1a2e"
BAND68 = "#0072B2"
BAND90 = "#0072B2"

SEED = 20250717
N_DRAWS = 3000
HORIZONS = 21
FC_HORIZONS = 13

VN = analysis.VARIABLE_NAMES
SN = analysis.SHOCK_NAMES
I_CPI, I_GDP = 5, 7


def covid_dummies(index):
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    D = np.zeros((len(index), len(quarters)))
    for j, q in enumerate(quarters):
        D[:, j] = (index == q).astype(float)
    return D


def residuals_for(draw, y, dummies, lags):
    T, k = y.shape
    rows = []
    for t in range(lags, T):
        xlags = np.concatenate([y[t - l] for l in range(1, lags + 1)])
        rows.append(np.concatenate([xlags, [1.0], dummies[t]]))
    X = np.asarray(rows)
    return y[lags:] - X @ np.asarray(draw.Pi).T


def main():
    rng = np.random.default_rng(SEED)
    df_full = load_data()
    df_full = df_full.loc[df_full.index >= pd.Period("1992Q1", "Q")]
    assert df_full.index[-1] == pd.Period("2024Q2", "Q")
    df_est = df_full.loc[df_full.index <= pd.Period("2023Q2", "Q")]
    y_est = df_est.to_numpy(float)
    dummies_est = covid_dummies(df_est.index)

    # ---------------- estimation-sample run (figures 1-3) ----------------
    model = BVAR(y_est, lags=4, dummies=dummies_est, lam=0.2, mu=1.0)
    draws = model.sample_posterior(N_DRAWS, seed=SEED)
    triples = identify(draws, rng=rng, compute_weights=False)
    pairs = [(d, B) for d, B, _ in triples]
    print(f"replication run: accepted {len(pairs)}/{N_DRAWS}")

    irf_b = analysis.irf_bands(pairs, horizons=HORIZONS)
    fevd_b = analysis.fevd_bands(pairs, horizons=HORIZONS)

    # ---- fig_irf: 3 shocks x 4 variables ----
    shocks = [(0, "World demand"), (1, "World energy"), (6, "UK mon. pol.")]
    vars_ = [(0, "World GDP"), (2, "Oil price"), (5, "UK CPI"), (7, "UK GDP")]
    x = np.arange(HORIZONS)
    fig, axes = plt.subplots(3, 4, figsize=(9.2, 5.6), sharex=True)
    for r, (j, sname) in enumerate(shocks):
        for c, (i, vname) in enumerate(vars_):
            ax = axes[r][c]
            ax.fill_between(x, irf_b["lo90"][i, j], irf_b["hi90"][i, j],
                            color=BAND90, alpha=0.15, linewidth=0)
            ax.fill_between(x, irf_b["lo68"][i, j], irf_b["hi68"][i, j],
                            color=BAND68, alpha=0.30, linewidth=0)
            ax.plot(x, irf_b["median"][i, j], color=LINE, lw=1.4)
            ax.axhline(0, color="#666666", lw=0.6)
            if r == 0:
                ax.set_title(vname)
            if c == 0:
                ax.set_ylabel(sname)
            if r == 2:
                ax.set_xlabel("Quarters")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_irf.pdf"))
    plt.close(fig)

    # ---- fig_fevd: stacked shares for UK GDP and UK CPI ----
    med = fevd_b["median"]
    med = med / med.sum(axis=1, keepdims=True)
    groups = [("World demand", 0), ("World energy", 1), ("World supply", 2),
              ("UK demand", 4), ("UK supply", 5), ("UK mon. pol.", 6)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), sharey=True)
    for ax, i, title in ((axes[0], I_GDP, "UK GDP"), (axes[1], I_CPI, "UK CPI")):
        series = [med[i, j] for _, j in groups]
        series.append(med[i, 3] + med[i, 7])  # unidentified combined
        labels = [g for g, _ in groups] + ["Unidentified"]
        colors = [SHOCK_COLORS[lab] for lab in labels]
        ax.stackplot(x, series, labels=labels, colors=colors,
                     edgecolor="white", linewidth=0.4)
        ax.set_xlim(0, HORIZONS - 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Quarters")
    axes[0].set_ylabel("Share of forecast-error variance")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_fevd.pdf"))
    plt.close(fig)

    # ---- fig_hd: historical decomposition of YoY UK CPI, 2016 onwards ----
    dummies_resid = dummies_est[4:]  # rows lags..T-1
    contrib, covid = analysis.hd_median(
        pairs, lambda d: residuals_for(d, y_est, dummies_est, 4),
        dummies=dummies_resid)
    idx = df_est.index[4:]  # residual sample index
    # YoY: t minus t-4 of the level contributions
    c_yoy = contrib[4:] - contrib[:-4]
    covid_yoy = covid[4:] - covid[:-4]
    idx_yoy = idx[4:]
    sel = idx_yoy >= pd.Period("2016Q1", "Q")
    c_yoy, covid_yoy, idx_yoy = c_yoy[sel], covid_yoy[sel], idx_yoy[sel]
    T = c_yoy.shape[0]
    xb = np.arange(T)
    bars = [(lab, SHOCK_COLORS[lab], c_yoy[:, I_CPI, j])
            for lab, j in groups]
    bars.append(("Unidentified", SHOCK_COLORS["Unidentified"],
                 c_yoy[:, I_CPI, [3, 7]].sum(axis=1)))
    bars.append(("Covid", SHOCK_COLORS["Covid"], covid_yoy[:, I_CPI]))
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    pos = np.zeros(T)
    neg = np.zeros(T)
    for lab, col, arr in bars:
        base = np.where(arr >= 0, pos, neg)
        ax.bar(xb, arr, bottom=base, width=0.8, color=col, label=lab)
        pos += np.clip(arr, 0, None)
        neg += np.clip(arr, None, 0)
    total = sum(arr for _, _, arr in bars)
    ax.plot(xb, total, color="k", lw=1.3, label="Data less deterministic")
    ax.axhline(0, color="#666666", lw=0.6)
    step = max(1, T // 10)
    ax.set_xticks(xb[::step])
    ax.set_xticklabels([str(q) for q in idx_yoy[::step]], rotation=45,
                       ha="right")
    ax.set_ylabel("pp deviation, YoY UK CPI inflation")
    ax.legend(loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_hd.pdf"))
    plt.close(fig)

    # ---------------- 2024Q2-edge forecast run (figure 4) ----------------
    rng2 = np.random.default_rng(SEED + 1)
    model2 = BVAR(y_est, lags=4, dummies=dummies_est, lam=0.2, mu=1.0)
    draws2 = model2.sample_posterior(N_DRAWS, seed=SEED + 1)
    triples2 = identify(draws2, rng=rng2, compute_weights=False)
    pairs2 = [(d, B) for d, B, _ in triples2]
    print(f"forecast run: accepted {len(pairs2)}/{N_DRAWS}")

    y_full = df_full.to_numpy(float)
    fb = forecast.forecast_bands(pairs2, y_full, horizons=FC_HORIZONS,
                                 n_paths=5, rng=rng2)
    fut_idx = pd.period_range("2024Q3", periods=FC_HORIZONS, freq="Q")

    # YoY paths: prepend last 4 observed rows so YoY is defined from 2024Q3
    hist_tail = y_full[-4:]

    def to_yoy(levels):
        return forecast.yoy(np.vstack([hist_tail, levels]))

    yoy_bands = {k2: to_yoy(v) for k2, v in fb.items()}
    hist_yoy = forecast.yoy(y_full)
    hist_idx = df_full.index[4:]
    hsel = hist_idx >= pd.Period("2022Q1", "Q")

    # ONS outturns (YoY, per cent): quarterly-average CPI inflation and
    # quarterly real GDP growth on a year earlier (see paper for sources).
    outturn_q = pd.period_range("2024Q3", "2025Q3", freq="Q")
    outturn_cpi = [2.0, 2.5, 2.8, 3.5, 3.8]
    outturn_gdp = [1.0, 1.5, 1.3, 1.4, 1.3]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for ax, i, title, pts in (
        (axes[0], I_GDP, "UK GDP growth (YoY, %)", outturn_gdp),
        (axes[1], I_CPI, "UK CPI inflation (YoY, %)", outturn_cpi),
    ):
        hx = np.arange(hsel.sum())
        fxs = hx[-1] + 1 + np.arange(FC_HORIZONS)
        ax.plot(hx, hist_yoy[hsel][:, i], color=LINE, lw=1.4, label="Data")
        ax.fill_between(fxs, yoy_bands["lo90"][:, i], yoy_bands["hi90"][:, i],
                        color=BAND90, alpha=0.15, linewidth=0,
                        label="90% band")
        ax.fill_between(fxs, yoy_bands["lo68"][:, i], yoy_bands["hi68"][:, i],
                        color=BAND68, alpha=0.30, linewidth=0,
                        label="68% band")
        ax.plot(fxs, yoy_bands["median"][:, i], color=OI["blue"], lw=1.4,
                ls="--", label="Median forecast")
        opos = [fxs[list(fut_idx).index(q)] for q in outturn_q]
        ax.scatter(opos, pts, s=22, color=OI["vermilion"], zorder=5,
                   label="ONS outturn")
        ax.axvline(hx[-1] + 0.5, color="#999999", lw=0.8, ls=":")
        all_idx = list(df_full.index[4:][hsel]) + list(fut_idx)
        ticks = np.arange(0, len(all_idx), 4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(all_idx[t]) for t in ticks], rotation=45,
                           ha="right")
        ax.set_title(title)
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_fan.pdf"))
    plt.close(fig)

    # ---- numbers for the paper's tables ----
    med_yoy = yoy_bands["median"]
    lo68, hi68 = yoy_bands["lo68"], yoy_bands["hi68"]
    lo90, hi90 = yoy_bands["lo90"], yoy_bands["hi90"]
    table = {}
    for qi, q in enumerate(fut_idx):
        table[str(q)] = {
            "gdp": [float(med_yoy[qi, I_GDP]), float(lo68[qi, I_GDP]),
                    float(hi68[qi, I_GDP]), float(lo90[qi, I_GDP]),
                    float(hi90[qi, I_GDP])],
            "cpi": [float(med_yoy[qi, I_CPI]), float(lo68[qi, I_CPI]),
                    float(hi68[qi, I_CPI]), float(lo90[qi, I_CPI]),
                    float(hi90[qi, I_CPI])],
        }
    out = {
        "accepted_replication": len(pairs),
        "accepted_forecast": len(pairs2),
        "n_draws": N_DRAWS,
        "seed": SEED,
        "forecast_table": table,
    }
    with open(os.path.join(HERE, "figure_numbers.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
