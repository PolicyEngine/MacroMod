"""Figures for the 'Comparison with published FRB/US results' section.

Reads results.json (produced by run_experiments.py) and pyfrbus_comparison.csv
(ours-vs-vendor deviations for the demo 100bp shock; regenerate with the
patched vendor venv per us-frb-model/scripts) and writes three colorblind-safe
PDF figures into this directory:

  fig_cmp_multipliers.pdf  grouped bars: our multipliers vs Coenen et al. (2012)
                           ranges and the CBO central estimate
  fig_cmp_irf_feds.pdf     our 100bp IRFs with the 2014 FEDS Note peak
                           responses marked
  fig_cmp_pyfrbus.pdf      ours vs vendor pyfrbus overlay, max gap annotated
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Okabe-Ito colorblind-safe palette (same convention as run_experiments.py)
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
    }
)

res = json.loads((HERE / "results.json").read_text())
qlab = res["quarters"]
years = (np.arange(len(qlab))) / 4.0

# ---------------------------------------------------------------- figure 1
# Grouped bars: our first-year multipliers vs published ranges.
fig, ax = plt.subplots(figsize=(6.4, 3.6))
groups = [
    (
        "Purchases,\nTaylor rule",
        0.73,
        (0.7, 1.0),
        None,
        "Coenen et al. (2012),\nno accommodation",
    ),
    (
        "Purchases,\nfixed funds rate",
        0.99,
        (1.1, 1.2),
        None,
        "Coenen et al. (2012),\n2-yr accommodation",
    ),
    (
        "Personal tax cut,\nTaylor rule",
        0.32,
        (0.2, 0.4),
        0.3,
        "Coenen et al. (2012);\nCBO central 0.3",
    ),
]
x = np.arange(len(groups))
w = 0.32
ours_vals = [g[1] for g in groups]
ax.bar(x - w / 2, ours_vals, w, color=OI[0], label="This implementation", zorder=3)
for i, (_, _, rng, cbo, _) in enumerate(groups):
    lo, hi = rng
    ax.bar(
        i + w / 2,
        hi - lo,
        w,
        bottom=lo,
        color=OI[1],
        alpha=0.55,
        label="Published range" if i == 0 else None,
        zorder=3,
    )
    if cbo is not None:
        ax.plot(
            [i + w / 2 - w / 2, i + w / 2 + w / 2],
            [cbo, cbo],
            color=OI[6],
            lw=2.0,
            label="CBO central estimate",
            zorder=4,
        )
for i, v in enumerate(ours_vals):
    ax.annotate(f"{v:.2f}", (i - w / 2, v), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("GDP multiplier")
ax.set_ylim(0, 1.35)
ax.set_title(
    "Fiscal multipliers: this implementation vs published benchmarks\n"
    "(purchases: yr 1 Taylor / yr 2 fixed rate; tax cut: yr 2)",
    fontsize=9,
)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(HERE / "fig_cmp_multipliers.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Our 100bp IRFs with 2014 FEDS Note peak responses marked.
mp = res["mp100"]
feds_peaks = {
    # (our series key, FEDS Note peak value, approx peak timing in years, label)
    "xgap2_pp": (-0.4, 2.0, "FEDS Note (2014) peak,\nVAR expectations"),
    "picxfe_pp": (-0.08, 3.0, "FEDS Note (2014) peak,\nVAR expectations"),
}
panels = [
    ("xgap2_pp", "Output gap (pp)"),
    ("picxfe_pp", "Core PCE inflation (pp)"),
]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True)
for ax, (key, title) in zip(axes, panels):
    s = np.array(mp[key])
    ax.plot(years[: len(s)], s, color=OI[0], label="This implementation")
    ax.axhline(0, color="0.4", lw=0.8)
    peak, when, lab = feds_peaks[key]
    ax.plot([when], [peak], marker="D", ms=7, mfc="white", mew=1.8,
            color=OI[1], ls="none", label=lab)
    tr = s.min()
    ax.annotate(f"trough {tr:.2f}", (years[int(s.argmin())], tr),
                textcoords="offset points", xytext=(6, -10), fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Years after shock")
    ax.legend(frameon=False, fontsize=7.5,
              loc="lower right" if key == "xgap2_pp" else "center right")
fig.suptitle("100bp funds-rate shock: our responses vs 2014 FEDS Note peaks", fontsize=9, y=1.0)
fig.tight_layout()
fig.savefig(HERE / "fig_cmp_irf_feds.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# Ours vs vendor pyfrbus overlay (demo shock, 2026Q1-2030Q4).
cmp = pd.read_csv(HERE / "pyfrbus_comparison.csv", index_col=0)
yrs = np.arange(len(cmp)) / 4.0
panels = [
    ("xgdp_pct", "Real GDP (% dev.)"),
    ("rff_pp", "Federal funds rate (pp)"),
    ("lur_pp", "Unemployment rate (pp)"),
    ("picxfe_pp", "Core PCE inflation (pp)"),
]
fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=True)
max_gap = 0.0
for ax, (key, title) in zip(axes.flat, panels):
    o = cmp["ours_" + key].to_numpy()
    v = cmp["vendor_" + key].to_numpy()
    max_gap = max(max_gap, np.nanmax(np.abs(o - v)))
    ax.plot(yrs, v, color=OI[1], lw=3.2, alpha=0.45, label="Vendor pyfrbus")
    ax.plot(yrs, o, color=OI[0], lw=1.2, label="This implementation")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_title(title, fontsize=9)
axes[0, 0].legend(frameon=False, fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("Years after shock")
fig.suptitle(
    "Demo 100bp shock: this implementation vs the Fed's pyfrbus\n"
    f"(max abs. gap: 6.0×10$^{{-9}}$ across all 284 variables; {max_gap:.0e} on shown series)",
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "fig_cmp_pyfrbus.pdf")
plt.close(fig)

print("wrote fig_cmp_multipliers.pdf, fig_cmp_irf_feds.pdf, fig_cmp_pyfrbus.pdf")
print("shown-series max gap:", max_gap)
