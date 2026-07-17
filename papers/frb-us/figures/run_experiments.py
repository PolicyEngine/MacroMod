"""Run all simulation experiments for the FRB/US paper and emit figures + numbers.

Usage (from a venv with `frbus` and matplotlib installed):
    python run_experiments.py

Writes colorblind-safe PDF figures into this directory and a results.json
with every reported series, plus prints headline numbers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import frbus.symbolic as fsym
from frbus import Frbus, load_data

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/janansadeqian/us-frb-model")
XML = str(REPO / "vendor" / "pyfrbus_package" / "models" / "model.xml")
LONGBASE = str(REPO / "vendor" / "data_only_package" / "LONGBASE.TXT")

# Okabe-Ito colorblind-safe palette
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

START = pd.Period("2026Q1")
END20 = pd.Period("2030Q4")
END40 = pd.Period("2035Q4")
QLAB20 = [str(p) for p in pd.period_range(START, END20)]
QLAB40 = [str(p) for p in pd.period_range(START, END40)]

# --- capture Jacobian sparsity while the model compiles --------------------
_jac_stats: dict = {}
_orig_create = fsym.create_jacobian


def _create_and_record(exprs, rhs_endos):
    jac = _orig_create(exprs, rhs_endos)
    n = len(exprs)
    _jac_stats.update(nnz=len(jac), n=n, density=len(jac) / n**2)
    return jac


fsym.create_jacobian = _create_and_record

# --- baseline --------------------------------------------------------------
data = load_data(LONGBASE)
t0 = time.perf_counter()
model = Frbus(XML)
t_parse = time.perf_counter() - t0

data.loc[START:END40, "dfpdbt"] = 0
data.loc[START:END40, "dfpsrp"] = 1

t0 = time.perf_counter()
base = model.init_trac(START, END40, data)
t_trac = time.perf_counter() - t0  # includes one-off compilation
jac_full = dict(_jac_stats)

results: dict = {
    "jacobian": jac_full,
    "timing": {"parse_s": t_parse, "init_trac_and_compile_s": t_trac},
}


def dev(sim, ref, s=START, e=END40):
    return {
        "rff_pp": (sim.loc[s:e, "rff"] - ref.loc[s:e, "rff"]).tolist(),
        "xgdp_pct": (100 * (sim.loc[s:e, "xgdp"] / ref.loc[s:e, "xgdp"] - 1)).tolist(),
        "lur_pp": (sim.loc[s:e, "lur"] - ref.loc[s:e, "lur"]).tolist(),
        "picxfe_pp": (sim.loc[s:e, "picxfe"] - ref.loc[s:e, "picxfe"]).tolist(),
        "rg10_pp": (sim.loc[s:e, "rg10"] - ref.loc[s:e, "rg10"]).tolist(),
        "xgap2_pp": (sim.loc[s:e, "xgap2"] - ref.loc[s:e, "xgap2"]).tolist(),
    }


# --- (a,e) 100bp monetary shock, 40 quarters -------------------------------
shocked = base.copy()
shocked.loc[START, "rffintay_aerr"] += 1
t0 = time.perf_counter()
sim_mp = model.solve(START, END40, shocked)
t_solve40 = time.perf_counter() - t0
results["timing"]["solve_40q_s"] = t_solve40

t0 = time.perf_counter()
model.solve(START, END20, shocked)
results["timing"]["solve_20q_s"] = time.perf_counter() - t0

results["mp100"] = dev(sim_mp, base)

# --- (b,d) government purchases +1% GDP, inertial Taylor vs fixed rate -----
gmodel = Frbus(XML)
gmodel.exogenize(["egfe"])
FISC_END = pd.Period("2027Q4")

gdata = base.copy()
gdata.loc[START:FISC_END, "egfe"] = base.loc[START:FISC_END, "egfe"] + 0.01 * base.loc[
    START:FISC_END, "xgdp"
]
sim_g_tay = gmodel.solve(START, END40, gdata)
results["gov_taylor"] = dev(sim_g_tay, base)

gdata_fix = gdata.copy()
gdata_fix.loc[START:END40, "dmpintay"] = 0
gdata_fix.loc[START:END40, "dmpex"] = 1
gdata_fix.loc[START:END40, "rfffix"] = base.loc[START:END40, "rff"]
sim_g_fix = gmodel.solve(START, END40, gdata_fix)
results["gov_fixed"] = dev(sim_g_fix, base)

# --- (c) personal tax cut, ex-ante 1% of GDP -------------------------------
tmodel = Frbus(XML)
tmodel.exogenize(["trp"])
tdata = base.copy()
dtrp = 0.01 * base.loc[START:FISC_END, "xgdpn"] / (
    base.loc[START:FISC_END, "ypn"] - base.loc[START:FISC_END, "gtn"]
)
tdata.loc[START:FISC_END, "trp"] = base.loc[START:FISC_END, "trp"] - dtrp
sim_tax = tmodel.solve(START, END40, tdata)
results["taxcut"] = dev(sim_tax, base)
results["taxcut"]["dtrp_pp"] = (100 * dtrp).tolist()
# realised revenue change as share of GDP (ex-post)
results["taxcut"]["drev_pctgdp"] = (
    100
    * (sim_tax.loc[START:END40, "tpn"] - base.loc[START:END40, "tpn"])
    / base.loc[START:END40, "xgdpn"]
).tolist()

# --- (f) cost-push price shock: +1pp to core PCE inflation for 4 quarters --
cdata = base.copy()
CP_END = pd.Period("2026Q4")
cdata.loc[START:CP_END, "picxfe_aerr"] += 1
sim_cp = model.solve(START, END40, cdata)
results["costpush"] = dev(sim_cp, base)

results["quarters"] = QLAB40

with open(HERE / "results.json", "w") as fh:
    json.dump(results, fh, indent=1)

# --- figures ---------------------------------------------------------------
q = pd.period_range(START, END40).to_timestamp()


def panelfig(fname, series, labels, panels, title=None):
    """series: list of result dicts; labels: legend labels; panels: (key, ylabel, ptitle)."""
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2), constrained_layout=True)
    for ax, (key, ylab, ptitle) in zip(axes.ravel(), panels):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for i, (s, lab) in enumerate(zip(series, labels)):
            ax.plot(q[: len(s[key])], s[key], color=OI[i], label=lab)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(ptitle, fontsize=9)
        ax.set_ylabel(ylab)
    if len(series) > 1:
        axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(HERE / fname)
    plt.close(fig)


panels = [
    ("rff_pp", "pp dev.", "Federal funds rate (rff)"),
    ("xgdp_pct", "% dev.", "Real GDP (xgdp)"),
    ("lur_pp", "pp dev.", "Unemployment rate (lur)"),
    ("picxfe_pp", "pp dev.", "Core PCE inflation (picxfe)"),
]

panelfig("fig_mp100.pdf", [results["mp100"]], ["100bp shock"], panels)
panelfig("fig_costpush.pdf", [results["costpush"]], ["cost-push shock"], panels)
panelfig(
    "fig_fiscal_rules.pdf",
    [results["gov_taylor"], results["gov_fixed"]],
    ["Inertial Taylor rule", "Fixed funds rate"],
    panels,
)
panelfig(
    "fig_fiscal_vs_tax.pdf",
    [results["gov_taylor"], results["taxcut"]],
    ["Purchases +1% of GDP", "Personal tax cut, 1% of GDP"],
    panels,
)

# --- headline printout -----------------------------------------------------
def yr1(key):
    x = results[key]["xgdp_pct"][:4]
    return sum(x) / 4


print("Jacobian:", jac_full)
print("Timing:", results["timing"])
print(f"Gov purchases multiplier: impact={results['gov_taylor']['xgdp_pct'][0]:.3f}, "
      f"year-1 avg={yr1('gov_taylor'):.3f}")
print(f"Gov purchases FIXED rate: impact={results['gov_fixed']['xgdp_pct'][0]:.3f}, "
      f"year-1 avg={yr1('gov_fixed'):.3f}, "
      f"year-2 avg={sum(results['gov_fixed']['xgdp_pct'][4:8])/4:.3f}")
print(f"Tax cut: impact={results['taxcut']['xgdp_pct'][0]:.3f}, year-1 avg={yr1('taxcut'):.3f}, "
      f"year-2 avg={sum(results['taxcut']['xgdp_pct'][4:8])/4:.3f}")
print("Tax cut peak xgdp:", max(results["taxcut"]["xgdp_pct"]))
print("gov_taylor year-2 avg:", sum(results["gov_taylor"]["xgdp_pct"][4:8]) / 4)
print("mp100 trough:", min(results["mp100"]["xgdp_pct"]),
      "at", results["quarters"][results["mp100"]["xgdp_pct"].index(min(results["mp100"]["xgdp_pct"]))])
print("mp100 40q final xgdp:", results["mp100"]["xgdp_pct"][-1],
      "lur:", results["mp100"]["lur_pp"][-1])
