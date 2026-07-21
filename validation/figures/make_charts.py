#!/usr/bin/env python3
"""Regenerate the inline SVG charts in validation/index.html from the papers' sources.

Run:  python3 validation/figures/make_charts.py          # rewrite validation/index.html
      python3 validation/figures/make_charts.py --check  # exit 1 if the page is stale

Why hand-emitted SVG rather than matplotlib: the charts must stay *inline* in the
HTML (the site ships no third-party JS and no external assets) and must inherit the
site's CSS custom properties so they retheme for light/dark. A matplotlib export
bakes in literal colours and would need post-processing to strip them; emitting the
markup directly keeps every fill and stroke on a `vc-*` class defined in style.css.

Every plotted number is either read from a committed data file under papers/*/figures/
or transcribed, with a per-value source pointer, in chart_data.json next to this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAGE = ROOT / "validation" / "index.html"


# ---------------------------------------------------------------- helpers

def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def n(x: float) -> str:
    """Format a coordinate the way the committed markup does: 1dp, no trailing .0 loss."""
    return f"{x:.1f}"


# ---------------------------------------------------------------- data loading

def load_obr_anchored():
    """Quarterly % deviation of the anchored emulator from the March-2026 EFO.

    Source: papers/obr-macro/figures/fig_anchored_data.csv (model vs EFO levels).
    """
    path = ROOT / "papers" / "obr-macro" / "figures" / "fig_anchored_data.csv"
    quarters, gdp, cons = [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            quarters.append(row[""])
            gdp.append((float(row["GDPM_model"]) / float(row["GDPM_efo"]) - 1) * 100)
            cons.append((float(row["CONS_model"]) / float(row["CONS_efo"]) - 1) * 100)
    return quarters, gdp, cons


def load_svar():
    """Global-shock FEVD shares at the 1-year horizon, ours vs the paper."""
    d = json.loads((ROOT / "papers" / "boe-svar" / "figures" / "comparison_numbers.json").read_text())
    return d["production_artifact"], d["paper"]


def load_levels(name, cols):
    """Quarterly levels from a papers/obr-macro/figures CSV, keyed by column."""
    path = ROOT / "papers" / "obr-macro" / "figures" / name
    quarters, out = [], {c: [] for c in cols}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            quarters.append(row[""])
            for c in cols:
                out[c].append(float(row[c]))
    return quarters, out


def load_outturn():
    """Quarterly %q/q GDP growth: emulator vs EFO vintage vs ONS outturn."""
    path = ROOT / "papers" / "obr-macro" / "figures" / "fig_outturn_data.csv"
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append((row["period"], float(row["model_qoq_pct"]),
                         float(row["efo_qoq_pct"]), float(row["ons_outturn_qoq_pct"])))
    return rows


def load_svar_fan():
    """Median/68% forecast paths from the frozen 2024Q2 edge, plus ONS outturns.

    Medians and bands: papers/boe-svar/figures/figure_numbers.json (forecast_table,
    entries [median, lo68, hi68, lo90, hi90]). Outturns: the committed arrays in
    papers/boe-svar/figures/make_figures.py.
    """
    d = json.loads((ROOT / "papers" / "boe-svar" / "figures" / "figure_numbers.json").read_text())
    ft = d["forecast_table"]
    quarters = list(ft)
    src = (ROOT / "papers" / "boe-svar" / "figures" / "make_figures.py").read_text()
    outturns = {}
    for var in ("gdp", "cpi"):
        m = re.search(rf"outturn_{var}\s*=\s*\[([^\]]+)\]", src)
        outturns[var] = [float(v) for v in m.group(1).split(",")]
    series = {var: {"median": [ft[q][var][0] for q in quarters],
                    "lo68": [ft[q][var][1] for q in quarters],
                    "hi68": [ft[q][var][2] for q in quarters]}
              for var in ("gdp", "cpi")}
    return quarters, series, outturns


def load_transcribed():
    return json.loads((HERE / "chart_data.json").read_text())


# ---------------------------------------------------------------- chart builders

def svg_open(view_w, view_h, chart_id, title, desc):
    tid, did = f"{chart_id}-t", f"{chart_id}-d"
    return [
        f'<svg class="vchart" data-chart="{chart_id}" viewBox="0 0 {view_w} {view_h}" '
        f'role="img" aria-labelledby="{tid} {did}">',
        f'<title id="{tid}">{esc(title)}</title>',
        f'<desc id="{did}">{esc(desc)}</desc>',
    ]


def chart_obr_anchored():
    quarters, gdp, cons = load_obr_anchored()
    W, H = 760, 326
    x0, x1 = 58.0, 724.0
    zero_y = 175.0
    per_unit = 55.5 / 0.3          # px per percentage point
    step = (x1 - x0) / (len(quarters) - 1)

    xs = [x0 + i * step for i in range(len(quarters))]
    def ymap(v):
        return zero_y - v * per_unit

    mape_g = sum(abs(v) for v in gdp) / len(gdp)
    mape_c = sum(abs(v) for v in cons) / len(cons)

    desc = (
        f"Line chart. Quarterly percentage deviation of the anchored emulator from the "
        f"published March 2026 EFO, {quarters[0]} to {quarters[-1]}. Real GDP ranges from "
        f"{min(gdp):+.2f}% to {max(gdp):+.2f}% (mean absolute deviation {mape_g:.2f}%); "
        f"consumption from {min(cons):+.2f}% to {max(cons):+.2f}% (mean absolute deviation "
        f"{mape_c:.2f}%). Both series stay well inside the plus or minus 1% band at which "
        f"continuous integration hard-fails the build, which is off the top and bottom of "
        f"this frame."
    )
    out = svg_open(W, H, "obr-anchored",
                   "obr-macro: anchored baseline vs March 2026 EFO, quarterly deviation",
                   desc)

    # legend
    out.append('<line class="vc-s1" x1="58" y1="26" x2="84" y2="26"/><circle class="vc-s1-dot" cx="71" cy="26" r="3"/>')
    out.append(f'<text class="vc-lab" x="92" y="30">real GDP (peak {max(gdp):+.2f}%)</text>')
    out.append('<line class="vc-s2" x1="288" y1="26" x2="314" y2="26"/><circle class="vc-s2-dot" cx="301" cy="26" r="3"/>')
    out.append(f'<text class="vc-lab vc-lab2" x="322" y="30">consumption (peak {max(cons):+.2f}%)</text>')

    # y grid
    for tick in (-0.6, -0.3, 0.0, 0.3, 0.6):
        y = ymap(tick)
        cls = "vc-axis" if tick == 0 else "vc-grid"
        label = "0" if tick == 0 else f"{tick:+.1f}%".replace("+0.", "+0.").replace("-0.", "-0.")
        out.append(f'<line class="{cls}" x1="{x0:.0f}" y1="{n(y)}" x2="{x1:.0f}" y2="{n(y)}"/>')
        out.append(f'<text class="vc-tick" x="48" y="{n(y + 4)}" text-anchor="end">{label}</text>')

    # x ticks: first, and each subsequent Q1, plus the last point
    tick_idx = [i for i, q in enumerate(quarters) if q.endswith("Q1")]
    if len(quarters) - 1 not in tick_idx:
        tick_idx.append(len(quarters) - 1)
    for i in tick_idx:
        out.append(f'<text class="vc-tick" x="{n(xs[i])}" y="306" text-anchor="middle">{quarters[i]}</text>')

    for cls, series in (("vc-s1", gdp), ("vc-s2", cons)):
        pts = " L".join(f"{n(xs[i])} {n(ymap(v))}" for i, v in enumerate(series))
        out.append(f'<path class="{cls}" d="M{pts}"/>')
        for i, v in enumerate(series):
            out.append(f'<circle class="{cls}-dot" cx="{n(xs[i])}" cy="{n(ymap(v))}" r="3"/>')

    out.append("</svg>")
    return "\n".join(out)


def grouped_bars(chart_id, title, desc, groups, y_max, y_step, fmt, unit_suffix=""):
    """Two side-by-side bars per group, shared layout for the OBR and SVAR charts."""
    W, H = 760, 304
    base_y, top_y = 258.0, 26.0
    scale = (base_y - top_y) / y_max
    bar_w = 79.2
    centres = (229.0, 559.0)

    out = svg_open(W, H, chart_id, title, desc)

    t = 0.0
    while t <= y_max + 1e-9:
        y = base_y - t * scale
        cls = "vc-axis" if t == 0 else "vc-grid"
        out.append(f'<line class="{cls}" x1="64" y1="{n(y)}" x2="724" y2="{n(y)}"/>')
        out.append(f'<text class="vc-tick" x="54" y="{n(y + 4)}" text-anchor="end">{t:g}{unit_suffix}</text>')
        t += y_step

    for centre, group in zip(centres, groups):
        for k, bar in enumerate(group["bars"]):
            x = centre - 87.1 + k * 91.1
            h = bar["value"] * scale
            y = base_y - h
            out.append(f'<rect class="vc-b{bar["series"]}" x="{n(x)}" y="{n(y)}" width="{bar_w}" height="{n(h)}"/>')
            cx = x + 39.6
            out.append(f'<text class="vc-val" x="{n(cx)}" y="{n(y - 8)}" text-anchor="middle">{fmt(bar["value"])}</text>')
            out.append(f'<text class="vc-tick" x="{n(cx)}" y="276" text-anchor="middle">{esc(bar["name"])}</text>')
        out.append(f'<text class="vc-lab" x="{n(centre - 0.0)}" y="296" text-anchor="middle">{esc(group["label"])}</text>')

    out.append("</svg>")
    return "\n".join(out)


def chart_obr_reform():
    data = load_transcribed()["obr_reform"]
    groups = data["groups"]
    parts = []
    for g in groups:
        ours, off = g["bars"][0]["value"], g["bars"][1]["value"]
        parts.append(f'For the {g["label"]} group, ours is {ours:.2f} against HMRC’s '
                     f'{off:.2f}, a deviation of {(ours / off - 1) * 100:+.1f}%.')
    desc = ("Grouped bar chart in billions of pounds per year. PolicyEngine's static costing of a "
            "1 percentage point rise in the UK basic rate of income tax, against HMRC's Direct "
            "effects of illustrative tax changes ready reckoner, June 2025 vintage. "
            + " ".join(parts) +
            " The 2028–29 emulator figure is interpolated between the scored endpoints "
            "£6.46bn in 2026 and £7.38bn in 2030.")
    return grouped_bars(
        "obr-reform",
        "obr-macro: 1p on the basic rate, ours vs HMRC ready reckoner (£bn/yr)",
        desc, groups, y_max=10, y_step=2, fmt=lambda v: f"{v:.2f}")


def chart_svar_fevd():
    ours, paper = load_svar()
    groups = [
        {"label": "UK GDP, 1-yr horizon",
         "bars": [{"name": "ours", "value": ours["gdp"], "series": 1},
                  {"name": "paper", "value": paper["gdp"], "series": 2}]},
        {"label": "UK CPI, 1-yr horizon",
         "bars": [{"name": "ours", "value": ours["cpi"], "series": 1},
                  {"name": "paper", "value": paper["cpi"], "series": 2}]},
    ]
    desc = (f"Grouped bar chart. Share of UK forecast-error variance attributed to identified "
            f"global shocks (world demand, energy and supply) at the one-year horizon. For GDP, "
            f"our 10,000-draw production run gives {ours['gdp']:.1f}% against the paper's "
            f"{paper['gdp']:.1f}%; for CPI, {ours['cpi']:.1f}% against {paper['cpi']:.1f}%. "
            f"Both deviations are a percentage point or less. The paper's values are approximate.")
    return grouped_bars(
        "svar-fevd",
        "boe-svar: global-shock FEVD shares, ours vs Brignone & Piffer (2025)",
        desc, groups, y_max=60, y_step=20, fmt=lambda v: f"{v:.1f}", unit_suffix="%")


def chart_frbus_residuals():
    data = load_transcribed()["frbus_residuals"]
    rows = data["rows"]
    W, H = 760, 300
    x0, x1 = 380.0, 692.7      # 1e-18 .. 1e-8
    lo_exp, hi_exp = -18, -8
    per_decade = (x1 - x0) / (hi_exp - lo_exp)

    desc = ("Horizontal bar chart on a base-10 logarithmic axis of maximum absolute residuals; "
            "shorter is closer. " +
            "; ".join(f"{r['label']}: {r['value']:.1e}" for r in rows) +
            ". The framing comparison is the last row: the Federal Reserve's own two pyfrbus "
            "releases disagree with each other by as much as this implementation disagrees with "
            "either, so our agreement sits at the scale of the reference implementation's own "
            "numerical noise rather than at a chosen tolerance.")

    out = svg_open(W, H, "frbus-residuals",
                   "frb-us: residuals against the Fed’s pyfrbus, log scale", desc)

    for k in range(6):
        e = lo_exp + 2 * k
        x = x0 + (e - lo_exp) * per_decade
        out.append(f'<line class="vc-grid" x1="{n(x)}" y1="26" x2="{n(x)}" y2="264.0"/>')
        out.append(f'<text class="vc-tick" x="{n(x)}" y="284.0" text-anchor="middle">1e{e}</text>')

    for i, r in enumerate(rows):
        y = 34.0 + 48 * i
        ty = y + 21
        w = (math.log10(r["value"]) - lo_exp) * per_decade
        out.append(f'<text class="vc-lab vc-rowlab" x="366" y="{n(ty)}" text-anchor="end">{esc(r["label"])}</text>')
        out.append(f'<rect class="vc-b{r["series"]}" x="380" y="{n(y)}" width="{n(w)}" height="34"/>')
        out.append(f'<text class="vc-val" x="{n(380 + w + 8)}" y="{n(ty)}">{r["value"]:.1e}</text>')

    out.append("</svg>")
    return "\n".join(out)


def chart_obr_freerun():
    quarters, free = load_levels("fig_free_running_data.csv", ["GDPM_model", "GDPM_efo"])
    _, anch = load_levels("fig_anchored_data.csv", ["GDPM_model", "GDPM_efo"])
    efo = [v / 1000 for v in free["GDPM_efo"]]        # £bn/qtr
    freerun = [v / 1000 for v in free["GDPM_model"]]
    anchored = [v / 1000 for v in anch["GDPM_model"]]

    W, H = 760, 330
    x0, x1 = 58.0, 724.0
    step = (x1 - x0) / (len(quarters) - 1)
    xs = [x0 + i * step for i in range(len(quarters))]
    lo, hi = 660.0, 740.0
    base_y, top_y = 292.0, 56.0
    per_unit = (base_y - top_y) / (hi - lo)
    def ymap(v):
        return base_y - (v - lo) * per_unit

    mad_a = sum(abs(m / e - 1) for m, e in zip(anch["GDPM_model"], anch["GDPM_efo"])) / len(quarters) * 100
    mad_f = sum(abs(m / e - 1) for m, e in zip(free["GDPM_model"], free["GDPM_efo"])) / len(quarters) * 100
    gap = efo[-1] - freerun[-1]

    desc = (
        f"Line chart of quarterly real GDP levels in billions of pounds, {quarters[0]} to "
        f"{quarters[-1]}. The published March 2026 EFO path rises from {efo[0]:.1f} to "
        f"{efo[-1]:.1f}. The anchored emulator is visually indistinguishable from it, running "
        f"from {anchored[0]:.1f} to {anchored[-1]:.1f} (mean absolute deviation {mad_a:.2f} per "
        f"cent, recomputed here from the plotted series). The free-running emulator, de-seeded "
        f"and with no add-factors, contracts from {freerun[0]:.1f} to {freerun[-1]:.1f} — a gap "
        f"that widens to {gap:.0f} billion pounds, {mad_f:.2f} per cent mean absolute deviation "
        f"over the horizon. Free-running and EFO paths from "
        f"papers/obr-macro/figures/fig_free_running_data.csv; anchored path from "
        f"papers/obr-macro/figures/fig_anchored_data.csv. Coordinates: value v in billions maps "
        f"to y = {base_y:g} - (v - {lo:g}) * {per_unit:g} on a {lo:g} to {hi:g} axis; quarter i "
        f"of {len(quarters)} maps to x = {x0:g} + i * {step:.3f}."
    )
    out = svg_open(W, H, "obr-freerun",
                   "obr-macro: real GDP level, anchored vs free-running vs the March 2026 EFO (£bn/qtr)",
                   desc)

    out.append('<line class="vc-s1" x1="58" y1="26" x2="84" y2="26"/><circle class="vc-s1-dot" cx="71" cy="26" r="3"/>')
    out.append(f'<text class="vc-lab" x="92" y="30">anchored ({mad_a:.2f}% MAD)</text>')
    out.append('<line class="vc-s2" x1="288" y1="26" x2="314" y2="26"/><circle class="vc-s2-dot" cx="301" cy="26" r="3"/>')
    out.append(f'<text class="vc-lab vc-lab2" x="322" y="30">free-running ({mad_f:.2f}% MAD)</text>')
    out.append('<line class="vc-grid" x1="540" y1="26" x2="566" y2="26"/><text class="vc-lab" x="574" y="30">EFO Mar 2026</text>')

    tick = lo
    while tick <= hi + 1e-9:
        y = ymap(tick)
        out.append(f'<line class="vc-grid" x1="{x0:.0f}" y1="{n(y)}" x2="{x1:.0f}" y2="{n(y)}"/>')
        out.append(f'<text class="vc-tick" x="48" y="{n(y + 4)}" text-anchor="end">{tick:g}</text>')
        tick += 20

    for cls, series in (("vc-s3", efo), ("vc-s1", anchored), ("vc-s2", freerun)):
        pts = " L".join(f"{n(xs[i])} {n(ymap(v))}" for i, v in enumerate(series))
        out.append(f'<path class="{cls}" d="M{pts}"/>')

    tick_idx = [i for i, q in enumerate(quarters) if q.endswith("Q1")]
    if len(quarters) - 1 not in tick_idx:
        tick_idx.append(len(quarters) - 1)
    for i in tick_idx:
        out.append(f'<text class="vc-tick" x="{n(xs[i])}" y="312" text-anchor="middle">{quarters[i]}</text>')

    out.append("</svg>")
    return "\n".join(out)


def chart_obr_outturn():
    rows = load_outturn()
    W, H = 760, 304
    base_y, top_y = 258.0, 26.0
    y_max = 0.7
    scale = (base_y - top_y) / y_max
    bar_w, bar_gap = 40.0, 8.0
    group_step = 165.0
    first_centre = 138.5

    parts = "; ".join(f"{q}: emulator {m:.2f}, EFO {e:.2f}, ONS {o:g}" for q, m, e, o in rows)
    desc = (
        f"Grouped bar chart, percentage quarter-on-quarter real GDP growth for the four "
        f"quarters with ONS outturns since anchoring. {parts}. The emulator tracks the three "
        f"2025 outturns to within 0.06 points; both the emulator and the November EFO it "
        f"inherits miss the strong 0.6 per cent 2026Q1 outturn by roughly a quarter of a point. "
        f"Data from papers/obr-macro/figures/fig_outturn_data.csv. Coordinates: value v maps to "
        f"y = {base_y:g} - v * {scale:.1f} on a 0 to {y_max:g} axis."
    )
    out = svg_open(W, H, "obr-outturn",
                   "obr-macro: quarterly real GDP growth — emulator vs EFO Nov 2025 vs ONS outturn (% q/q)",
                   desc)

    out.append('<rect class="vc-b1" x="64" y="8" width="14" height="10"/><text class="vc-lab" x="84" y="17">emulator</text>')
    out.append('<rect class="vc-b3" x="196" y="8" width="14" height="10"/><text class="vc-lab" x="216" y="17">EFO Nov 2025</text>')
    out.append('<rect class="vc-b2" x="356" y="8" width="14" height="10"/><text class="vc-lab" x="376" y="17">ONS outturn</text>')

    t = 0.0
    while t <= 0.6 + 1e-9:
        y = base_y - t * scale
        cls = "vc-axis" if t == 0 else "vc-grid"
        out.append(f'<line class="{cls}" x1="64" y1="{n(y)}" x2="724" y2="{n(y)}"/>')
        out.append(f'<text class="vc-tick" x="54" y="{n(y + 4)}" text-anchor="end">{t:g}</text>')
        t += 0.2

    for g, (q, m, e, o) in enumerate(rows):
        centre = first_centre + g * group_step
        for k, (cls, v) in enumerate((("vc-b1", m), ("vc-b3", e), ("vc-b2", o))):
            x = centre - 60.0 + k * (bar_w + bar_gap)
            h = v * scale
            y = base_y - h
            out.append(f'<rect class="{cls}" x="{n(x)}" y="{n(y)}" width="{bar_w:g}" height="{n(h)}"/>')
            out.append(f'<text class="vc-val" x="{n(x + bar_w / 2)}" y="{n(y - 6)}" text-anchor="middle">{v:.2f}</text>')
        out.append(f'<text class="vc-lab" x="{n(centre)}" y="296" text-anchor="middle">{q}</text>')

    out.append("</svg>")
    return "\n".join(out)


def chart_svar_fan():
    quarters, series, outturns = load_svar_fan()
    W, H = 760, 340
    base_y, top_y = 292.0, 56.0
    per_unit = (base_y - top_y) / 4.0     # 4-unit span on each panel
    panels = {"gdp": {"x0": 58.0, "x1": 366.0, "v_lo": -1.0},
              "cpi": {"x0": 416.0, "x1": 724.0, "v_lo": 1.0}}
    npts = len(quarters)
    for p in panels.values():
        p["step"] = (p["x1"] - p["x0"]) / (npts - 1)
        p["xs"] = [p["x0"] + i * p["step"] for i in range(npts)]
        p["ymap"] = (lambda lo: lambda v: base_y - (v - lo) * per_unit)(p["v_lo"])

    n_out = len(outturns["gdp"])
    rmse = {var: math.sqrt(sum((m - o) ** 2 for m, o in
                               zip(series[var]["median"][:n_out], outturns[var])) / n_out)
            for var in ("gdp", "cpi")}
    med_g = ", ".join(f"{v:.1f}" for v in series["gdp"]["median"][:n_out])
    med_c = ", ".join(f"{v:.1f}" for v in series["cpi"]["median"][:n_out])
    out_g = ", ".join(f"{v:.1f}" for v in outturns["gdp"])
    out_c = ", ".join(f"{v:.1f}" for v in outturns["cpi"])

    desc = (
        f"Two-panel fan chart. Left panel: year-on-year UK GDP growth; right panel: year-on-year "
        f"UK CPI inflation. Each shows the posterior median forecast from the frozen 2024Q2 data "
        f"edge as a line, the 68 per cent credible band as a shaded region over thirteen quarters "
        f"{quarters[0]} to {quarters[-1]}, and ONS outturns for the seven evaluated quarters "
        f"{quarters[0]} to {quarters[n_out - 1]} as dots. All fourteen outturn dots fall inside "
        f"the 68 per cent band; RMSE {rmse['gdp']:.2f} percentage points for both variables. GDP "
        f"medians run {med_g} per cent over the evaluated quarters against outturns of {out_g}; "
        f"CPI medians {med_c} against outturns of {out_c}. Medians and 68 per cent bands from "
        f"papers/boe-svar/figures/figure_numbers.json (forecast_table, entries [median, lo68, "
        f"hi68]); ONS outturns from papers/boe-svar/figures/make_figures.py. Coordinates: GDP "
        f"panel maps value v to y = {base_y:g} - (v + 1) * {per_unit:g} for the -1 to 3 per cent "
        f"axis; CPI panel y = {base_y:g} - (v - 1) * {per_unit:g} for the 1 to 5 per cent axis; "
        f"quarter i of {npts} maps to x = 58 + i * {panels['gdp']['step']:.3f} (GDP) or "
        f"416 + i * {panels['cpi']['step']:.3f} (CPI)."
    )
    out = svg_open(W, H, "svar-fan",
                   "boe-svar: out-of-sample forecast fan from the frozen 2024Q2 edge vs ONS outturns",
                   desc)

    out.append('<line class="vc-s1" x1="58" y1="26" x2="84" y2="26"/><text class="vc-lab" x="92" y="30">median forecast + 68% band</text>')
    out.append('<circle class="vc-s2-dot" cx="352" cy="26" r="3.5"/><text class="vc-lab vc-lab2" x="362" y="30">ONS outturn</text>')
    out.append('<text class="vc-note" x="540" y="30">frozen 2024Q2 edge at left of each panel</text>')

    gdp_labels = {-1: "-1%", 0: "0", 1: "+1%", 2: "+2%", 3: "+3%"}
    for var, labels in (("gdp", gdp_labels), ("cpi", None)):
        p = panels[var]
        for k in range(5):
            v = p["v_lo"] + k
            y = p["ymap"](v)
            cls = "vc-axis" if var == "gdp" and v == 0 else "vc-grid"
            out.append(f'<line class="{cls}" x1="{p["x0"]:.0f}" y1="{n(y)}" x2="{p["x1"]:.0f}" y2="{n(y)}"/>')
            label = labels[v] if labels else f"{v:g}%"
            out.append(f'<text class="vc-tick" x="{p["x0"] - 6:.0f}" y="{n(y + 4)}" text-anchor="end">{label}</text>')

    for var in ("gdp", "cpi"):
        p = panels[var]
        hi_pts = " L".join(f"{n(p['xs'][i])} {n(p['ymap'](v))}" for i, v in enumerate(series[var]["hi68"]))
        lo_pts = " L".join(f"{n(p['xs'][i])} {n(p['ymap'](v))}"
                           for i, v in reversed(list(enumerate(series[var]["lo68"]))))
        out.append(f'<path class="vc-band" d="M{hi_pts} L{lo_pts} Z"/>')

    for var in ("gdp", "cpi"):
        p = panels[var]
        out.append(f'<line class="vc-edge" x1="{p["x0"]:.0f}" y1="{top_y:.0f}" x2="{p["x0"]:.0f}" y2="{base_y:.0f}"/>')
    for var in ("gdp", "cpi"):
        p = panels[var]
        pts = " L".join(f"{n(p['xs'][i])} {n(p['ymap'](v))}" for i, v in enumerate(series[var]["median"]))
        out.append(f'<path class="vc-s1" d="M{pts}"/>')
    for var in ("gdp", "cpi"):
        p = panels[var]
        for i, v in enumerate(outturns[var]):
            out.append(f'<circle class="vc-s2-dot" cx="{n(p["xs"][i])}" cy="{n(p["ymap"](v))}" r="3.5"/>')

    out.append('<text class="vc-lab" x="212" y="318" text-anchor="middle">GDP growth (YoY, %)</text>')
    out.append('<text class="vc-lab" x="570" y="318" text-anchor="middle">CPI inflation (YoY, %)</text>')
    for var in ("gdp", "cpi"):
        p = panels[var]
        for i in (0, (npts - 1) // 2, npts - 1):
            out.append(f'<text class="vc-tick" x="{n(p["xs"][i])}" y="334" text-anchor="middle">{quarters[i][2:]}</text>')

    out.append("</svg>")
    return "\n".join(out)


BUILDERS = [
    ("obr-anchored", chart_obr_anchored),
    ("obr-reform", chart_obr_reform),
    ("obr-freerun", chart_obr_freerun),
    ("obr-outturn", chart_obr_outturn),
    ("svar-fevd", chart_svar_fevd),
    ("svar-fan", chart_svar_fan),
    ("frbus-residuals", chart_frbus_residuals),
]

SVG_RE = re.compile(r'<svg class="vchart".*?</svg>', re.DOTALL)


def render_page(html: str) -> str:
    charts = [build() for _, build in BUILDERS]
    found = SVG_RE.findall(html)
    if len(found) != len(charts):
        sys.exit(f"expected {len(charts)} .vchart SVGs in {PAGE}, found {len(found)}")
    it = iter(charts)
    return SVG_RE.sub(lambda _m: next(it).replace("\\", "\\\\"), html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the page is out of date")
    args = ap.parse_args()

    html = PAGE.read_text()
    new = render_page(html)
    if args.check:
        if new != html:
            print(f"{PAGE} is out of date; run: python3 validation/figures/make_charts.py")
            return 1
        print("validation/index.html charts are up to date.")
        return 0
    if new == html:
        print("no change.")
    else:
        PAGE.write_text(new)
        print(f"wrote {len(BUILDERS)} charts into {PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
