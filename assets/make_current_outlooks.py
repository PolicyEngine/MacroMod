#!/usr/bin/env python3
"""Regenerate current-outlook SVGs on the OBR and boe-svar model pages."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def points(values, xs, ymap):
    return " ".join(f"{x:.1f},{ymap(v):.1f}" for x, v in zip(xs, values))


def obr_chart() -> str:
    source = ROOT / "papers/obr-macro/figures/current_outlook.csv"
    rows = list(csv.DictReader(source.open()))
    q = [r["quarter"] for r in rows]
    g0, c0 = float(rows[0]["real_gdp_m"]), float(rows[0]["consumption_m"])
    g = [100 * float(r["real_gdp_m"]) / g0 for r in rows]
    c = [100 * float(r["consumption_m"]) / c0 for r in rows]
    x0, x1, top, bot = 62, 738, 42, 278
    xs = [x0 + i * (x1 - x0) / (len(q) - 1) for i in range(len(q))]

    def ymap(v):
        return bot - (v - 99) / 11 * (bot - top)

    out = [
        '<figure class="vfig current-outlook">',
        '<svg class="vchart" viewBox="0 0 800 326" role="img" aria-labelledby="obr-current-t obr-current-d">',
        '<title id="obr-current-t">March 2026 OBR outlook for real GDP and household consumption</title>',
        '<desc id="obr-current-d">Indexed line chart from 2026Q1 to 2031Q1. Real GDP rises from 100 to %.1f and household consumption from 100 to %.1f.</desc>'
        % (g[-1], c[-1]),
        '<text class="vc-lab" x="62" y="25">March 2026 EFO outlook · 2026Q1 = 100</text>',
    ]
    for tick in (100, 102, 104, 106, 108, 110):
        y = ymap(tick)
        out += [
            f'<line class="vc-grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>',
            f'<text class="vc-tick" x="52" y="{y + 4:.1f}" text-anchor="end">{tick}</text>',
        ]
    out += [
        f'<polyline class="vc-s1" points="{points(g, xs, ymap)}"/>',
        f'<polyline class="vc-s2" points="{points(c, xs, ymap)}"/>',
        f'<circle class="vc-s1-dot" cx="{xs[-1]:.1f}" cy="{ymap(g[-1]):.1f}" r="3.5"/>',
        f'<circle class="vc-s2-dot" cx="{xs[-1]:.1f}" cy="{ymap(c[-1]):.1f}" r="3.5"/>',
        '<text class="vc-lab" x="565" y="25">real GDP</text><text class="vc-lab vc-lab2" x="650" y="25">consumption</text>',
    ]
    for i in (0, 4, 8, 12, 16, 20):
        out.append(
            f'<text class="vc-tick" x="{xs[i]:.1f}" y="302" text-anchor="middle">{q[i]}</text>'
        )
    out += [
        "</svg>",
        "<figcaption>Latest official baseline available on 21 July 2026. Levels are indexed to 2026Q1 so the two series can share one honest scale. Source: OBR March 2026 detailed forecast tables; committed values in <code>papers/obr-macro/figures/current_outlook.csv</code>.</figcaption>",
        "</figure>",
    ]
    return "\n".join(out)


def boe_chart() -> str:
    source = ROOT / "papers/boe-svar/figures/current_forecast.json"
    data = json.loads(source.read_text())
    table = data["forecast"]
    q = list(table)
    med = [table[x]["gdp"]["median"] for x in q]
    lo68 = [table[x]["gdp"]["lo68"] for x in q]
    hi68 = [table[x]["gdp"]["hi68"] for x in q]
    lo90 = [table[x]["gdp"]["lo90"] for x in q]
    hi90 = [table[x]["gdp"]["hi90"] for x in q]
    x0, x1, top, bot = 62, 738, 42, 278
    xs = [x0 + i * (x1 - x0) / (len(q) - 1) for i in range(len(q))]
    vlo = min(lo90) - 0.25
    vhi = max(hi90) + 0.25

    def ymap(v):
        return bot - (v - vlo) / (vhi - vlo) * (bot - top)

    def band(lo, hi):
        upper = " L".join(f"{x:.1f} {ymap(v):.1f}" for x, v in zip(xs, hi))
        lower = " L".join(
            f"{x:.1f} {ymap(v):.1f}" for x, v in reversed(list(zip(xs, lo)))
        )
        return f"M{upper} L{lower} Z"

    ticks = range(int(vlo), int(vhi) + 1)
    out = [
        '<figure class="vfig current-outlook">',
        '<svg class="vchart" viewBox="0 0 800 326" role="img" aria-labelledby="boe-current-t boe-current-d">',
        '<title id="boe-current-t">Current boe-svar UK real GDP growth forecast</title>',
        f'<desc id="boe-current-d">Fan chart from the latest complete quarterly data edge, {data["data_edge"]}, forecasting {q[0]} to {q[-1]}. Median with 68 and 90 percent predictive bands.</desc>',
        f'<text class="vc-lab" x="62" y="25">Current forecast from {data["data_edge"]} · year-on-year %</text>',
    ]
    for tick in ticks:
        y = ymap(tick)
        cls = "vc-axis" if tick == 0 else "vc-grid"
        out += [
            f'<line class="{cls}" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>',
            f'<text class="vc-tick" x="52" y="{y + 4:.1f}" text-anchor="end">{tick:+d}%</text>',
        ]
    out += [
        f'<path class="vc-band90" d="{band(lo90, hi90)}"/>',
        f'<path class="vc-band68" d="{band(lo68, hi68)}"/>',
        f'<polyline class="vc-s1" points="{points(med, xs, ymap)}"/>',
        f'<circle class="vc-s1-dot" cx="{xs[-1]:.1f}" cy="{ymap(med[-1]):.1f}" r="3.5"/>',
        '<text class="vc-note" x="738" y="25" text-anchor="end">90% / 68% bands · median</text>',
    ]
    for i in (0, 4, 8, 12):
        out.append(
            f'<text class="vc-tick" x="{xs[i]:.1f}" y="302" text-anchor="middle">{q[i]}</text>'
        )
    out += [
        "</svg>",
        f"<figcaption>Live-facing forecast, distinct from the frozen 2024Q2 validation experiment. Data through {data['data_edge']}; forecast begins {data['forecast_start']}. {data['draws']:,} posterior draws, {data['accepted']} accepted; five stochastic paths per accepted draw. Source and provenance: <code>papers/boe-svar/figures/current_forecast.json</code>.</figcaption>",
        "</figure>",
    ]
    return "\n".join(out)


def inject(page: Path, name: str, content: str) -> None:
    start, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    html = page.read_text()
    replacement = f"{start}\n{content}\n{end}"
    updated, count = re.subn(
        re.escape(start) + ".*?" + re.escape(end),
        lambda _: replacement,
        html,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"expected one {name} block in {page}, found {count}")
    page.write_text(updated)


if __name__ == "__main__":
    inject(ROOT / "obr/index.html", "obr-current-outlook", obr_chart())
    inject(ROOT / "svar/index.html", "boe-current-outlook", boe_chart())
    print("updated current-outlook charts")
