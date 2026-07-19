#!/usr/bin/env python3
"""Regenerate the inline SVG fan chart in the homepage hero from committed data.

Run:  python3 assets/make_hero.py          # rewrite index.html
      python3 assets/make_hero.py --check  # exit 1 if the page is stale

Same approach, and the same reasons, as validation/figures/make_charts.py: the
markup must stay *inline* (the site ships no third-party assets and enforces
default-src 'self'), and every fill and stroke must sit on a class defined in
style.css so the figure retint itself for light and dark.

The hero motif is the boe-svar UK real-GDP growth fan: the 68% and 90% credible
bands and the posterior median, straight out of
papers/boe-svar/figures/figure_numbers.json — the same numbers the paper's
Figure "fan" is drawn from. Nothing here is invented for decoration.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "index.html"
SRC = ROOT / "papers" / "boe-svar" / "figures" / "figure_numbers.json"

MARK_OPEN = "<!-- hero-fan:begin -->"
MARK_CLOSE = "<!-- hero-fan:end -->"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def n(x: float) -> str:
    return f"{x:.1f}"


def load_fan():
    """Quarterly year-on-year UK real GDP growth: median, 68% and 90% bands."""
    table = json.loads(SRC.read_text())["forecast_table"]
    quarters = list(table)
    med, lo68, hi68, lo90, hi90 = [], [], [], [], []
    for q in quarters:
        m, l68, h68, l90, h90 = table[q]["gdp"]
        med.append(m); lo68.append(l68); hi68.append(h68)
        lo90.append(l90); hi90.append(h90)
    return quarters, med, lo68, hi68, lo90, hi90


def build() -> str:
    quarters, med, lo68, hi68, lo90, hi90 = load_fan()

    W, H = 640, 400
    x0, x1 = 30.0, 604.0
    top, bot = 40.0, 344.0          # plot area for the value range below
    v_lo, v_hi = -1.7, 3.4
    step = (x1 - x0) / (len(quarters) - 1)
    xs = [x0 + i * step for i in range(len(quarters))]
    ymap = lambda v: bot - (v - v_lo) / (v_hi - v_lo) * (bot - top)

    def band(lo, hi):
        up = " L".join(f"{n(xs[i])} {n(ymap(v))}" for i, v in enumerate(hi))
        dn = " L".join(f"{n(xs[i])} {n(ymap(v))}" for i, v in reversed(list(enumerate(lo))))
        return f"M{up} L{dn} Z"

    desc = (
        f"Fan chart. Posterior forecast of UK year-on-year real GDP growth from the "
        f"boe-svar model, {quarters[0]} to {quarters[-1]}, from a 3,000-draw run. The "
        f"median path runs from {med[0]:+.2f}% to {med[-1]:+.2f}%. The 90% credible band "
        f"is widest at {quarters[lo90.index(min(lo90))]}, spanning {min(lo90):+.2f}% to "
        f"{max(hi90):+.2f}%, and the lower edge stays below zero throughout the forecast: "
        f"the model does not rule out a contraction. This figure is the site's hero motif "
        f"and repeats data published on the boe-svar paper page."
    )

    out = [
        f'{MARK_OPEN}',
        f'<svg class="hero-fan" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-labelledby="hero-fan-t hero-fan-d">',
        f'<title id="hero-fan-t">boe-svar: UK real GDP growth forecast, median with 68% and 90% credible bands</title>',
        f'<desc id="hero-fan-d">{esc(desc)}</desc>',
    ]

    for tick in (-1, 0, 1, 2, 3):
        y = ymap(tick)
        cls = "hf-zero" if tick == 0 else "hf-grid"
        out.append(f'<line class="{cls}" x1="{n(x0)}" y1="{n(y)}" x2="{n(x1)}" y2="{n(y)}"/>')
        out.append(f'<text class="hf-tick" x="{n(x0)}" y="{n(y - 7)}">{tick:+d}%</text>')

    out.append(f'<path class="hf-b90" d="{band(lo90, hi90)}"/>')
    out.append(f'<path class="hf-b68" d="{band(lo68, hi68)}"/>')
    line = " L".join(f"{n(xs[i])} {n(ymap(v))}" for i, v in enumerate(med))
    out.append(f'<path class="hf-med" d="M{line}"/>')
    out.append(f'<circle class="hf-dot" cx="{n(xs[-1])}" cy="{n(ymap(med[-1]))}" r="4"/>')

    out.append(f'<text class="hf-tick hf-q" x="{n(x0)}" y="366">{esc(quarters[0])}</text>')
    out.append(f'<text class="hf-tick hf-q" x="{n(x1)}" y="366" text-anchor="end">{esc(quarters[-1])}</text>')
    # No in-SVG caption: the <figcaption> next to this figure says the same thing
    # at real body size. Below 1000px the viewBox scales to roughly half, which
    # would render these 11px labels at ~6px, so style.css hides .hf-tick there
    # and the figure reads as a shape with the prose caption carrying the detail.
    out.append("</svg>")
    out.append(MARK_CLOSE)
    return "\n".join(out)


BLOCK_RE = re.compile(re.escape(MARK_OPEN) + r".*?" + re.escape(MARK_CLOSE), re.DOTALL)


def render_page(html: str) -> str:
    if not BLOCK_RE.search(html):
        sys.exit(f"no {MARK_OPEN} … {MARK_CLOSE} block in {PAGE}")
    return BLOCK_RE.sub(lambda _m: build().replace("\\", "\\\\"), html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the page is out of date")
    args = ap.parse_args()

    html = PAGE.read_text()
    new = render_page(html)
    if args.check:
        if new != html:
            print(f"{PAGE} is out of date; run: python3 assets/make_hero.py")
            return 1
        print("index.html hero fan is up to date.")
        return 0
    if new == html:
        print("no change.")
    else:
        PAGE.write_text(new)
        print(f"wrote the hero fan into {PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
