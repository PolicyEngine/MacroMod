"""Figure 3: the 1p basic-rate reform -- revenue and second-round GDP effects.

Left panel: the PolicyEngine static costing by year (endpoints from the scored
reform: £6.46bn in 2026 rising to £7.38bn by 2030, interior years linearly
interpolated) against the range of HMRC's published ready reckoner (~£6-8bn
per 1p on the basic rate).

Right panel: the second-round GDP effect from propagating the corresponding
quarterly HHDI shock (costing x 1000/4, sign-flipped) through the anchored
emulator under the demand closure, per quarter over the scenario horizon.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import BLUE, ORANGE, GREY, VERMILLION, quarters_axis

from obr_macro.reform_analysis import _build_reform_template

HERE = Path(__file__).parent
START, END = "2025Q1", "2027Q4"
SHOCK_START = "2026Q1"

# Annual static costing, £bn (PolicyEngine microsimulation, basic rate 20p->21p
# from April 2026): 2026 and 2030 as scored through the bridge, interior years
# linearly interpolated.
YEARS = [2026, 2027, 2028, 2029, 2030]
COSTING = list(np.linspace(6.46, 7.38, 5))
HMRC_LO, HMRC_HI = 6.0, 8.0


def main():
    template = _build_reform_template("HHDI", START, END, False)
    baseline = template.clone()
    shocked = template.clone()

    # Quarterly HHDI shock path: -costing * 1000/4 (£m per quarter), flat
    # within each calendar year, applied from 2026Q1 to the model horizon end.
    t0 = shocked.period_idx(SHOCK_START)
    t1 = shocked.period_idx(END)
    for t in range(t0, t1 + 1):
        year = shocked.index[t].year
        q_shock = -COSTING[YEARS.index(year)] * 1000 / 4
        shocked.apply_shock("HHDI", q_shock, str(shocked.index[t]), periods=1)

    baseline.solve(START, END)
    shocked.solve(START, END)

    idx = range(t0, t1 + 1)
    periods = [baseline.index[t] for t in idx]
    gdp_b = baseline.data.iloc[list(idx)]["GDPM"].values
    gdp_s = shocked.data.iloc[list(idx)]["GDPM"].values
    pct = 100 * (gdp_s - gdp_b) / gdp_b

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    ax.axhspan(HMRC_LO, HMRC_HI, color=GREY, alpha=0.18,
               label="HMRC ready reckoner range")
    ax.bar([str(y) for y in YEARS], COSTING, color=BLUE, width=0.6,
           label="PolicyEngine static costing")
    ax.set_ylabel("£bn per year")
    ax.set_title("Static costing: basic rate +1pp")
    ax.set_ylim(0, 9)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.plot(range(len(periods)), pct, color=VERMILLION, lw=1.6, marker="o", ms=3)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("% deviation of GDP from baseline")
    ax.set_title("Second-round GDP effect")
    quarters_axis(ax, periods, step=1)

    fig.tight_layout()
    fig.savefig(HERE / "fig_reform.pdf")

    import pandas as pd
    pd.DataFrame({"period": [str(p) for p in periods], "pct_gdp": pct}).to_csv(
        HERE / "fig_reform_data.csv", index=False
    )
    print("wrote fig_reform.pdf")
    for p, v in zip(periods, pct):
        print(p, f"{v:+.4f}")


if __name__ == "__main__":
    main()
