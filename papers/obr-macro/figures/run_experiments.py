"""Additional worked policy experiments through the direct macro levers.

Runs three experiments with obr_macro.run_reform and writes a summary CSV
used for the tables in the paper:

  A. Government consumption +£5bn/yr (CGG +£1.25bn per quarter, 12 quarters).
  B. VAT-style price shock: CPI +1.4 index points (~+1 per cent price level)
     for 12 quarters -- the CPI is exogenous in this configuration,
     so the shock propagates via PCE (PCE/PCE(-4) = CPI/CPI(-4)) into real
     household income and consumption.
  C. Corporation tax +2pp (TCPRO +0.02, investment closure, 12 quarters).
"""

from pathlib import Path

import pandas as pd

from obr_macro import run_reform

HERE = Path(__file__).parent

EXPERIMENTS = [
    dict(name="Gov consumption +£5bn/yr", var="CGG", shock=1250,
         investment_closure=False),
    dict(name="VAT-style price shock +1%", var="CPI", shock=1.4,
         investment_closure=False),
    dict(name="Corporation tax +2pp", var="TCPRO", shock=0.02,
         investment_closure=True),
]


def main():
    frames = []
    for exp in EXPERIMENTS:
        print(f"Running: {exp['name']} ...")
        df = run_reform(
            name=exp["name"],
            var=exp["var"],
            shock=exp["shock"],
            periods=12,
            investment_closure=exp["investment_closure"],
        )
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(HERE / "experiments_quarterly.csv", index=False)

    # Annual summary: mean quarterly deltas by calendar year
    all_df["year"] = all_df["period"].str[:4]
    summary = (
        all_df.groupby(["reform", "year"])
        .agg(
            avg_pct_gdp=("pct_gdp", "mean"),
            avg_delta_gdp_bn=("delta_gdp_bn", "mean"),
            avg_delta_cons_bn=("delta_cons_m", lambda s: s.mean() / 1000),
            avg_delta_if_bn=("delta_if_m", lambda s: s.mean() / 1000),
        )
        .round(4)
    )
    summary.to_csv(HERE / "experiments_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
