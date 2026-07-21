"""Build the current boe-svar forecast data used by the live site.

This is deliberately separate from ``make_figures.py``.  The latter freezes
the information set at 2024Q2 for honest out-of-sample validation; this script
uses the latest complete quarterly data edge (currently 2026Q1) and forecasts
from 2026Q2.  The replication coefficients remain estimated on the paper's
1992Q1--2023Q2 sample, matching the hosted adapter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/janansadeqian/boe-var-model")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from boe_var import analysis, forecast  # noqa: E402
from boe_var.bvar import BVAR  # noqa: E402
from boe_var.data import load_data  # noqa: E402
from boe_var.identification import identify  # noqa: E402

SEED = 20260721
N_DRAWS = 3000
N_PATHS = 5
HORIZONS = 13
I_CPI, I_GDP = 5, 7


def covid_dummies(index: pd.PeriodIndex) -> np.ndarray:
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    return np.column_stack([(index == q).astype(float) for q in quarters])


def main() -> None:
    rng = np.random.default_rng(SEED)
    full = load_data().loc["1992Q1":"2026Q1"]
    if full.index[-1] != pd.Period("2026Q1", "Q"):
        raise RuntimeError(f"expected 2026Q1 data edge, got {full.index[-1]}")

    estimation = full.loc[:"2023Q2"]
    model = BVAR(
        estimation.to_numpy(float),
        lags=4,
        dummies=covid_dummies(estimation.index),
        lam=0.2,
        mu=1.0,
    )
    draws = model.sample_posterior(N_DRAWS, seed=SEED)
    triples = identify(draws, rng=rng, compute_weights=False)
    pairs = [(draw, impact) for draw, impact, _weight in triples]
    if not pairs:
        raise RuntimeError("no accepted identification draws")

    history = full.to_numpy(float)
    tail = history[-4:]
    paths = []
    for draw, _impact in pairs:
        for _ in range(N_PATHS):
            levels = forecast.sample_forecast(draw, history, horizons=HORIZONS, rng=rng)
            paths.append(forecast.yoy(np.vstack([tail, levels])))
    bands = analysis.aggregate(paths)
    quarters = pd.period_range("2026Q2", periods=HORIZONS, freq="Q")

    def values(i: int, h: int) -> dict[str, float]:
        return {
            "median": float(bands["median"][h, i]),
            "lo68": float(bands["lo68"][h, i]),
            "hi68": float(bands["hi68"][h, i]),
            "lo90": float(bands["lo90"][h, i]),
            "hi90": float(bands["hi90"][h, i]),
        }

    payload = {
        "model": "boe-svar",
        "generated": "2026-07-21",
        "data_edge": "2026Q1",
        "forecast_start": "2026Q2",
        "estimation_sample": "1992Q1-2023Q2",
        "draws": N_DRAWS,
        "accepted": len(pairs),
        "paths_per_draw": N_PATHS,
        "units": "year-on-year percent",
        "source": "PolicyEngine/boe-var-model public-data pipeline",
        "forecast": {
            str(q): {"gdp": values(I_GDP, h), "cpi": values(I_CPI, h)}
            for h, q in enumerate(quarters)
        },
    }
    target = HERE / "current_forecast.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {target} ({len(pairs)}/{N_DRAWS} draws accepted)")


if __name__ == "__main__":
    main()
