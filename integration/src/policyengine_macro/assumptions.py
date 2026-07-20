"""Macro -> micro EconomicAssumptions overlay (PolicyEngine/macro#11).

Carries the OG-UK model's long-run price changes (wages, labour supply,
interest rates) into the PolicyEngine microsimulation as a parametric
overlay on the OBR uprating indices, so a "dynamic" population score is
the ordinary static score run under macro-adjusted economic assumptions.

DOUBLE-COUNTING INVARIANT
-------------------------
The overlay carries only the reform/baseline RATIO from the macro model,
never a level. The baseline microsim run uses the stock parameters — which
already embed the OBR forecast the OG-UK baseline is calibrated to — so the
static effect of the reform is never counted twice: a no-op macro result
(w_reform == w_baseline) produces an EMPTY overlay and dynamic scoring
reduces exactly to static scoring. Tests assert this.

WHY THE DERIVED INDICES, NOT yoy_growth
---------------------------------------
policyengine-uk 2.88.x stores OBR year-on-year growth under
``gov.economic_assumptions.yoy_growth.obr.*`` but uprates input variables
against the DERIVED cumulative indices ``gov.economic_assumptions.indices.
obr.*``, built ONCE at system load by create_economic_assumption_indices.
Overriding yoy_growth in a reform does NOT rebuild the indices, so the
overlay must override the derived index values directly.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from pydantic import BaseModel

# The single uprating parameter the v1 overlay touches: nominal average
# earnings, against which employment_income_before_lsr is uprated.
EARNINGS_INDEX_PARAM = "gov.economic_assumptions.indices.obr.average_earnings"

# Any user reform touching this subtree collides with the overlay and is
# refused (the merge would silently drop one side's values).
OVERLAY_PARAM_PREFIX = "gov.economic_assumptions."

# Years the flat steady-state factor is applied from start_year.
DEFAULT_OVERLAY_YEARS = 10

# create_economic_assumption_indices builds indices up to (exclusive) 2040.
_INDEX_LAST_YEAR = 2039


class EconomicAssumptions(BaseModel):
    """Macro-model price changes expressed as microsim uprating factors.

    Steady-state comparative statics: the factors are LONG-RUN level shifts
    (reform/baseline ratios), applied flat from ``start_year`` with no
    transition dynamics — that assumption is spelled out in ``notes`` and
    must be carried into any ScoreResult built from this object.

    v1 scope (deliberately narrow, and honest about it):
    - ``earnings_factor`` is applied to the average-earnings uprating index
      (EARNINGS_INDEX_PARAM) only.
    - ``labour_supply_factor`` is REPORTED in assumptions/caveats but not
      allocated to any parameter: an aggregate hours change has no
      distributional incidence the microsim could apply without inventing
      one.
    - No price-level overlay: the OG model is real (no price level).
    """

    source: str
    start_year: int
    earnings_factor: float        # w_reform / w_baseline
    labour_supply_factor: float   # L_reform / L_baseline
    interest_rate_baseline: float
    interest_rate_reform: float
    notes: list[str] = []

    @classmethod
    def from_og_result(cls, og_payload: dict) -> "EconomicAssumptions":
        """Construct from an og_score_reform payload.

        Uses the two ``*_steady_state_model_units`` dicts (fields r, w, Y,
        K, L, ...). The model is real, so w and L ratios are the only price
        signals carried; r is reported for context.
        """
        base = og_payload["baseline_steady_state_model_units"]
        ref = og_payload["reform_steady_state_model_units"]
        start_year = int(og_payload["start_year"])
        return cls(
            source=(
                "OG-UK overlapping generations (steady state), "
                "pooled ages, single representative sector"
            ),
            start_year=start_year,
            earnings_factor=round(ref["w"] / base["w"], 6),
            labour_supply_factor=round(ref["L"] / base["L"], 6),
            interest_rate_baseline=base["r"],
            interest_rate_reform=ref["r"],
            notes=[
                f"steady-state overlay: long-run factor applied uniformly "
                f"from {start_year}; no transition dynamics",
                "overlay carries only the reform/baseline ratio, so the "
                "static effect embedded in the stock parameters is never "
                "counted twice",
            ],
        )

    def to_parameter_reform(
        self,
        base_index_values: dict[int, float],
        years: int = DEFAULT_OVERLAY_YEARS,
    ) -> dict:
        """Overlay as a {parameter_path: {date: value}} PolicyEngine reform.

        MULTIPLIES the baseline derived-index values by ``earnings_factor``
        for start_year..start_year+years-1 (years present in
        ``base_index_values`` only). Invariant: a no-op macro result
        (earnings_factor == 1) returns an EMPTY dict, so dynamic scoring
        with a null macro effect reduces exactly to static scoring — the
        double-counting guard in module docstring form.
        """
        if self.earnings_factor == 1.0:
            return {}
        window = range(self.start_year, self.start_year + int(years))
        dated = {
            f"{y}-01-01": round(base_index_values[y] * self.earnings_factor, 6)
            for y in window
            if y in base_index_values
        }
        if not dated:
            raise ValueError(
                f"no baseline index values available for years "
                f"{window.start}..{window.stop - 1}; got years "
                f"{sorted(base_index_values)}"
            )
        return {EARNINGS_INDEX_PARAM: dated}

    def assumption_strings(self) -> list[str]:
        return [f"macro source: {self.source}", *self.notes]

    def caveat_strings(self) -> list[str]:
        hours_pct = 100.0 * (self.labour_supply_factor - 1.0)
        return [
            f"aggregate hours change {hours_pct:+.2f}% not distributionally "
            "allocated in v1 (labour_supply_factor is reported, not applied "
            "to any parameter)",
            "earnings factor applied to the average-earnings uprating index "
            "only; other uprated incomes (self-employment, pensions) are "
            "not adjusted in v1",
            "no price-level overlay: the OG model is real",
        ]


# ---------------------------------------------------------------------------
# Baseline derived-index values
# ---------------------------------------------------------------------------
#
# The derived indices do not exist in YAML — policyengine-uk builds them at
# system load. Two ways to read the baseline values:
#   (a) engine path (primary): read the built parameter at runtime — works
#       wherever the policyengine engine imports (e.g. the hosted image);
#   (b) YAML fallback (also the tests' cross-check): compound the
#       yoy_growth.obr.* series exactly as create_economic_assumption_indices
#       does — base 1.0 at the series' earliest year, index[y] =
#       round(index[y-1] * (1 + growth[y]), 5), through 2039.


def compound_index(growth_by_year: dict[int, float]) -> dict[int, float]:
    """Mirror create_economic_assumption_indices' compounding rule.

    Base 1.0 at the series' earliest year; each later year multiplies by
    (1 + that year's growth), rounded to 5 places (matching upstream, so
    the fallback reproduces the engine-built values exactly). Growth after
    the last listed year is held at the last listed value (a dated
    parameter applies from its effective date onward), through 2039.
    """
    if not growth_by_year:
        raise ValueError("growth_by_year is empty")
    first = min(growth_by_year)
    values = {first: 1.0}
    last_growth = 0.0
    for year in range(first + 1, _INDEX_LAST_YEAR + 1):
        last_growth = growth_by_year.get(year, last_growth)
        values[year] = round(values[year - 1] * (1 + last_growth), 5)
    return values


def yoy_growth_yaml_candidates() -> list[Path]:
    """Possible locations of policyengine-uk's yoy_growth.yaml, best first."""
    rel = "parameters/gov/economic_assumptions/yoy_growth.yaml"
    candidates: list[Path] = []
    try:
        import policyengine_uk

        candidates.append(Path(policyengine_uk.__file__).parent / rel)
    except Exception:
        pass
    # uv wheel cache (lets tests validate paths without a working engine).
    pattern = os.path.expanduser(
        f"~/.cache/uv/archive-v0/*/policyengine_uk/{rel}"
    )
    candidates.extend(Path(p) for p in sorted(glob.glob(pattern)))
    return [p for p in candidates if p.exists()]


def load_yoy_growth(series: str = "average_earnings",
                    path: Path | None = None) -> dict[int, float]:
    """Read one obr.* yoy_growth series from policyengine-uk's YAML."""
    import yaml

    if path is None:
        found = yoy_growth_yaml_candidates()
        if not found:
            raise FileNotFoundError(
                "policyengine-uk yoy_growth.yaml not found (engine not "
                "installed and no wheel cache); pass path= explicitly"
            )
        path = found[0]
    tree = yaml.safe_load(Path(path).read_text())
    values = tree["obr"][series]["values"]
    return {int(str(k)[:4]): float(v) for k, v in values.items()}


def _engine_index_values(series: str) -> dict[int, float]:
    """Read the built derived index from the loaded policyengine-uk system."""
    from policyengine.tax_benefit_models.uk import uk_latest

    param = uk_latest.get_parameter(
        f"gov.economic_assumptions.indices.obr.{series}"
    )
    values = getattr(param, "values", None) or {}
    out = {}
    for key, value in dict(values).items():
        out[int(str(key)[:4])] = float(value)
    if not out:
        raise RuntimeError(f"engine returned no values for {param!r}")
    return out


def baseline_index_values(
    series: str = "average_earnings",
) -> tuple[dict[int, float], str]:
    """Baseline derived-index values, with the source used.

    Engine path first; the YAML-compounding fallback mirrors the engine's
    own construction rule, so both agree to the engine's 5-decimal rounding
    (tests cross-check this where the engine imports).
    """
    try:
        return _engine_index_values(series), "engine"
    except Exception:
        return compound_index(load_yoy_growth(series)), "yaml-compound"
