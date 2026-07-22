"""Macro -> micro EconomicAssumptions overlay (PolicyEngine/macro#11).

Carries the OG-UK model's long-run price changes (wages, labour supply,
interest rates) into the PolicyEngine microsimulation, so a "dynamic"
population score is the ordinary static score run under macro-adjusted
economic inputs.

DOUBLE-COUNTING INVARIANT
-------------------------
The overlay carries only the reform/baseline RATIO from the macro model,
never a level. The baseline microsim run uses the stock inputs — which
already embed the OBR forecast the OG-UK baseline is calibrated to — so the
static effect of the reform is never counted twice: a no-op macro result
(w_reform == w_baseline) produces NO modifier at all and dynamic scoring
reduces exactly to static scoring. Tests assert this.

APPLICATION MECHANISM: DIRECT INPUT SCALING (not parameter overrides)
---------------------------------------------------------------------
The factor is applied by scaling the employment-income input arrays of the
REFORM simulation only, through the engine's supported hook: a
``policyengine.core.Dynamic`` with a ``simulation_modifier`` callable,
which ``PolicyEngineUKLatest.run`` invokes on the underlying
``policyengine_uk.Microsimulation`` after construction and before any
calculation (policyengine tax_benefit_models/uk/model.py). The modifier
uses ``Simulation.set_input`` (policyengine_core) on the populated
employment-income input holder.

WHY NOT A PARAMETER OVERLAY ON THE UPRATING INDICES (empirical finding,
2026-07-20): an earlier version of this module overrode the derived index
``gov.economic_assumptions.indices.obr.average_earnings`` in the reform
dict. That mechanism is DEAD in population runs: the per-year population
datasets (e.g. enhanced_frs_2023_24-year-2026) are pre-uprated at dataset
BUILD time — ``policyengine.tax_benefit_models.uk.datasets.create_datasets``
materialises ``sim.dataset[year]`` once under stock parameters into the
per-year .h5, and ``PolicyEngineUKLatest.run`` feeds those stored input
arrays straight into ``UKSingleYearDataset`` — so simulation-time uprating
parameters are never consulted for input variables. Verified against the
production engine: two population_reform_impact calls overriding the 2026
index by x0.99 (1.66561) and then drastically to 0.84 BOTH returned exactly
zero everywhere (£0.0bn, 0 winners, 0 losers, all deciles 0.0).
"""

from __future__ import annotations

from pydantic import BaseModel

# Input variables carrying employment income, in the order tried: after
# policyengine_uk's Simulation.__init__ move_values step the dataset's
# employment income lives in employment_income_before_lsr;
# employment_income itself is kept as a fallback for data layouts that
# did not go through that step.
SCALED_INPUT_VARIABLES = ("employment_income_before_lsr", "employment_income")

# The dynamic score refuses user reforms under this prefix: uprating
# overrides there are silently dead in population runs (see module
# docstring), so accepting them would ship a plausible-looking no-op.
OVERLAY_PARAM_PREFIX = "gov.economic_assumptions."


class EconomicAssumptions(BaseModel):
    """Macro-model price changes expressed as microsim input adjustments.

    Steady-state comparative statics: the factors are LONG-RUN level shifts
    (reform/baseline ratios), applied flat from ``start_year`` with no
    transition dynamics — that assumption is spelled out in ``notes`` and
    must be carried into any ScoreResult built from this object.

    v1 scope (deliberately narrow, and honest about it):
    - ``earnings_factor`` scales the employment-income input arrays of the
      reform simulation only (see ``input_scaling_modifier``).
    - Other earnings-linked inputs (self-employment/mixed income, pension
      income) are NOT scaled in v1. This is a v1 INCIDENCE CHOICE, not a
      distinction OG identifies: the OG ``w`` is the price of an effective
      labour unit (its calibration blends employment and self-employment
      income) and ``L`` is effective labour, not raw hours — restricting
      the pass-through to employment income keeps the applied margin
      narrow and explicit rather than asserting broader incidence.
    - ``labour_supply_factor`` is REPORTED in assumptions/caveats but not
      allocated to any input: an aggregate hours change has no
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
        try:
            base = og_payload["baseline_steady_state_model_units"]
            ref = og_payload["reform_steady_state_model_units"]
            start_year = int(og_payload["start_year"])
            (base["w"], base["L"], base["r"], ref["w"], ref["L"], ref["r"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                "og_payload is not an og-score result (missing field "
                f"{e}); pass the unmodified output of "
                "`pe-macro og-score --json`"
            ) from e
        for name in ("w", "L"):
            for side, vals in (("baseline", base), ("reform", ref)):
                try:
                    v = float(vals[name])
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"OG {side} steady state has non-numeric "
                        f"{name}={vals[name]!r}; pass the unmodified output "
                        "of `pe-macro og-score --json`"
                    ) from e
                if not (v and v > 0) or v != v or v in (float("inf"),):
                    raise ValueError(
                        f"OG {side} steady state has non-positive/non-finite "
                        f"{name}={v!r}; refusing to build an overlay from a "
                        "degenerate solve"
                    )
        earnings_factor = ref["w"] / base["w"]
        labour_supply_factor = ref["L"] / base["L"]
        for label, f in (("earnings", earnings_factor),
                         ("labour-supply", labour_supply_factor)):
            if not 0.5 <= f <= 2.0:
                raise ValueError(
                    f"implausible steady-state {label} ratio {f:.4f} "
                    "(outside [0.5, 2.0]) — inspect the OG solve rather "
                    "than applying it as an overlay"
                )
        return cls(
            source=(
                "OG-UK overlapping generations (steady state), "
                "pooled ages, single representative sector"
            ),
            start_year=start_year,
            earnings_factor=earnings_factor,
            labour_supply_factor=labour_supply_factor,
            interest_rate_baseline=base["r"],
            interest_rate_reform=ref["r"],
            notes=[
                f"steady-state overlay: long-run factor applied uniformly "
                f"from {start_year}; no transition dynamics",
                "overlay carries only the reform/baseline ratio, so the "
                "static effect embedded in the stock inputs is never "
                "counted twice",
            ],
        )

    def input_scaling_modifier(self):
        """The overlay as a simulation modifier, or None for a null result.

        Returns a callable suitable for ``policyengine.core.Dynamic(
        simulation_modifier=...)``: it multiplies the first POPULATED
        employment-income input variable (SCALED_INPUT_VARIABLES order) by
        ``earnings_factor`` on every known period, via the engine's
        supported ``set_input`` API. Input scaling is used because
        parameter overrides on the uprating indices are dead in population
        runs (pre-uprated per-year datasets; see module docstring).

        Invariant: a no-op macro result (earnings_factor == 1) returns
        None — the caller attaches NO dynamic, so the reform simulation is
        bit-identical to the static one.
        """
        if self.earnings_factor == 1.0:
            return None
        factor = self.earnings_factor

        def modifier(microsim):
            for name in SCALED_INPUT_VARIABLES:
                holder = microsim.get_holder(name)
                periods = list(holder.get_known_periods())
                if not periods:
                    continue
                for period in periods:
                    values = holder.get_array(period)
                    holder.delete_arrays(period)
                    microsim.set_input(name, period, values * factor)
                return microsim
            raise RuntimeError(
                "EconomicAssumptions overlay found no populated "
                f"employment-income input among {SCALED_INPUT_VARIABLES}; "
                "the earnings factor cannot be applied — refusing to "
                "return a silently static result as a dynamic one."
            )

        return modifier

    def assumption_strings(self) -> list[str]:
        return [
            f"macro source: {self.source}",
            *self.notes,
            "application: employment-income input arrays of the reform "
            f"simulation scaled by {self.earnings_factor} (direct input "
            "scaling; uprating-parameter overrides are dead on pre-built "
            "per-year datasets)",
        ]

    def caveat_strings(self) -> list[str]:
        labour_pct = 100.0 * (self.labour_supply_factor - 1.0)
        return [
            f"aggregate effective-labour change {labour_pct:+.2f}% not "
            "distributionally allocated in v1 (labour_supply_factor is "
            "reported, not applied to any input; OG's L is effective "
            "labour units, not raw hours)",
            "earnings factor applied to employment income only; "
            "self-employment/mixed income and pension income are not "
            "adjusted in v1",
            "no price-level overlay: the OG model is real",
        ]
