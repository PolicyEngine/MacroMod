"""Model adapters. Single source of truth for the CLI and the MCP server.

Every public function returns plain dicts/lists that are directly JSON
serialisable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# Fallback checkout of boe-var-model for svar_summary when the boe_var
# package is not installed; unset means no fallback.
BOE_VAR_REPO_ENV = "MACROMOD_BOE_VAR_REPO"

# boe_var column names for the two headline series. Resolved to indices by
# name against the loaded dataset's own columns, so an upstream reorder can
# never silently mislabel a series.
_COL_CPI, _COL_GDP = "cpisa", "uk_gdp"


# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------

def _import_obr():
    try:
        import obr_macro  # noqa: F401
        from obr_macro import run_reform
    except ImportError as e:
        raise ImportError(
            "The OBR emulator package `obr_macro` is not importable. "
            "Install it with: pip install git+https://github.com/PolicyEngine/obr-macroeconomic-model"
        ) from e
    return run_reform


def _import_boe_var():
    try:
        import boe_var  # noqa: F401
        from boe_var import analysis, forecast
        from boe_var.bvar import BVAR
        from boe_var.data import load_data
        from boe_var.identification import ess, identify
    except ImportError as e:
        raise ImportError(
            "The UK SVAR package `boe_var` is not importable. "
            "Install it with: pip install git+https://github.com/PolicyEngine/boe-var-model"
        ) from e
    return analysis, forecast, BVAR, load_data, identify, ess


# ---------------------------------------------------------------------------
# OBR emulator adapters
# ---------------------------------------------------------------------------

# Curated list of commonly shocked exogenous policy variables (from the repo's
# reform_analysis examples and docs).
OBR_VARIABLES = [
    {
        "var": "CGG",
        "description": "Real government consumption",
        "units": "£m per quarter (e.g. 1250 = £5bn/year increase)",
        "investment_closure": False,
    },
    {
        "var": "TCPRO",
        "description": "Corporation tax (main) rate",
        "units": "rate change in decimal (e.g. -0.05 = 5pp cut from 25% to 20%)",
        "investment_closure": True,
    },
    {
        "var": "CGIPS",
        "description": "Nominal central government investment (feeds real GGI via the GGIPS/GGIDEF chain)",
        "units": "£m nominal per quarter (e.g. 3000 ≈ £2.5bn real per quarter)",
        "investment_closure": False,
    },
]


def obr_list_variables() -> list[dict]:
    """Commonly shocked policy variables with descriptions and units."""
    return [dict(v) for v in OBR_VARIABLES]


def obr_shock(
    var: str,
    shock: float,
    periods: int = 12,
    name: str | None = None,
    investment_closure: bool | None = None,
) -> dict:
    """Shock one OBR exogenous variable directly, in model units.

    The escape hatch under score_reform: no PolicyEngine reform translation,
    just a raw additive shock (£m per quarter for CGG/CGIPS, decimal rate
    change for TCPRO). Runs baseline vs shocked solves and returns per-period
    GDP deltas plus a headline cumulative GDP effect over the shocked periods.

    ``investment_closure`` defaults per variable from the curated list
    (e.g. True for TCPRO): a corporation-tax shock without the investment
    closure solves to all-zero deltas, which would read as a misleading
    "no effect" result rather than a mis-specified run.
    """
    run_reform = _import_obr()
    if investment_closure is None:
        known = {v["var"]: v["investment_closure"] for v in OBR_VARIABLES}
        investment_closure = known.get(var, False)
    if name is None:
        name = f"{var} shock {shock:+g}"
    df = run_reform(
        name=name,
        var=var,
        shock=float(shock),
        periods=int(periods),
        investment_closure=bool(investment_closure),
    )
    rows = [
        {
            "period": str(r.period),
            "delta_gdp_bn": round(float(r.delta_gdp_bn), 4),
            "pct_gdp": round(float(r.pct_gdp), 4),
            "delta_cons_m": round(float(r.delta_cons_m), 1),
            "delta_if_m": round(float(r.delta_if_m), 1),
        }
        for r in df.itertuples()
    ]
    shocked = rows[: int(periods)]
    peak = max(rows, key=lambda r: abs(r["pct_gdp"]))
    return {
        "name": name,
        "var": var,
        "shock": float(shock),
        "periods": int(periods),
        "investment_closure": bool(investment_closure),
        "results": rows,
        "cumulative_delta_gdp_bn_over_shock_periods": round(
            sum(r["delta_gdp_bn"] for r in shocked), 3
        ),
        "peak_pct_gdp": peak["pct_gdp"],
        "peak_period": peak["period"],
    }


# ---------------------------------------------------------------------------
# UK SVAR adapters
# ---------------------------------------------------------------------------

# In-process caches. Estimation + identification is the slow part; keyed by
# draws. Forecast results keyed by (horizons, draws).
_ESTIMATION_CACHE: dict[int, dict] = {}
_FORECAST_CACHE: dict[tuple[int, int], dict] = {}
_SHOCKS_CACHE: dict[int, dict] = {}


def _covid_dummies(index) -> np.ndarray:
    import pandas as pd

    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    D = np.zeros((len(index), len(quarters)))
    for j, q in enumerate(quarters):
        D[:, j] = (index == q).astype(float)
    return D


def _estimate(draws: int = 500, seed: int = 0) -> dict:
    """Estimate the BVAR and identify structural shocks. Cached by draws."""
    if draws in _ESTIMATION_CACHE:
        return _ESTIMATION_CACHE[draws]
    analysis, forecast, BVAR, load_data, identify, ess = _import_boe_var()
    import pandas as pd

    rng = np.random.default_rng(seed)
    df_full = load_data()
    df_full = df_full.loc[df_full.index >= pd.Period("1992Q1", "Q")]
    df_est = df_full.loc[df_full.index <= pd.Period("2023Q2", "Q")]
    dummies_est = _covid_dummies(df_est.index)
    dummies_full = _covid_dummies(df_full.index)

    model = BVAR(df_est.to_numpy(dtype=float), lags=4, dummies=dummies_est)
    posterior = model.sample_posterior(draws, seed=seed)
    triples = identify(posterior, rng=rng)
    if not triples:
        raise RuntimeError(
            f"No accepted identified draws out of {draws}; raise draws."
        )
    pairs = [(d, B) for d, B, _ in triples]
    w = np.array([t[2] for t in triples], dtype=float)
    out = {
        "df_full": df_full,
        "y_full": df_full.to_numpy(dtype=float),
        "dummies_full": dummies_full,
        "pairs": pairs,
        "weights": w,
        "n_accepted": len(pairs),
        "n_draws": draws,
        "ess": float(ess(w)),
        "rng": rng,
        "modules": (analysis, forecast),
    }
    _ESTIMATION_CACHE[draws] = out
    return out


def svar_forecast(horizons: int = 12, draws: int = 500) -> dict:
    """YoY UK GDP growth and CPI inflation forecast from the UK SVAR.

    Returns median and 68/90 percent bands per future quarter. Bands combine
    parameter and shock uncertainty. Cached in-process by (horizons, draws).
    """
    key = (int(horizons), int(draws))
    if key in _FORECAST_CACHE:
        return _FORECAST_CACHE[key]
    est = _estimate(int(draws))
    analysis, forecast = est["modules"]
    y_full = est["y_full"]
    rng = np.random.default_rng(1)

    tail = y_full[-4:]  # last 4 actual levels for the YoY transform
    yoy_paths, pw = [], []
    n_paths = 5
    for i, (d, _B) in enumerate(est["pairs"]):
        for _ in range(n_paths):
            path = forecast.sample_forecast(d, y_full, horizons=int(horizons), rng=rng)
            yoy_paths.append(forecast.yoy(np.vstack([tail, path])))
            pw.append(est["weights"][i])
    bands = analysis.aggregate(yoy_paths, weights=np.asarray(pw))

    last_q = est["df_full"].index[-1]
    quarters = [str(last_q + h) for h in range(1, int(horizons) + 1)]

    def _series(idx: int) -> list[dict]:
        return [
            {
                "quarter": quarters[h],
                "median": round(float(bands["median"][h, idx]), 3),
                "lo68": round(float(bands["lo68"][h, idx]), 3),
                "hi68": round(float(bands["hi68"][h, idx]), 3),
                "lo90": round(float(bands["lo90"][h, idx]), 3),
                "hi90": round(float(bands["hi90"][h, idx]), 3),
            }
            for h in range(int(horizons))
        ]

    out = {
        "model": "UK SVAR (BVAR, sign-identified)",
        "forecast_origin": str(last_q),
        "horizons": int(horizons),
        "draws": int(draws),
        "accepted_draws": est["n_accepted"],
        "ess": round(est["ess"], 1),
        "units": "YoY percent (4-quarter log difference of 100*log levels)",
        "gdp_growth_yoy": _series(list(est["df_full"].columns).index(_COL_GDP)),
        "cpi_inflation_yoy": _series(list(est["df_full"].columns).index(_COL_CPI)),
    }
    _FORECAST_CACHE[key] = out
    return out


def svar_latest_shocks(draws: int = 500) -> dict:
    """P(sign) of the 6 identified structural shocks in the latest data quarter."""
    key = int(draws)
    if key in _SHOCKS_CACHE:
        return _SHOCKS_CACHE[key]
    est = _estimate(key)
    analysis, forecast = est["modules"]
    from boe_var.analysis import SHOCK_NAMES

    y_full, dummies_full = est["y_full"], est["dummies_full"]

    resid_cache: dict[int, np.ndarray] = {}

    def resid_fn(draw):
        k = id(draw)
        if k not in resid_cache:
            resid_cache[k] = forecast.reduced_form_residuals(draw, y_full, dummies_full)
        return resid_cache[k]

    dist = forecast.shock_distribution_T(est["pairs"], resid_fn, weights=est["weights"])
    last_q = str(est["df_full"].index[-1])

    # Canonical upstream schema (boe_var's own index lists), not a label
    # heuristic; the contract test asserts SHOCK_NAMES stays aligned with it.
    from boe_var.analysis import UK_SHOCKS, WORLD_SHOCKS

    identified = sorted(WORLD_SHOCKS + UK_SHOCKS)

    shocks = []
    for j in identified:
        p = float(dist["p_pos"][j])
        p_dom = max(p, 1 - p)
        sign = "positive" if p >= 0.5 else "negative"
        conf = "clearly" if p_dom >= 0.8 else ("probably" if p_dom >= 0.65 else "ambiguously")
        shocks.append(
            {
                "shock": SHOCK_NAMES[j],
                "p_positive": round(p, 3),
                "p_negative": round(1 - p, 3),
                "reading": f"{SHOCK_NAMES[j]} shock was {conf} {sign} in {last_q} "
                           f"(P({'+' if p >= 0.5 else '−'}) = {p_dom:.2f}).",
            }
        )
    out = {
        "quarter": last_q,
        "draws": key,
        "accepted_draws": est["n_accepted"],
        "ess": round(est["ess"], 1),
        "shocks": shocks,
    }
    _SHOCKS_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# PolicyEngine microsimulation adapters (household calculator)
# ---------------------------------------------------------------------------
#
# policyengine imports its full UK+US country models on first import (~20s),
# so it is imported lazily inside the adapters, never at module level.
# Population-level scoring lives in pe_population_impact below; UK data
# requires a HUGGING_FACE_TOKEN for the first download.

def _import_pe():
    try:
        import policyengine as pe
    except ImportError as e:
        raise ImportError(
            "The `policyengine` package is not importable. "
            "Install it with: pip install policyengine"
        ) from e
    return pe


# Curated well-known reform parameters. Every path below has been verified to
# resolve through a calculate_household(reform=...) run.
PE_PARAMETERS = [
    {
        "country": "uk",
        "path": "gov.hmrc.income_tax.rates.uk[0].rate",
        "description": "Income tax basic rate (England/Wales/NI)",
        "unit": "decimal rate",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.income_tax.rates.uk[1].rate",
        "description": "Income tax higher rate",
        "unit": "decimal rate",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.income_tax.allowances.personal_allowance.amount",
        "description": "Income tax personal allowance",
        "unit": "GBP per year",
    },
    {
        "country": "uk",
        "path": "gov.dwp.universal_credit.means_test.reduction_rate",
        "description": "Universal Credit taper (earnings reduction) rate",
        "unit": "decimal rate",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.national_insurance.class_1.rates.employee.main",
        "description": "Employee National Insurance main rate",
        "unit": "decimal rate",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.child_benefit.amount.eldest",
        "description": "Child Benefit weekly amount for the eldest child",
        "unit": "GBP per week",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.cgt.basic_rate",
        "description": "Capital gains tax rate for basic-rate taxpayers",
        "unit": "decimal rate",
        "note": "valid reform path, but calculate_household does not compute CGT: household results will not move. Use with population_reform_impact / pe_population_impact.",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.cgt.higher_rate",
        "description": "Capital gains tax rate for higher/additional-rate taxpayers",
        "unit": "decimal rate",
        "note": "valid reform path, but calculate_household does not compute CGT: household results will not move. Use with population_reform_impact / pe_population_impact.",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.cgt.annual_exempt_amount",
        "description": "Capital gains tax annual exempt amount",
        "unit": "GBP per year",
        "note": "valid reform path, but calculate_household does not compute CGT: household results will not move. Use with population_reform_impact / pe_population_impact.",
    },
    {
        "country": "us",
        "path": "gov.irs.credits.ctc.amount.base[0].amount",
        "description": "Child Tax Credit base amount per child",
        "unit": "USD per year",
    },
    {
        "country": "us",
        "path": "gov.irs.credits.ctc.amount.adult_dependent",
        "description": "CTC amount for adult dependents (credit for other dependents)",
        "unit": "USD per year",
    },
    {
        "country": "us",
        "path": "gov.irs.income.bracket.rates.2",
        "description": "Federal income tax rate (bracket 2 — the 12% bracket)",
        "unit": "decimal rate",
    },
    {
        "country": "us",
        "path": "gov.irs.deductions.standard.amount.JOINT",
        "description": "Standard deduction for joint filers",
        "unit": "USD per year",
    },
]


def _pe_current_value(param):
    """The parameter value in force today, from upstream's own value history.

    The value with the latest start_date that is <= today. end_date is
    deliberately ignored: upstream builds parameter_values from a
    newest-first values_list, so each entry's end_date is the chronologically
    *previous* instant, not a validity end. "Today" is the server's local
    date (date.today()): around midnight a UK/US effective-date boundary can
    differ by a few hours — acceptable for listing metadata.
    """
    from datetime import date

    def _d(v):
        return v.date() if hasattr(v, "date") else v

    today = date.today()
    best_start, current = None, None
    for pv in param.parameter_values:
        start = _d(pv.start_date) if pv.start_date else None
        if start is not None and start <= today and (
            best_start is None or start > best_start
        ):
            best_start, current = start, pv.value
    return current


def pe_list_common_parameters(resolve: bool = True) -> list[dict]:
    """Curated PolicyEngine reform parameters, enriched from the live model.

    The path list is curated here, but baseline values, labels, and units are
    resolved from the policyengine package's own parameter tree at call time,
    so they can never go stale. A path that no longer resolves upstream is
    returned with ``"live": False`` and an explicit ``"live_error"`` instead
    of a silently wrong entry. ``resolve=False`` skips the (heavy) policyengine
    import and returns just the static catalogue.
    """
    out = [dict(p) for p in PE_PARAMETERS]
    if not resolve:
        return out
    try:
        pe = _import_pe()
    except Exception as e:
        # policyengine missing or its import broken: return the static
        # catalogue rather than failing the whole listing, but say so.
        for p in out:
            p["live"] = False
            p["live_error"] = f"{type(e).__name__}: {e}"
        return out
    for p in out:
        try:
            # getattr can return None on a base-only policyengine install
            # (no country models), so the whole lookup is guarded, not just
            # the parameter resolution.
            model = getattr(pe, p["country"]).model
            param = model.get_parameter(p["path"])
        except Exception as e:
            p["live"] = False
            p["live_error"] = f"{type(e).__name__}: {e}"
            continue
        p["live"] = True
        if param.label:
            p["label"] = param.label
        if param.unit:
            p["upstream_unit"] = param.unit
        p["baseline_value"] = _pe_jsonify_exact(_pe_current_value(param))
    return out


def _pe_jsonify_exact(value):
    """numpy scalar -> python scalar, NO rounding: policy values are exact
    (the display-rounding _pe_jsonify below would turn NI 0.1325 into 0.13)."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pe_jsonify(value):
    """numpy scalar -> python scalar; round floats for readability."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 2)
    return value


def _pe_entity_dict(mapping) -> dict:
    return {k: _pe_jsonify(v) for k, v in mapping.items()}


def _pe_run(country, people, year, reform, benunit, tax_unit, household):
    # Validate country before importing PolicyEngine so bad input fails fast
    # with a clear ValueError even where PE is not installed (matches
    # pe_population_impact and lets the wiring tests run without the heavy dep).
    country = country.lower()
    if country not in ("uk", "us"):
        raise ValueError(f"country must be 'uk' or 'us', got {country!r}")
    pe = _import_pe()
    if country == "uk":
        return pe.uk.calculate_household(
            people=people, benunit=benunit, household=household,
            year=int(year), reform=reform,
        )
    if country == "us":
        return pe.us.calculate_household(
            people=people, tax_unit=tax_unit, household=household,
            year=int(year), reform=reform,
        )
    raise ValueError(f"country must be 'uk' or 'us', got {country!r}")


def _pe_summary(country, result) -> dict:
    """Key headline outputs, robust across UK/US result shapes."""
    hh = result.household
    if country == "uk":
        return {
            "income_tax_by_person": [_pe_jsonify(p.income_tax) for p in result.person],
            "national_insurance_by_person": [
                _pe_jsonify(p.national_insurance) for p in result.person
            ],
            "household_net_income": _pe_jsonify(hh.hbai_household_net_income),
            "household_tax": _pe_jsonify(hh.household_tax),
            "household_benefits": _pe_jsonify(hh.household_benefits),
            "universal_credit": _pe_jsonify(result.benunit.universal_credit),
            "child_benefit": _pe_jsonify(result.benunit.child_benefit),
        }
    return {
        "federal_income_tax": _pe_jsonify(result.tax_unit.income_tax),
        "employee_payroll_tax": _pe_jsonify(result.tax_unit.employee_payroll_tax),
        "state_income_tax": _pe_jsonify(result.tax_unit.state_income_tax),
        "ctc": _pe_jsonify(result.tax_unit.ctc),
        "eitc": _pe_jsonify(result.tax_unit.eitc),
        "household_net_income": _pe_jsonify(hh.household_net_income),
        "household_tax": _pe_jsonify(hh.household_tax),
        "household_benefits": _pe_jsonify(hh.household_benefits),
    }


def pe_household(
    country: str,
    people: list[dict],
    year: int = 2026,
    reform: dict | None = None,
    benunit: dict | None = None,
    tax_unit: dict | None = None,
    household: dict | None = None,
) -> dict:
    """Calculate taxes and benefits for a custom household with PolicyEngine.

    country: 'uk' or 'us'. people: list of person dicts, e.g.
    [{"age": 35, "employment_income": 50000}]. reform: optional
    {"gov.param.path": value} dict. UK groups people into a benunit; US uses
    tax_unit (e.g. {"filing_status": "SINGLE"}) and household
    (e.g. {"state_code_str": "CA"}).
    """
    country = country.lower()
    result = _pe_run(country, people, year, reform, benunit, tax_unit, household)
    out = {
        "country": country,
        "year": int(year),
        "currency": "GBP" if country == "uk" else "USD",
        "reform": dict(reform) if reform else None,
        "summary": _pe_summary(country, result),
        "person": [_pe_entity_dict(p) for p in result.person],
        "household": _pe_entity_dict(result.household),
    }
    if country == "uk":
        out["benunit"] = _pe_entity_dict(result.benunit)
    else:
        out["tax_unit"] = _pe_entity_dict(result.tax_unit)
        out["spm_unit"] = _pe_entity_dict(result.spm_unit)
    return out


def pe_household_impact(
    country: str,
    people: list[dict],
    reform: dict,
    year: int = 2026,
    benunit: dict | None = None,
    tax_unit: dict | None = None,
    household: dict | None = None,
) -> dict:
    """Baseline vs reform for one household: what does this reform do to
    this family? Returns baseline and reform summaries plus their deltas."""
    if not reform:
        raise ValueError("reform must be a non-empty {parameter_path: value} dict")
    country = country.lower()
    base = _pe_summary(
        country, _pe_run(country, people, year, None, benunit, tax_unit, household)
    )
    ref = _pe_summary(
        country, _pe_run(country, people, year, reform, benunit, tax_unit, household)
    )

    def _delta(b, r):
        if isinstance(b, list):
            return [round(rv - bv, 2) for bv, rv in zip(b, r)]
        if isinstance(b, (int, float)) and isinstance(r, (int, float)):
            return round(r - b, 2)
        return None

    deltas = {k: _delta(base[k], ref[k]) for k in base}
    return {
        "country": country,
        "year": int(year),
        "currency": "GBP" if country == "uk" else "USD",
        "reform": dict(reform),
        "baseline": base,
        "with_reform": ref,
        "change": deltas,
        "net_income_change": deltas.get("household_net_income"),
    }


# ---------------------------------------------------------------------------
# PolicyEngine population-level reform scoring
# ---------------------------------------------------------------------------
#
# Runs the reform against the full representative household microdata
# (UK: enhanced FRS, ~125MB download from a private HuggingFace repo the
# first time — set HUGGING_FACE_TOKEN; US: CPS-based, public). Measured on
# the UK 2026 enhanced FRS (53,508 households): ~6s per simulation run,
# ~1.8GB peak RSS, ~92MB derived per-year .h5 on disk. The baseline
# simulation is cached in-process per (country, year, dataset), so repeat
# reform scores only pay one ~6s reform run.

PE_POP_DEFAULT_DATASET = {"uk": "enhanced_frs_2023_24", "us": None}

# (country, year, dataset_name) -> (dataset_obj, baseline_simulation)
_PE_POP_BASELINE_CACHE: dict[tuple[str, int, str | None], tuple] = {}


def _pe_pop_data_folder() -> str:
    import os

    return os.environ.get(
        "MACROMOD_PE_DATA_DIR",
        os.path.expanduser("~/.cache/macromod/policyengine-data"),
    )


def _pe_pop_extra_variables(country: str) -> dict:
    # gov_balance (tax minus spending, includes CGT and employer NI) is the
    # UK budget headline; it is not in the model's default output set.
    return {"household": ["gov_balance", "gov_tax"]} if country == "uk" else {}


def _pe_pop_dataset(pe, country: str, year: int, dataset: str | None):
    """Download/build (first call) and load the population dataset."""
    module = getattr(pe, country)
    name = dataset or PE_POP_DEFAULT_DATASET[country]
    kwargs = {"years": [int(year)], "data_folder": _pe_pop_data_folder()}
    if name is not None:
        kwargs["datasets"] = [name]
    try:
        datasets = module.ensure_datasets(**kwargs)
    except Exception as e:
        import os

        hint = ""
        if country == "uk" and not (
            os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
        ):
            hint = (
                " The UK population microdata lives in a private HuggingFace"
                " repo: set HUGGING_FACE_TOKEN to a token from an account"
                " with access to policyengine/policyengine-uk-data-private."
            )
        raise RuntimeError(
            f"Could not obtain the {country.upper()} population dataset "
            f"{name!r} for {year} ({type(e).__name__}: {e}).{hint}"
        ) from e
    for ds in datasets.values():
        if int(ds.year) == int(year):
            return ds
    raise RuntimeError(
        f"ensure_datasets returned no dataset for year {year}: "
        f"{sorted(datasets)}"
    )


def _pe_pop_baseline(country: str, year: int, dataset: str | None):
    """Dataset + baseline simulation, cached in-process."""
    key = (country, int(year), dataset)
    if key in _PE_POP_BASELINE_CACHE:
        return _PE_POP_BASELINE_CACHE[key]
    pe = _import_pe()
    from policyengine.core import Simulation

    ds = _pe_pop_dataset(pe, country, year, dataset)
    sim = Simulation(
        dataset=ds,
        tax_benefit_model_version=getattr(pe, country).model,
        extra_variables=_pe_pop_extra_variables(country),
    )
    sim.run()
    _PE_POP_BASELINE_CACHE[key] = (ds, sim)
    return ds, sim


def _pe_pop_sum(sim, variable: str) -> float:
    from policyengine.outputs.aggregate import Aggregate, AggregateType

    agg = Aggregate(
        simulation=sim, variable=variable,
        aggregate_type=AggregateType.SUM, entity="household",
    )
    agg.run()
    return float(agg.result)


def pe_population_impact(
    country: str = "uk",
    reform: dict | None = None,
    year: int = 2026,
    dataset: str | None = None,
) -> dict:
    """Score a reform against the whole population with PolicyEngine.

    Runs baseline and reform microsimulations over representative household
    microdata (UK: enhanced FRS; US: CPS-based) and returns the budgetary
    impact — the change in government revenue net of spending, in £bn/$bn
    per year (positive = the reform raises revenue) — plus income-decile
    impacts and winner/loser counts.

    reform is a flat {parameter_path: value} dict, e.g. equalising CGT with
    income tax rates: {"gov.hmrc.cgt.basic_rate": 0.20,
    "gov.hmrc.cgt.higher_rate": 0.40, "gov.hmrc.cgt.additional_rate": 0.45}.

    The baseline simulation is cached in-process per (country, year,
    dataset). UK data needs HUGGING_FACE_TOKEN on first download.
    """
    if not reform:
        raise ValueError("reform must be a non-empty {parameter_path: value} dict")
    country = country.lower()
    if country not in ("uk", "us"):
        raise ValueError(f"country must be 'uk' or 'us', got {country!r}")

    ds, base = _pe_pop_baseline(country, year, dataset)
    pe = _import_pe()
    from policyengine.core import Simulation
    from policyengine.outputs.decile_impact import calculate_decile_impacts

    ref = Simulation(
        dataset=ds,
        tax_benefit_model_version=getattr(pe, country).model,
        policy=dict(reform),
        extra_variables=_pe_pop_extra_variables(country),
    )
    ref.run()

    if country == "uk":
        budget_bn = (
            _pe_pop_sum(ref, "gov_balance") - _pe_pop_sum(base, "gov_balance")
        ) / 1e9
        budget_basis = (
            "change in gov_balance (all modelled taxes incl. CGT and "
            "employer NI, minus benefit spending)"
        )
    else:
        d_tax = _pe_pop_sum(ref, "household_tax") - _pe_pop_sum(base, "household_tax")
        d_ben = (
            _pe_pop_sum(ref, "household_benefits")
            - _pe_pop_sum(base, "household_benefits")
        )
        budget_bn = (d_tax - d_ben) / 1e9
        budget_basis = "change in household_tax minus change in household_benefits"

    net_income_change_bn = (
        _pe_pop_sum(ref, "household_net_income")
        - _pe_pop_sum(base, "household_net_income")
    ) / 1e9

    # Measure the change in household_net_income (which, unlike the UK's
    # HBAI income concept, moves under e.g. CGT reforms), grouped by the
    # baseline income decile.
    decile_kwargs = {
        "income_variable": "household_net_income",
        "entity": "household",
    }
    if country == "uk":
        decile_kwargs["decile_variable"] = "household_income_decile"
    deciles = calculate_decile_impacts(
        baseline_simulation=base, reform_simulation=ref, **decile_kwargs
    )
    decile_rows, winners, losers = [], 0.0, 0.0
    for d in deciles.outputs:
        avg_change = float(d.reform_mean - d.baseline_mean)
        decile_rows.append(
            {
                "decile": int(d.decile),
                "avg_income_change": round(avg_change, 2),
                # Change in the decile's mean income, in percent. (The
                # library's DecileImpact.relative_change is the mean of
                # per-household percent changes, which tiny-income outliers
                # dominate.)
                "relative_change_pct": round(
                    100 * avg_change / d.baseline_mean, 3
                ) if d.baseline_mean else None,
                "count_better_off": int(d.count_better_off),
                "count_worse_off": int(d.count_worse_off),
            }
        )
        winners += d.count_better_off
        losers += d.count_worse_off

    sym = "£" if country == "uk" else "$"
    return {
        "model": "PolicyEngine population microsimulation",
        "country": country,
        "year": int(year),
        "dataset": ds.name,
        "n_households": int(len(ds.data.household)),
        "currency": "GBP" if country == "uk" else "USD",
        "reform": dict(reform),
        "budgetary_impact_bn": round(budget_bn, 3),
        "budgetary_impact_basis": budget_basis,
        "headline": (
            f"The reform {'raises' if budget_bn >= 0 else 'costs'} "
            f"{sym}{abs(budget_bn):.1f}bn/year in {year}."
        ),
        "household_net_income_change_bn": round(net_income_change_bn, 3),
        "decile_impacts": decile_rows,
        "winners": int(winners),
        "losers": int(losers),
    }


# ---------------------------------------------------------------------------
# OG-UK overlapping-generations adapters (steady state only)
# ---------------------------------------------------------------------------
#
# Cheapest sensible configuration, which is oguk's own default: pooled ages
# (one tax function for all ages), single representative sector (M=1),
# steady-state comparison only (no transition path). A single solve —
# PolicyEngine microdata calibration + OG-Core SS solve — takes minutes of
# CPU, so the baseline solve is cached at module level per (start_year,
# max_iter) and reused across reform scores in the same process.

_OG_BASELINE_CACHE: dict[tuple[int, int], object] = {}

OG_DEFAULT_MAX_ITER = 250


def _import_oguk():
    try:
        from oguk import map_to_real_world, solve_steady_state
    except ImportError as e:
        raise ImportError(
            "The OG-UK package `oguk` is not importable. Install it with: "
            "pip install git+https://github.com/PSLmodels/OG-UK"
        ) from e
    return solve_steady_state, map_to_real_world


def _og_build_policy(reform: dict, start_year: int):
    """Build a PolicyEngine Policy from a {parameter_path: value} reform."""
    from datetime import datetime

    from policyengine.core import ParameterValue, Policy
    from policyengine.tax_benefit_models.uk import uk_latest

    start = datetime(int(start_year), 1, 1)
    return Policy(
        name=", ".join(f"{p} = {v}" for p, v in reform.items()),
        parameter_values=[
            ParameterValue(
                parameter=uk_latest.get_parameter(path),
                value=value,
                start_date=start,
            )
            for path, value in reform.items()
        ],
    )


def _og_solve(solve_fn, **kwargs):
    """Run an oguk steady-state solve, failing fast with an actionable error.

    oguk's calibration builds tax functions from the PolicyEngine
    enhanced-FRS UK microdata, downloaded via
    policyengine.tax_benefit_models.uk.ensure_datasets. Known environment
    failures are translated into clear messages: (a) no HuggingFace access to
    the dataset — set HUGGING_FACE_TOKEN; (b) policyengine-uk >= 2.89 renamed
    the dataset keys from enhanced_frs_2023_24_<year> to populace_uk_*, which
    makes oguk 0.3.0 (pinning policyengine-uk==2.88.0) fail with a KeyError.
    """
    try:
        return solve_fn(**kwargs)
    except KeyError as e:
        raise RuntimeError(
            "OG-UK calibration could not find the PolicyEngine enhanced-FRS "
            f"microdata dataset (missing dataset key {e}). Likely causes: "
            "(a) no access to the enhanced-FRS dataset on HuggingFace — set "
            "HUGGING_FACE_TOKEN to a token with access "
            "(policyengine.tax_benefit_models.uk.ensure_datasets downloads "
            "it); (b) incompatible policyengine-uk version — oguk 0.3.0 "
            "requires policyengine-uk==2.88.0 (>= 2.89 renamed the datasets)"
            ": pip install 'policyengine-uk==2.88.0'."
        ) from e


def _og_solve_baseline(start_year: int, max_iter: int, use_cache: bool = True):
    solve_steady_state, _ = _import_oguk()
    key = (int(start_year), int(max_iter))
    if use_cache and key in _OG_BASELINE_CACHE:
        return _OG_BASELINE_CACHE[key]
    ss = _og_solve(
        solve_steady_state, start_year=int(start_year), max_iter=int(max_iter)
    )
    if use_cache:
        _OG_BASELINE_CACHE[key] = ss
    return ss


def _og_ss_dict(ss) -> dict:
    """SteadyStateResult -> plain rounded dict (model units)."""
    return {k: round(float(v), 4) for k, v in ss.model_dump().items()}


def og_baseline(start_year: int = 2026, max_iter: int = OG_DEFAULT_MAX_ITER) -> dict:
    """Baseline long-run steady state of the OG-UK overlapping-generations model.

    Solves (or reuses a cached) baseline steady state under the simplest
    assumptions: pooled-age tax functions, single representative firm/sector.
    Returns model-unit aggregates (r, w, Y, K, L, C, I, G, tax_revenue, debt).
    """
    ss = _og_solve_baseline(start_year, max_iter)
    return {
        "model": "OG-UK overlapping generations (steady state)",
        "assumptions": "pooled ages, single representative sector, "
                       "steady state only",
        "start_year": int(start_year),
        "max_iter": int(max_iter),
        "steady_state_model_units": _og_ss_dict(ss),
    }


def og_score_reform(
    reform: dict,
    start_year: int = 2026,
    max_iter: int = OG_DEFAULT_MAX_ITER,
    baseline_cache: bool = True,
) -> dict:
    """Score a PolicyEngine parametric reform with the OG-UK OLG model.

    ``reform`` is the suite's shared reform shape — a flat
    {parameter_path: value} dict, the same one pe_population_impact and the
    household tools take. Solves baseline (module-level cached) and reform
    steady states, and maps the long-run changes to real-world £bn via
    oguk.map_to_real_world.
    """
    _validate_reform(reform)
    solve_steady_state, map_to_real_world = _import_oguk()
    policy = _og_build_policy(dict(reform), int(start_year))
    baseline_ss = _og_solve_baseline(start_year, max_iter, use_cache=baseline_cache)
    reform_ss = _og_solve(
        solve_steady_state, start_year=int(start_year), policy=policy,
        max_iter=int(max_iter),
    )
    impact = map_to_real_world(baseline_ss, reform_ss)
    imp = impact.model_dump()
    return {
        "model": "OG-UK overlapping generations (steady state)",
        "assumptions": "pooled ages, single representative sector, "
                       "long-run steady-state comparison (not a budget-window "
                       "costing)",
        "reform": dict(reform),
        "start_year": int(start_year),
        "impact": {
            "levels_bn": {k: imp[k] for k in
                          ("gdp", "consumption", "investment", "government",
                           "tax_revenue", "debt")},
            "changes_bn": {k: imp[k] for k in imp if k.endswith("_change")},
            "changes_pct": {k: imp[k] for k in imp if k.endswith("_pct")},
            "interest_rate": {"baseline": imp["r_baseline"],
                              "reform": imp["r_reform"]},
        },
        "baseline_steady_state_model_units": _og_ss_dict(baseline_ss),
        "reform_steady_state_model_units": _og_ss_dict(reform_ss),
    }


# ---------------------------------------------------------------------------
# Unified reform scoring across the suite
# ---------------------------------------------------------------------------

SCORE_MODELS = ("og", "obr")


def _validate_reform(reform) -> None:
    if not reform or not isinstance(reform, dict):
        raise ValueError(
            "reform must be a non-empty {parameter_path: value} dict, e.g. "
            '{"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}'
        )


def score_reform(
    country: str,
    reform: dict,
    model: str,
    start_year: int = 2026,
    **model_kwargs,
) -> dict:
    """Score a PolicyEngine reform with one of the suite's macro models.

    One reform vocabulary across the suite: ``reform`` is the same flat
    {parameter_path: value} dict the microsimulation tools take
    (pe_population_impact, pe_household_impact). Each model consumes it
    through its declared contract:

    - "og": the reform enters through PolicyEngine-estimated tax functions
      (long-run steady-state general-equilibrium comparison; UK only).
    - "obr": pending the microsim static-costing bridge
      (https://github.com/PolicyEngine/MacroMod/issues/9). Raw
      exogenous-variable shocks in model units remain available via
      obr_shock.
    """
    country = country.lower()
    _validate_reform(reform)
    if model == "og":
        if country != "uk":
            raise ValueError(
                "the OG member is UK-only (OG-UK); country must be 'uk'"
            )
        return og_score_reform(
            reform=reform, start_year=start_year, **model_kwargs
        )
    if model == "obr":
        raise NotImplementedError(
            "The OBR member does not take PolicyEngine reforms yet — the "
            "microsim static-costing bridge is tracked at "
            "https://github.com/PolicyEngine/MacroMod/issues/9. For a raw "
            "exogenous-variable shock in model units use obr_shock (MCP) or "
            "`macromod obr-shock` (CLI)."
        )
    raise ValueError(f"model must be one of {SCORE_MODELS}, got {model!r}")


# ---------------------------------------------------------------------------
# Cheap summary from the repo's committed results
# ---------------------------------------------------------------------------

def _parse_kv_lines(text: str) -> list[str]:
    return [ln.strip("- ").strip() for ln in text.splitlines()
            if ln.strip().startswith("- ")]


def _parse_md_table(text: str, heading: str) -> list[dict]:
    """Parse the first markdown pipe table after a heading line containing `heading`."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.startswith("#") and heading in ln)
    except StopIteration:
        return []
    rows, header = [], None
    for ln in lines[start + 1:]:
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            if header is None:
                header = cells
            else:
                rows.append(dict(zip(header, cells)))
        elif header is not None:
            break
    return rows


def svar_summary() -> dict:
    """Parse the SVAR repo's committed results (no estimation; instant).

    Reads summary.md and forecast_summary.md via boe_var.data.results_dir(),
    which resolves the repo checkout when present and the packaged snapshot
    otherwise (so this works from a bare pip install).
    """
    try:
        from boe_var.data import results_dir
        rdir = results_dir()
    except ImportError:
        env = os.environ.get(BOE_VAR_REPO_ENV)
        if not env:
            return {
                "error": (
                    "boe_var is not installed and MACROMOD_BOE_VAR_REPO is "
                    "not set; install the SVAR package (pip install git+"
                    "https://github.com/PolicyEngine/boe-var-model) or point "
                    "MACROMOD_BOE_VAR_REPO at a boe-var-model checkout"
                )
            }
        rdir = Path(env) / "results"
    summary_path = rdir / "summary.md"
    fsummary_path = rdir / "forecast_summary.md"
    out: dict = {"source": str(rdir)}

    if summary_path.exists():
        text = summary_path.read_text()
        out["replication"] = {
            "metadata": _parse_kv_lines(text.split("##")[0]),
            "fevd_1yr_headline": _parse_md_table(text, "FEVD at 1-year horizon"),
        }
    else:
        out["replication"] = {"error": f"missing {summary_path}"}

    if fsummary_path.exists():
        text = fsummary_path.read_text()
        out["forecast_revision"] = {
            "metadata": _parse_kv_lines(text.split("##")[0]),
            "latest_shock_signs": _parse_md_table(text, "P(sign)"),
            "composite_irf": _parse_kv_lines(
                text.split("Composite impulse response")[-1].split("##")[0]
            ),
        }
    else:
        out["forecast_revision"] = {"error": f"missing {fsummary_path}"}
    return out
