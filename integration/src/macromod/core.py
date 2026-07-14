"""Model adapters. Single source of truth for the CLI and the MCP server.

Every public function returns plain dicts/lists that are directly JSON
serialisable.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BOE_VAR_REPO = Path("/Users/janansadeqian/boe-var-model")

# Column indices in the boe_var 8-variable dataset.
_I_CPI, _I_GDP = 5, 7
# The 6 identified shocks (indices into SHOCK_NAMES; 3 and 7 are unidentified).
_IDENTIFIED = [0, 1, 2, 4, 5, 6]


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
            "Install it with: pip install -e /Users/janansadeqian/obr-macroeconomic-model"
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
            "Install it with: pip install -e /Users/janansadeqian/boe-var-model"
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


def obr_score_reform(
    var: str,
    shock: float,
    periods: int = 12,
    name: str | None = None,
    investment_closure: bool = False,
) -> dict:
    """Score a policy reform with the OBR model emulator.

    Runs baseline vs shocked solves and returns per-period GDP deltas plus a
    headline cumulative GDP effect over the shocked periods.
    """
    run_reform = _import_obr()
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
        "gdp_growth_yoy": _series(_I_GDP),
        "cpi_inflation_yoy": _series(_I_CPI),
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

    shocks = []
    for j in _IDENTIFIED:
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
# Population-level scoring (pe.uk.ensure_datasets + Simulation) needs large
# dataset downloads (UK data requires a HUGGING_FACE_TOKEN); it is documented
# as planned, not implemented.

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
        "unit": "decimal rate (baseline 0.20)",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.income_tax.rates.uk[1].rate",
        "description": "Income tax higher rate",
        "unit": "decimal rate (baseline 0.40)",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.income_tax.allowances.personal_allowance.amount",
        "description": "Income tax personal allowance",
        "unit": "GBP per year (baseline 12,570)",
    },
    {
        "country": "uk",
        "path": "gov.dwp.universal_credit.means_test.reduction_rate",
        "description": "Universal Credit taper (earnings reduction) rate",
        "unit": "decimal rate (baseline 0.55)",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.national_insurance.class_1.rates.employee.main",
        "description": "Employee National Insurance main rate",
        "unit": "decimal rate (baseline 0.08)",
    },
    {
        "country": "uk",
        "path": "gov.hmrc.child_benefit.amount.eldest",
        "description": "Child Benefit weekly amount for the eldest child",
        "unit": "GBP per week",
    },
    {
        "country": "us",
        "path": "gov.irs.credits.ctc.amount.base[0].amount",
        "description": "Child Tax Credit base amount per child",
        "unit": "USD per year (baseline 2,000)",
    },
    {
        "country": "us",
        "path": "gov.irs.credits.ctc.amount.adult_dependent",
        "description": "CTC amount for adult dependents (credit for other dependents)",
        "unit": "USD per year (baseline 500)",
    },
    {
        "country": "us",
        "path": "gov.irs.income.bracket.rates.2",
        "description": "Federal income tax rate in the third bracket",
        "unit": "decimal rate (baseline 0.22)",
    },
    {
        "country": "us",
        "path": "gov.irs.deductions.standard.amount.JOINT",
        "description": "Standard deduction for joint filers",
        "unit": "USD per year",
    },
]


def pe_list_common_parameters() -> list[dict]:
    """Curated, verified PolicyEngine reform parameters with paths and units."""
    return [dict(p) for p in PE_PARAMETERS]


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
    pe = _import_pe()
    country = country.lower()
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

    Reads results/summary.md and results/forecast_summary.md from the
    boe-var-model checkout.
    """
    summary_path = BOE_VAR_REPO / "results" / "summary.md"
    fsummary_path = BOE_VAR_REPO / "results" / "forecast_summary.md"
    out: dict = {"source": str(BOE_VAR_REPO / "results")}

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
