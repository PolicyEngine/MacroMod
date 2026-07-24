"""Authoritative capability registry for routing and status surfaces.

This deliberately records capabilities, not model-level marketing badges. A
model can be production-ready for one use and inappropriate for another.
"""

from __future__ import annotations

from copy import deepcopy


MODELS = {
    "pe-microsim": {
        "display_name": "PolicyEngine tax-benefit microsimulation",
        "model_class": "microsimulation",
        "geography": ["uk", "us"],
        "question_types": ["household", "population", "policy_reform"],
        "inputs": ["household data", "population data", "parameter reform"],
        "outputs": ["taxes", "benefits", "net_income", "revenue", "distribution"],
        "cannot_answer": ["GDP", "inflation", "interest rates", "macro feedback"],
        "horizon": "single policy year",
        "access": ["hosted", "CLI", "Python"],
        "runtime": "sub-second household; minutes population",
        "uncertainty": "none for household arithmetic; survey/calibration uncertainty for population estimates",
        "status": "production-ready for selected household applications",
        "data_vintage": "country package and dataset dependent; recorded per run",
    },
    "og+microsim": {
        "display_name": "OG-UK overlay dynamic scoring (og+microsim)",
        "model_class": "olg-ge overlay on microsimulation",
        "geography": ["uk"],
        "question_types": ["dynamic_scoring", "policy_reform"],
        "inputs": ["parameter reform", "optional pre-computed OG payload"],
        "outputs": ["revenue (dynamic)", "distribution", "distribution under GE feedback"],
        "cannot_answer": [
            "transition paths (steady-state factor applied flat)",
            "price-level effects (the OG model is real)",
            "distributional incidence of effective-labour changes (reported, not allocated)",
        ],
        "horizon": "one policy year under long-run steady-state assumptions",
        "access": ["CLI", "Python"],
        "runtime": "two OG steady-state solves (baseline cached; >10 min cold) + one microsim run",
        "uncertainty": "none quantified; steady-state comparative statics",
        "status": (
            "experimental; local-only (oguk excluded from the hosted image), "
            "and until PSLmodels/OG-UK#68 the OG solve needs its own "
            "environment — use the two-step --og-payload pipeline"
        ),
        "data_vintage": "PolicyEngine dataset + OG-UK packaged calibration inputs",
    },
    "obr-macro": {
        "display_name": "OBR macroeconometric emulator",
        "model_class": "macroeconometric",
        "geography": ["uk"],
        "question_types": ["economic_shock", "translated_policy_scenario"],
        "inputs": ["curated model-variable shock", "reviewed reform translation"],
        "outputs": ["gdp", "consumption", "investment"],
        "cannot_answer": ["arbitrary statutory reform incidence", "borrowing through the current adapter"],
        "horizon": "quarterly, typically 3-5 years",
        "access": ["hosted", "CLI", "Python"],
        "runtime": "seconds for raw shocks; minutes for translated reform scenarios",
        "uncertainty": "not comprehensive",
        "status": "validated for selected scenarios",
        "data_vintage": "March 2026 EFO baseline",
    },
    "boe-svar": {
        "display_name": "Bank of England structural VAR replication",
        "model_class": "structural VAR",
        "geography": ["uk"],
        "question_types": ["forecast", "economic_diagnosis"],
        "inputs": ["packaged quarterly macroeconomic data"],
        "outputs": ["GDP forecast", "inflation forecast", "identified shocks", "uncertainty ranges"],
        "cannot_answer": ["statutory policy reform effects"],
        "horizon": "quarterly short-run forecast",
        "access": ["hosted forecast and latest shocks", "CLI", "Python package for wider analysis"],
        "runtime": "minutes per estimation and identification run",
        "uncertainty": "posterior 68% and 90% intervals",
        "status": "validated replication for selected outputs",
        "estimation_sample": "1992Q1-2025Q1",
        "data_edge": "2026Q1",
        "data_vintage": "2026Q1 conditioning data; estimation ends 2025Q1",
    },
    "frb-us": {
        "display_name": "Federal Reserve FRB-US implementation",
        "model_class": "macroeconometric",
        "geography": ["us"],
        "question_types": ["economic_shock"],
        "inputs": ["reviewed FRB-US add-factor shock"],
        "outputs": ["GDP", "unemployment", "inflation", "prices", "federal funds rate"],
        "cannot_answer": ["PolicyEngine reforms", "model-consistent-expectations scenarios"],
        "horizon": "quarterly",
        "access": ["hosted raw shocks", "CLI with editable model checkout", "Python"],
        "runtime": "seconds to minutes",
        "uncertainty": "not comprehensive",
        "status": "validated software replication with scope limits",
        "data_vintage": "LONGBASE file from the installed frbus package",
    },
    "og-uk": {
        "display_name": "OG-UK overlapping generations model",
        "model_class": "overlapping-generations general equilibrium",
        "geography": ["uk"],
        "question_types": ["policy_reform", "structural_change"],
        "inputs": ["PolicyEngine parameter reform", "calibration parameters"],
        "outputs": ["GDP", "work", "saving", "capital", "wages", "interest rates", "debt"],
        "cannot_answer": ["short-run forecast", "fast hosted custom scenario"],
        "horizon": "long-run steady state; package also supports transition paths",
        "access": ["local CLI steady state", "Python package"],
        "runtime": "17+ minutes per steady-state solve; transition paths can take hours",
        "uncertainty": "sensitivity analysis not yet comprehensive",
        "status": "research prototype; calibrated counterfactual",
        "data_vintage": "OG-UK packaged calibration inputs",
    },
}

# Evidence is deliberately split into dimensions. A model can reproduce its
# reference software perfectly while still having limited independent evidence
# for forecasts or policy counterfactuals. These are categorical audit
# judgements, not a synthetic score that invites false precision.
QUALITY_LEVELS = {"strong", "moderate", "weak", "not_assessed", "not_applicable"}
QUALITY_DIMENSIONS = {
    "implementation_fidelity",
    "predictive_validation",
    "identification_robustness",
    "policy_counterfactual_validity",
    "uncertainty_calibration",
    "vintage_reproducibility",
}


def _quality(level: str, evidence: str, next_gate: str) -> dict:
    return {"level": level, "evidence": evidence, "next_gate": next_gate}


MODEL_QUALITY = {
    "obr-macro": {
        "implementation_fidelity": _quality(
            "moderate",
            "Anchored GDP/consumption reproduce the March 2026 EFO within 1%, "
            "but passthrough and inactive equations limit equation coverage.",
            "Exercise every published behavioural equation and eliminate or "
            "explicitly scope every inactive channel.",
        ),
        "predictive_validation": _quality(
            "weak",
            "Free-running GDP and consumption MAPE are 5.75% and 9.56%; the "
            "anchored fit is by construction.",
            "Pass rolling-origin historical-vintage tests against simple "
            "benchmarks and first-release outturns.",
        ),
        "identification_robustness": _quality(
            "not_applicable",
            "The model is an equation-based emulator rather than an identified "
            "structural-shock model.",
            "Keep this dimension explicitly not applicable.",
        ),
        "policy_counterfactual_validity": _quality(
            "weak",
            "One income-tax costing is independently compared with HMRC; trade, "
            "labour, prices and parts of household income remain constrained.",
            "Validate a frozen suite of fiscal shocks against independent "
            "official costings and published multiplier ranges.",
        ),
        "uncertainty_calibration": _quality(
            "weak",
            "Results are point scenarios without comprehensive uncertainty over "
            "add factors, closures or missing-input proxies.",
            "Publish sensitivity envelopes for judgement, closure and bridge "
            "assumptions.",
        ),
        "vintage_reproducibility": _quality(
            "moderate",
            "The live March 2026 baseline and November 2025 paper vintage are "
            "labelled, but a multi-vintage archive is not yet a test fixture.",
            "Archive source hashes and reproduce at least three historical EFO "
            "vintages end to end.",
        ),
    },
    "boe-svar": {
        "implementation_fidelity": _quality(
            "strong",
            "Zero/sign restrictions and decomposition identities are tested on "
            "real data to numerical precision.",
            "Keep all exact invariants hard-gated for every specification.",
        ),
        "predictive_validation": _quality(
            "weak",
            "Against a no-change random walk the model looked strong on CPI "
            "(0.63 at h=1), but a driftless walk on a trending log level is "
            "too weak a benchmark: against a random walk WITH DRIFT the CPI "
            "ratio becomes 0.83 at h=1 and 1.03 at h=8, i.e. no better than "
            "naive. Bank Rate is the one series that improves under the "
            "harder benchmark (0.79 at h=1, p=0.018) and is the defensible "
            "forecasting claim. UK GDP is not distinguishable from either "
            "benchmark (p=0.38-0.67), and excluding six Covid-target origins "
            "its ratio falls to 0.77. The frozen-edge run gives 0.32pp RMSE "
            "from a single origin.",
            "Score the predictive densities rather than point forecasts, "
            "report rolling interval coverage, and re-run once the estimation "
            "sample extends past the Covid dummies.",
        ),
        "identification_robustness": _quality(
            "moderate",
            "Headline FEVD shares replicate the paper in the weighted production "
            "run, but proxy world data and undisclosed source settings matter.",
            "Show conclusions across lag, prior, proxy-data and weighting grids "
            "with effective-sample-size diagnostics.",
        ),
        "policy_counterfactual_validity": _quality(
            "not_applicable",
            "The model diagnoses shocks and forecasts; it does not score statutory "
            "policy reforms.",
            "Continue to refuse reform-scoring requests.",
        ),
        "uncertainty_calibration": _quality(
            "moderate",
            "Posterior 68% and 90% intervals are produced, but empirical coverage "
            "has been checked over only seven forecast quarters.",
            "Report rolling empirical coverage and proper predictive scores.",
        ),
        "vintage_reproducibility": _quality(
            "moderate",
            "The estimation sample and conditioning edge are recorded, while key "
            "internal Bank world aggregates require public proxies.",
            "Freeze input manifests and retain both real-time and revised vintages.",
        ),
    },
    "frb-us": {
        "implementation_fidelity": _quality(
            "strong",
            "The baseline and four like-for-like scenarios (monetary, fiscal "
            "egfe, tax trp, non-inertial Taylor) match LONGBASE and pyfrbus at "
            "the reference solver's numerical noise floor.",
            "Extend like-for-like gates across further official demos, closures "
            "and recodes; add the MCE expectations path.",
        ),
        "predictive_validation": _quality(
            "not_assessed",
            "LONGBASE is an illustrative tracking baseline, not an official Fed "
            "forecast, and no historical forecast evaluation is published here.",
            "Run vintage-preserving pseudo-out-of-sample forecast evaluation.",
        ),
        "identification_robustness": _quality(
            "not_applicable",
            "FRB/US is a large behavioural equation model, not a structural VAR "
            "identified by sign or zero restrictions.",
            "Keep this dimension explicitly not applicable.",
        ),
        "policy_counterfactual_validity": _quality(
            "moderate",
            "Selected monetary and fiscal multipliers lie in published ranges, but "
            "only VAR expectations are supported.",
            "Implement and cross-validate model-consistent expectations before "
            "forward-guidance or permanent-policy use.",
        ),
        "uncertainty_calibration": _quality(
            "weak",
            "A seeded joint historical-residual bootstrap now exists upstream, but "
            "the public run surface and published experiments remain deterministic.",
            "Review residual windows and closures, expose the stochastic surface, "
            "and publish coverage and convergence diagnostics.",
        ),
        "vintage_reproducibility": _quality(
            "moderate",
            "Model and LONGBASE archives are independently SHA-256 gated because "
            "the Board updates their pages and artifacts on separate schedules.",
            "Retain and test multiple historical model/data artifact pairs.",
        ),
    },
}

for _model_id, _model in MODELS.items():
    _model["quality"] = deepcopy(MODEL_QUALITY.get(_model_id, {
        dimension: _quality(
            "not_assessed",
            "Outside the scope of the current three-model audit.",
            "Complete a model-specific evidence review before assigning a level.",
        )
        for dimension in QUALITY_DIMENSIONS
    }))

REQUIRED_CAPABILITY_FIELDS = {
    "display_name", "model_class", "geography", "question_types", "inputs",
    "outputs", "cannot_answer", "horizon", "access", "runtime", "uncertainty",
    "status", "data_vintage", "quality",
}


def validate_registry(registry: dict | None = None) -> None:
    """Fail fast when a model bypasses the public capability contract."""
    registry = MODELS if registry is None else registry
    for model_id, model in registry.items():
        missing = REQUIRED_CAPABILITY_FIELDS - set(model)
        if missing:
            raise ValueError(f"{model_id} missing capability fields: {sorted(missing)}")
        for field in (
            "geography", "question_types", "inputs", "outputs", "cannot_answer", "access"
        ):
            if not isinstance(model[field], list) or not model[field]:
                raise ValueError(f"{model_id}.{field} must be a non-empty list")
        quality = model["quality"]
        if set(quality) != QUALITY_DIMENSIONS:
            raise ValueError(
                f"{model_id}.quality must contain exactly "
                f"{sorted(QUALITY_DIMENSIONS)}"
            )
        for dimension, assessment in quality.items():
            if assessment.get("level") not in QUALITY_LEVELS:
                raise ValueError(
                    f"{model_id}.quality.{dimension}.level must be one of "
                    f"{sorted(QUALITY_LEVELS)}"
                )
            for field in ("evidence", "next_gate"):
                if not assessment.get(field):
                    raise ValueError(
                        f"{model_id}.quality.{dimension}.{field} is required"
                    )


validate_registry()


def list_capabilities() -> list[dict]:
    return [{"model_id": model_id, **deepcopy(data)} for model_id, data in MODELS.items()]


def get_status(model_id: str) -> dict:
    if model_id not in MODELS:
        raise ValueError(f"unknown model_id {model_id!r}; choose one of {sorted(MODELS)}")
    return {"model_id": model_id, **deepcopy(MODELS[model_id])}


def recommend(
    question_type: str,
    country: str = "uk",
    needs_distribution: bool = False,
    horizon: str | None = None,
) -> dict:
    """Deterministic router; it never invents an unsupported model mapping."""
    country = country.lower()
    candidates = []
    for model_id, model in MODELS.items():
        if country not in model["geography"]:
            continue
        if question_type not in model["question_types"]:
            continue
        if needs_distribution and model_id not in ("pe-microsim", "og+microsim"):
            continue
        candidates.append(model_id)
    return {
        "question_type": question_type,
        "country": country,
        "needs_distribution": needs_distribution,
        "horizon": horizon,
        "primary_model": candidates[0] if candidates else None,
        "candidate_models": candidates,
        "warning": None if candidates else (
            "No registered model supports this request. Do not infer a mapping; "
            "refine the question or add an explicitly reviewed capability."
        ),
        "details": [get_status(model_id) for model_id in candidates],
    }
