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
        "outputs": ["taxes", "benefits", "net_income", "revenue", "distribution"],
        "cannot_answer": ["GDP", "inflation", "interest rates", "macro feedback"],
        "horizon": "single policy year",
        "access": ["hosted", "CLI", "Python"],
        "runtime": "sub-second household; minutes population",
        "uncertainty": "none for household arithmetic; survey/calibration uncertainty for population estimates",
        "status": "production-ready for selected household applications",
    },
    "obr-macro": {
        "display_name": "OBR macroeconometric emulator",
        "model_class": "macroeconometric",
        "geography": ["uk"],
        "question_types": ["economic_shock", "translated_policy_scenario"],
        "outputs": ["GDP", "consumption", "business_investment"],
        "cannot_answer": ["arbitrary statutory reform incidence", "borrowing through the current adapter"],
        "horizon": "quarterly, typically 3-5 years",
        "access": ["hosted", "CLI", "Python"],
        "runtime": "seconds for raw shocks; minutes for translated reform scenarios",
        "uncertainty": "not comprehensive",
        "status": "validated for selected scenarios",
    },
    "boe-svar": {
        "display_name": "Bank of England structural VAR replication",
        "model_class": "structural VAR",
        "geography": ["uk"],
        "question_types": ["forecast", "economic_diagnosis"],
        "outputs": ["GDP forecast", "inflation forecast", "identified shocks", "uncertainty ranges"],
        "cannot_answer": ["statutory policy reform effects"],
        "horizon": "quarterly short-run forecast",
        "access": ["hosted forecast and latest shocks", "CLI", "Python package for wider analysis"],
        "runtime": "minutes per estimation and identification run",
        "uncertainty": "posterior 68% and 90% intervals",
        "status": "validated replication for selected outputs",
        "estimation_sample": "1992Q1-2023Q2",
        "data_edge": "2026Q1",
    },
    "frb-us": {
        "display_name": "Federal Reserve FRB-US implementation",
        "model_class": "macroeconometric",
        "geography": ["us"],
        "question_types": ["economic_shock"],
        "outputs": ["GDP", "unemployment", "inflation", "prices", "federal funds rate"],
        "cannot_answer": ["PolicyEngine reforms", "model-consistent-expectations scenarios"],
        "horizon": "quarterly",
        "access": ["hosted raw shocks", "CLI with editable model checkout", "Python"],
        "runtime": "seconds to minutes",
        "uncertainty": "not comprehensive",
        "status": "validated software replication with scope limits",
    },
    "og-uk": {
        "display_name": "OG-UK overlapping generations model",
        "model_class": "overlapping-generations general equilibrium",
        "geography": ["uk"],
        "question_types": ["policy_reform", "structural_change"],
        "outputs": ["GDP", "work", "saving", "capital", "wages", "interest rates", "debt"],
        "cannot_answer": ["short-run forecast", "fast hosted custom scenario"],
        "horizon": "long-run steady state; package also supports transition paths",
        "access": ["local CLI steady state", "Python package"],
        "runtime": "17+ minutes per steady-state solve; transition paths can take hours",
        "uncertainty": "sensitivity analysis not yet comprehensive",
        "status": "research prototype; calibrated counterfactual",
    },
}


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
        if needs_distribution and model_id != "pe-microsim":
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
