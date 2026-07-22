from policyengine_macro import capabilities


def test_registry_has_exact_integrated_models():
    assert set(capabilities.MODELS) == {
        "pe-microsim", "obr-macro", "boe-svar", "frb-us", "og-uk",
        "og+microsim",
    }


def test_og_is_uk_only_and_experimental():
    og = capabilities.get_status("og-uk")
    assert og["geography"] == ["uk"]
    assert "research prototype" in og["status"]


def test_router_refuses_unsupported_mapping():
    result = capabilities.recommend("forecast", country="us")
    assert result["primary_model"] is None
    assert result["warning"]


def test_router_selects_distribution_model():
    result = capabilities.recommend(
        "policy_reform", country="uk", needs_distribution=True
    )
    assert result["primary_model"] == "pe-microsim"


def test_every_model_declares_adapter_acceptance_metadata():
    capabilities.validate_registry()
    for model in capabilities.list_capabilities():
        assert model["inputs"]
        assert model["outputs"]
        assert model["data_vintage"]
        assert model["cannot_answer"]


def test_quality_contract_separates_fidelity_from_economic_evidence():
    obr = capabilities.get_status("obr-macro")["quality"]
    svar = capabilities.get_status("boe-svar")["quality"]
    frbus = capabilities.get_status("frb-us")["quality"]

    assert set(obr) == capabilities.QUALITY_DIMENSIONS
    assert obr["predictive_validation"]["level"] == "weak"
    assert svar["identification_robustness"]["level"] == "moderate"
    assert frbus["implementation_fidelity"]["level"] == "strong"
    assert frbus["predictive_validation"]["level"] == "not_assessed"


def test_quality_assessments_are_explanatory_not_numeric_scores():
    for model in capabilities.list_capabilities():
        for assessment in model["quality"].values():
            assert assessment["level"] in capabilities.QUALITY_LEVELS
            assert assessment["evidence"]
            assert assessment["next_gate"]
            assert "score" not in assessment


def test_distribution_routing_keeps_dynamic_member():
    """Reverting the needs_distribution filter to pe-microsim-only must
    fail here, not pass silently."""
    rec = capabilities.recommend(
        "policy_reform", country="uk", needs_distribution=True
    )
    assert "og+microsim" in rec["candidate_models"]
