from policyengine_macro import capabilities


def test_registry_has_exact_integrated_models():
    assert set(capabilities.MODELS) == {
        "pe-microsim", "obr-macro", "boe-svar", "frb-us", "og-uk"
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
