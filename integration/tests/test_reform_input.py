"""Reform-input validation: the shapes we accept and the errors we return.

These are the developer-experience guarantees for the `reform`, `country` and
`people` arguments shared by the CLI and the MCP tools. Nothing here imports
PolicyEngine — validation deliberately runs before the heavy import so bad
input fails fast with an actionable ValueError instead of a raw pydantic or
strptime dump.
"""

import pytest

from policyengine_macro import core

RATE = "gov.hmrc.income_tax.rates.uk[0].rate"


# --- accepted shapes -------------------------------------------------------

def test_flat_value_passes_through_unchanged():
    """A scalar stays a scalar: PolicyEngine applies it from {year}-01-01."""
    assert core.validate_reform({RATE: 0.21}) == {RATE: 0.21}


def test_flat_integer_and_bool_values_accepted():
    reform = {"gov.a.amount": 3000, "gov.b.enabled": True}
    assert core.validate_reform(reform) == reform


def test_single_date_key_accepted():
    reform = {RATE: {"2026-01-01": 0.21}}
    assert core.validate_reform(reform) == reform


def test_multiple_dates_on_one_parameter_accepted():
    reform = {RATE: {"2026-01-01": 0.21, "2028-01-01": 0.22}}
    assert core.validate_reform(reform) == reform


def test_mixed_flat_and_dated_parameters_accepted():
    reform = {RATE: 0.21, "gov.hmrc.cgt.basic_rate": {"2026-04-06": 0.20}}
    assert core.validate_reform(reform) == reform


# --- date ranges: explicitly rejected, not faked ---------------------------

@pytest.mark.parametrize(
    "key", ["2026-01-01.2029-12-31", "2026-01-01:2029-12-31"]
)
def test_date_range_key_rejected_with_actionable_message(key):
    """PolicyEngine reform values are open-ended (ParameterValue.end_date is
    None), so a range cannot be honoured. We must say so, name the end date we
    cannot express, and show the correct single-date form."""
    with pytest.raises(ValueError) as exc:
        core.validate_reform({RATE: {key: 0.21}})
    msg = str(exc.value)
    assert key in msg
    assert "2029-12-31" in msg
    assert '{"2026-01-01": 0.21}' in msg
    assert "range" in msg.lower()
    # No raw upstream error leaks through.
    assert "unconverted data remains" not in msg
    assert "validation error" not in msg


def test_date_range_error_suggests_per_year_scoring():
    with pytest.raises(ValueError, match="year"):
        core.validate_reform({RATE: {"2026-01-01.2029-12-31": 0.21}})


# --- other malformed reform inputs -----------------------------------------

def _assert_helpful(msg):
    """Every reform error names the supported shapes and shows an example."""
    assert "flat value" in msg
    assert "effective date" in msg
    assert '"2026-01-01"' in msg


@pytest.mark.parametrize("bad", [None, {}])
def test_empty_reform_rejected(bad):
    with pytest.raises(ValueError) as exc:
        core.validate_reform(bad)
    assert "non-empty" in str(exc.value)
    _assert_helpful(str(exc.value))


def test_non_dict_reform_rejected():
    with pytest.raises(ValueError) as exc:
        core.validate_reform([RATE, 0.21])
    assert "non-empty" in str(exc.value)
    assert "got list" in str(exc.value)
    _assert_helpful(str(exc.value))


def test_malformed_date_key_rejected():
    with pytest.raises(ValueError) as exc:
        core.validate_reform({RATE: {"2026": 0.21}})
    msg = str(exc.value)
    assert "invalid date key '2026'" in msg
    assert "YYYY-MM-DD" in msg
    _assert_helpful(msg)


def test_non_numeric_value_rejected():
    with pytest.raises(ValueError) as exc:
        core.validate_reform({RATE: "0.21"})
    msg = str(exc.value)
    assert "must be a number" in msg
    _assert_helpful(msg)


def test_non_numeric_dated_value_rejected():
    with pytest.raises(ValueError) as exc:
        core.validate_reform({RATE: {"2026-01-01": "twenty-one percent"}})
    assert "must be a number" in str(exc.value)


def test_empty_dated_spec_rejected():
    with pytest.raises(ValueError) as exc:
        core.validate_reform({RATE: {}})
    assert "empty dict" in str(exc.value)


def test_non_string_parameter_path_rejected():
    with pytest.raises(ValueError) as exc:
        core.validate_reform({0: 0.21})
    assert "parameter-path strings" in str(exc.value)


def test_error_message_names_the_argument():
    with pytest.raises(ValueError, match="policy"):
        core.validate_reform({}, argument="policy")


# --- country / people ------------------------------------------------------

@pytest.mark.parametrize("missing", [None, ""])
def test_missing_country_gives_actionable_error(missing):
    """The bug: an omitted country returned a raw pydantic dump. It must now
    explain that there is no default and why."""
    with pytest.raises(ValueError) as exc:
        core.pe_household(missing, [{"age": 30}])
    msg = str(exc.value)
    assert "country is required" in msg
    assert "'uk' or 'us'" in msg
    assert "no default" in msg
    assert "validation error" not in msg


def test_bad_country_still_rejected():
    with pytest.raises(ValueError, match="must be 'uk' or 'us'"):
        core.pe_household("fr", [{"age": 30}])


def test_missing_people_gives_actionable_error():
    with pytest.raises(ValueError) as exc:
        core.pe_household("uk", None)
    msg = str(exc.value)
    assert "people is required" in msg
    assert "employment_income" in msg


def test_people_must_be_list_of_dicts():
    with pytest.raises(ValueError, match="list of person dicts"):
        core.pe_household("uk", ["age 30"])


# --- validation happens before the heavy PolicyEngine import ---------------

@pytest.mark.parametrize(
    "call",
    [
        lambda: core.pe_household("uk", [{"age": 30}], reform={"p": {"a.b": 1}}),
        lambda: core.pe_household_impact("uk", [{"age": 30}], reform={}),
        lambda: core.pe_population_impact("uk", reform={}),
        lambda: core.score_reform("uk", {}, "microsim"),
    ],
)
def test_bad_reform_fails_before_importing_policyengine(call, monkeypatch):
    monkeypatch.setattr(
        core,
        "_import_pe",
        lambda: pytest.fail("PolicyEngine imported despite invalid input"),
    )
    with pytest.raises(ValueError):
        call()


def test_score_reform_requires_a_known_model():
    with pytest.raises(ValueError) as exc:
        core.score_reform("uk", {RATE: 0.21}, model=None)
    msg = str(exc.value)
    assert "model is required" in msg
    assert "microsim" in msg


# --- the MCP tools expose the same behaviour -------------------------------

def test_mcp_household_tools_accept_omitted_arguments():
    """calculate_household / household_reform_impact must not raise a pydantic
    'Field required' dump when country or people are omitted — the arguments
    are optional at the schema level and validated by core."""
    from policyengine_macro import mcp_server

    # FastMCP versions differ: some wrap the function, some return it as-is.
    def _tool_fn(tool):
        return getattr(tool, "fn", tool)

    with pytest.raises(ValueError, match="country is required"):
        _tool_fn(mcp_server.calculate_household)()
    with pytest.raises(ValueError, match="reform must be a non-empty"):
        _tool_fn(mcp_server.household_reform_impact)(
            country="uk", people=[{"age": 30}]
        )
