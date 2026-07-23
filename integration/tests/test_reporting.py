"""Common serialisable and human-readable result reports."""

import json

from click.testing import CliRunner

from policyengine_macro import core
from policyengine_macro import reporting
from policyengine_macro.cli import main


def _score():
    keys = ("gdp", "consumption", "investment", "government", "tax_revenue", "debt")
    return core._og_score_block({
        "reform": {"gov.example": 1},
        "assumptions": "test assumptions",
        "impact": {
            "levels_bn": {key: 100.0 for key in keys},
            "changes_bn": {f"{key}_change": 1.0 for key in keys},
            "changes_pct": {f"{key}_pct": 1.0 for key in keys},
        },
    })


def test_build_report_preserves_interpretation_and_reproduction_metadata():
    report = reporting.build_report(_score())
    assert report["schema"] == "policyengine-macro/report/v1"
    assert report["headline"]["model"] == "og-uk"
    assert report["headline"]["runtime"]
    assert report["quantities"]["gdp"]["units"]
    assert report["quantities"]["gdp"]["time_basis"]
    assert report["assumptions"]
    assert report["limitations"]
    assert report["validation"]
    assert report["provenance"]["model_version"]
    assert report["provenance"]["data_vintage"]
    assert report["provenance"]["reproducibility"]


def test_markdown_report_contains_required_sections():
    text = reporting.render_markdown(_score())
    for heading in (
        "## Results",
        "## Assumptions",
        "## Limitations",
        "## Validation",
        "## Warnings",
        "## Provenance and reproducibility",
    ):
        assert heading in text


def test_cli_report_accepts_score_envelope_from_stdin():
    result = CliRunner().invoke(
        main,
        ["report", "--format", "json"],
        input=json.dumps({"score": _score()}),
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["schema"] == "policyengine-macro/report/v1"
