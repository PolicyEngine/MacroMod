"""High-risk website claims that must stay aligned with runtime contracts."""

from pathlib import Path
import csv
import json
import re


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_svar_site_uses_runtime_estimation_endpoint():
    pages = _read("svar/index.html") + _read("docs/index.html")
    assert "estimated through 2023Q2" in pages
    assert "estimation sample to 2025Q1" not in pages
    assert "Estimation to 2025Q1" not in pages


def test_og_is_not_labelled_as_us_model():
    pages = "".join(
        _read(path)
        for path in (
            "index.html",
            "models/index.html",
            "papers/index.html",
            "docs/index.html",
        )
    )
    assert "psl-og · UK + US" not in pages


def test_site_does_not_promise_obr_borrowing_output():
    pages = _read("index.html") + _read("models/index.html") + _read("docs/index.html")
    assert "growth and borrowing" not in pages
    assert "borrowing after" not in pages


def test_obr_site_uses_declared_household_costing_injection_point():
    page = _read("obr/index.html")
    assert "HHDI_ADDFACTOR" in page
    assert "Corporation-tax channel unstable" not in page
    assert "corporation-tax scenarios are excluded" not in page


def test_connect_page_matches_current_clients_and_switches_both_views():
    page = _read("connect/index.html")
    assert "eligible ChatGPT web plan" in page
    assert "Settings → Apps → Advanced Settings" in page
    assert "For Codex CLI" in page
    assert 'document.querySelectorAll("[data-for]")' in page


def test_mobile_css_does_not_hide_document_overflow():
    css = _read("style.css")
    body = css.split("body {", 1)[1].split("}", 1)[0]
    assert "overflow-x: hidden" not in body
    assert "min-height: 112px" in css
    assert "min-height: 44px" in css


def test_current_boe_forecast_starts_after_latest_complete_data_edge():
    payload = json.loads(_read("papers/boe-svar/figures/current_forecast.json"))
    assert payload["data_edge"] == "2026Q1"
    assert payload["forecast_start"] == "2026Q2"
    assert list(payload["forecast"])[0] == "2026Q2"
    assert list(payload["forecast"])[-1] == "2029Q2"
    assert "2026Q2 to 2029Q2" in _read("index.html")
    assert "frozen 2024Q2 validation experiment" in _read("svar/index.html")


def test_current_obr_outlook_uses_latest_official_efo_window():
    rows = list(
        csv.DictReader((ROOT / "papers/obr-macro/figures/current_outlook.csv").open())
    )
    assert rows[0]["quarter"] == "2026Q1"
    assert rows[-1]["quarter"] == "2031Q1"
    page = _read("obr/index.html")
    assert "Latest official baseline available on 21 July 2026" in page
    assert "OBR March 2026 detailed forecast tables" in page


def test_frozen_validation_vintages_are_not_relabelled_as_current_forecasts():
    page = _read("validation/index.html")
    assert "frozen 2024Q2 data edge" in page
    assert "outturn backtest above retains November 2025" in page


def test_validation_and_paper_landings_lead_with_current_uk_vintages():
    validation = _read("validation/index.html")
    assert "anchored baseline vs March 2026 EFO" in validation
    assert "Current March 2026 EFO baseline" in validation
    assert "anchored baseline vs November 2025 EFO" not in validation

    obr_paper = _read("papers/obr-macro/index.html")
    assert "Current baseline: March 2026 EFO" in obr_paper
    boe_paper = _read("papers/boe-svar/index.html")
    assert "Current forecast: data through" in boe_paper
    assert "2026Q2&ndash;2029Q2" in boe_paper


def test_paper_page_counts_match_embedded_pdfs():
    expected = {
        "obr-macro": 36,
        "boe-svar": 29,
        "frb-us": 36,
        "psl-og": 34,
    }
    listing = _read("papers/index.html")
    for slug, pages in expected.items():
        landing = _read(f"papers/{slug}/index.html")
        assert f"{pages} pages" in landing
        assert re.search(rf"{re.escape(slug)} · [^<]* · {pages} pages", listing)
