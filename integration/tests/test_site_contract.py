"""High-risk website claims that must stay aligned with runtime contracts."""

from pathlib import Path


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
        _read(path) for path in (
            "index.html", "models/index.html", "papers/index.html",
            "docs/index.html",
        )
    )
    assert "psl-og · UK + US" not in pages


def test_site_does_not_promise_obr_borrowing_output():
    pages = _read("index.html") + _read("models/index.html") + _read("docs/index.html")
    assert "growth and borrowing" not in pages
    assert "borrowing after" not in pages


def test_mobile_css_does_not_hide_document_overflow():
    css = _read("style.css")
    body = css.split("body {", 1)[1].split("}", 1)[0]
    assert "overflow-x: hidden" not in body
    assert "min-height: 112px" in css
    assert "min-height: 44px" in css
