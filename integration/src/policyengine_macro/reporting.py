"""Stable, serialisable reports for common ``ScoreResult`` objects."""

from __future__ import annotations

from typing import Any

from policyengine_macro.capabilities import get_status
from policyengine_macro.core import ScoreResult


def build_report(value: ScoreResult | dict[str, Any]) -> dict[str, Any]:
    """Return a versioned report envelope suitable for storage or rendering."""
    if isinstance(value, dict) and "score" in value:
        value = value["score"]
    score = ScoreResult.model_validate(value)
    runtime = get_status(score.model)["runtime"]
    return {
        "schema": "policyengine-macro/report/v1",
        "headline": {
            "model": score.model,
            "model_class": score.model_class,
            "analysis_type": score.analysis_type,
            "result_type": score.result_type,
            "country": score.country,
            "horizon": score.horizon,
            "baseline": score.baseline,
            "runtime": runtime,
        },
        "reform": score.reform,
        "quantities": {
            name: quantity.model_dump(mode="json")
            for name, quantity in score.quantities.items()
        },
        "uncertainty": score.uncertainty,
        "assumptions": score.assumptions,
        "limitations": score.caveats,
        "validation": score.validation,
        "warnings": score.warnings,
        "provenance": score.provenance.model_dump(mode="json"),
        "distributional": (
            score.distributional.model_dump(mode="json")
            if score.distributional else None
        ),
    }


def render_markdown(value: ScoreResult | dict[str, Any]) -> str:
    """Render the common report without dropping interpretation metadata."""
    report = build_report(value)
    h = report["headline"]
    lines = [
        f"# {h['model']} result",
        "",
        f"- Analysis: {h['analysis_type']} ({h['result_type']})",
        f"- Geography: {h['country'].upper()}",
        f"- Horizon: {h['horizon']}",
        f"- Baseline: {h['baseline']}",
        f"- Expected runtime: {h['runtime']}",
        "",
        "## Results",
        "",
        "| Quantity | Change | Units | Time basis | Comparability |",
        "|---|---:|---|---|---|",
    ]
    for name, quantity in report["quantities"].items():
        change = quantity.get("delta_bn")
        if change is None:
            change = quantity.get("delta_pct")
        lines.append(
            f"| {name} | {change if change is not None else 'n/a'} "
            f"| {quantity['units']} | {quantity['time_basis']} "
            f"| {quantity['comparability']} |"
        )

    for title, key in (
        ("Assumptions", "assumptions"),
        ("Limitations", "limitations"),
        ("Validation", "validation"),
        ("Warnings", "warnings"),
    ):
        lines.extend(["", f"## {title}", ""])
        items = report[key]
        lines.extend(f"- {item}" for item in items)
        if not items:
            lines.append("- None declared.")

    provenance = report["provenance"]
    lines.extend([
        "",
        "## Provenance and reproducibility",
        "",
        f"- Model version: {provenance['model_version']}",
        f"- Package: {provenance['package']} {provenance['package_version']}",
        f"- Adapter version: {provenance['adapter_version']}",
        f"- Source revision: {provenance['source_revision']}",
        f"- Data vintage: {provenance['data_vintage']}",
        f"- Baseline vintage: {provenance['baseline_vintage']}",
        f"- Run at: {provenance['run_at']}",
        f"- Reproduce: {provenance['reproducibility']}",
    ])
    return "\n".join(lines) + "\n"
