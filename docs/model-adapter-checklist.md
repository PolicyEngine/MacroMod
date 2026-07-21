# New-model acceptance checklist

No model is accepted because it can merely produce numbers. Reviewers must be
able to establish what it answers, reproduce a benchmark, and interpret every
output without an inferred mapping or unit conversion.

## Feasibility gate

- [ ] Public equations, calibration, coefficients, code, and required data are inventoried.
- [ ] Redistribution licences for code, data, fixtures, and documentation are recorded.
- [ ] Proprietary, restricted, panel, or secure-environment data are identified and excluded from hosted artefacts.
- [ ] At least one independent replication benchmark and its tolerance are named.
- [ ] Runtime, memory, concurrency, and hosted/local/asynchronous access are measured.
- [ ] The model adds a registered question type or materially distinct method.
- [ ] A named maintainer owns upstream-version and data-vintage updates.
- [ ] The feasibility report ends with an explicit go/no-go decision.

## Adapter contract

- [ ] Add one entry to `capabilities.MODELS`; `validate_registry()` passes.
- [ ] Subclass `ModelAdapter` and accept only `AnalysisRequest`.
- [ ] Declare geography, question types, inputs, outputs, horizon, runtime, access, uncertainty, validation status, and unsupported uses.
- [ ] Validate input names, ranges, dimensions, units, and frequency before execution.
- [ ] Reject unknown mappings; never ask a language model to invent a mapping or conversion.
- [ ] Return `AnalysisResult` with non-empty assumptions, limitations, validation evidence, and provenance.
- [ ] Label the result as forecast, scenario, historical estimate, calibration, or illustration.
- [ ] Wrap every numeric output in `Quantity`, including units, price/time basis, geography, baseline, uncertainty, and comparability.
- [ ] Record model and adapter versions, source URL/revision, data and baseline vintages, run time, and reproduction instructions.

## Tests and release evidence

- [ ] Reuse the shared contract tests for valid, invalid, and unsupported requests.
- [ ] Test missing units/provenance and mismatched provenance fail closed.
- [ ] Test one immutable upstream replication fixture within a declared tolerance.
- [ ] Test clean installation and the documented access path.
- [ ] Test CLI, MCP, documentation, and website claims against the registry.
- [ ] Separate synthetic public CI from tests requiring protected data.
- [ ] Document limitations, unsupported applications, benchmark results, citation, and licence.

COMPASS or any other pilot starts with the feasibility gate. If public material
cannot support a defensible replication, record a no-go rather than publishing
an approximate implementation under the source model's name.
