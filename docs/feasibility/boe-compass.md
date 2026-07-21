# Bank of England COMPASS feasibility review

**Decision (21 July 2026): no-go for implementation under the COMPASS name.**

COMPASS adds a valuable forward-looking UK monetary-policy capability, but the
public artefacts do not currently support a defensible, redistributable
replication. Reassess if the Bank releases executable model code, a clear
software licence, the estimation/calibration inputs and a machine-checkable
benchmark bundle. Until then, do not publish an approximate model as COMPASS.

## Evidence

### Equations, coefficients, code and data

The [2013 COMPASS paper](https://www.bankofengland.co.uk/working-paper/2013/the-boes-forecasting-platform-compass-maps-ease-and-the-suite-of-models)
provides a detailed model description, equations, measurement mappings,
calibrated values, posterior summaries and impulse-response figures. Its
estimation uses 15 quarterly observables over 1993Q1–2007Q4, with an earlier
training sample. Some inputs are Bank-constructed or adjusted rather than
direct public series.

The paper describes execution through the Bank's internal MATLAB MAPS toolkit
and `.MAPS`/`.MAT` model files, but the publication page provides the paper and
appendix—not executable COMPASS or MAPS source. The [2025 successor technical
paper](https://www.bankofengland.co.uk/macro-technical-paper/2025/decompositions-forecasts-and-scenarios-from-an-estimated-dsge-model-for-the-uk-economy)
documents an energy-augmented evolution and its data sources, estimation,
decompositions, forecasts and scenarios, but likewise does not publish a code
or model download on its publication page.

The Bank has released a [real-time data archive for forecast rounds from 2000
to 2013](https://www.bankofengland.co.uk/working-paper/2015/evaluating-uk-point-and-density-forecasts-from-an-estimated-dgse-model-the-role-of-off-model),
which is useful evidence, but the Bank cautions that it cannot answer questions
about construction of individual series. That archive is not a complete
estimation-and-execution package.

### Replication benchmarks

The 2013 paper contains posterior tables and impulse responses, and the 2025
paper contains moments, empirical comparisons, decompositions and scenario
figures. These are useful visual targets, but no public deterministic fixture,
parameter-draw archive, solved state-space matrices or expected-output file was
found. Independent reproduction would therefore require interpretive
reconstruction and could not establish software-level equivalence.

### Licensing

No software licence accompanies the model because no model code is published.
The Bank's [general legal terms](https://www.bankofengland.co.uk/legal) allow
ordinary resources to be downloaded or printed for personal or internal
non-commercial use unless stated otherwise; broader reuse requires permission.
Database statistics have separate Open Government Licence terms, while
third-party material retains its original rights. These terms are insufficient
for redistributing a COMPASS implementation or its non-public inputs under this
project's open package without written clarification.

### Proprietary or restricted inputs

The published observable tables mix ONS series, Bank calculations, in-house
series/adjustments and conditioning paths. Public ONS/Bank series could support
an independent related DSGE model, but they do not reproduce the historical
Bank information set exactly. No household panel is required for the aggregate
COMPASS core; the constraint is institutional series construction and missing
model artefacts rather than longitudinal microdata.

### Runtime and hosting

A solved log-linear state-space model would be cheap to host: forecasts,
impulse responses and conditional scenarios should run in seconds once
parameters and solution matrices are fixed. Bayesian re-estimation and
posterior simulation are materially heavier and should be asynchronous or
offline. Runtime is not the blocker; reproducibility and licensing are.

### Distinct capability

COMPASS would add model-consistent expectations, a structural monetary-policy
rule, medium-term inflation/GDP forecasts, shock decompositions and conditional
counterfactual scenarios. That is distinct from the backward-looking empirical
identification in `boe-svar`, the fiscal transmission in `obr-macro`, static
distributional microsimulation and long-run OG-UK comparisons. The Bank still
describes COMPASS as a central medium-term framework, while its [2024 external
review](https://www.bankofengland.co.uk/independent-evaluation-office/forecasting-for-monetary-policy-making-and-communication-at-the-bank-of-england-a-review/forecasting-for-monetary-policy-making-and-communication-at-the-bank-of-england-a-review)
also records important shortcomings and recommends replacement or substantial
revamping; any integration would need version-specific limitations.

### Common-result translation

Outputs can be represented safely in the common schema only after an official
or independently reproducible implementation exists. Each output would need to
be classified as forecast, historical decomposition or conditional scenario;
record the information set, conditioning paths, estimation sample, data
vintage, posterior uncertainty and units; and default to not comparable with
SVAR/OBR outputs unless horizon, baseline and quantity definitions match.

## Reconsideration gate

Proceed only when all of the following are available or supplied by the Bank:

- executable model and solution/estimation code;
- a redistribution-compatible licence;
- exact parameter/calibration files and required data transformations;
- a versioned benchmark with numerical tolerances;
- documentation of proprietary conditioning paths and replaceable inputs;
- permission to use the COMPASS name for the packaged replication.

If those conditions remain unmet, assess a separately named open UK DSGE or a
smaller distinct UK labour-market/participation model next.
