# PolicyEngine Macro integration layer

A single Python package (`policyengine-macro`) exposing the suite's models behind
one CLI and one MCP server:

- **OBR emulator** (`obr_macro`): runs the OBR's published model equations —
  raw exogenous-variable shocks via `obr_shock`.
- **FRB/US** (`frbus`): the Federal Reserve Board's US macroeconometric
  model (VAR expectations, 284 endogenous equations) — raw variable and
  add-factor shocks via `frbus_shock`, under a selectable monetary policy
  rule. There is deliberately NO PolicyEngine-reform bridge for FRB/US.
- **UK SVAR** (`boe_var`): sign-identified Bayesian VAR for UK GDP/CPI
  forecasts and structural shock readings.
- **PolicyEngine microsimulation** (`policyengine` from PyPI, v4): full
  UK/US tax-benefit rules — household calculator, baseline-vs-reform
  impacts, and population-level reform scoring over representative
  microdata (UK data is private: set `HUGGING_FACE_TOKEN`; the hosted
  deployment provisions it server-side).
- **OG-UK** (`oguk`, optional/local-only): overlapping-generations
  steady-state scoring through `pe-macro score --model og`.

`score_reform` is the one reform vocabulary across the suite: the same flat
`{parameter_path: value}` dict as the microsimulation tools, dispatched to a
scoring model by its declared contract:

- `og` — the reform enters through PolicyEngine-estimated tax functions
  (long-run steady-state general equilibrium).
- `obr` — the microsim static-costing bridge
  ([#9](https://github.com/PolicyEngine/macro/issues/9)): the reform is
  costed per year with the PolicyEngine population microsimulation, the
  annual budgetary impacts enter the OBR emulator as a quarterly household
  disposable income (`HHDI`) shock path (sign-corrected: revenue raised
  lowers HHDI, flat within each year), and the second-round demand effects
  come out. Demand-side incidence only; corporation-tax reforms are refused
  with a pointer to the direct `obr_shock --var TCPRO` lever.
- `microsim` — the PolicyEngine population costing itself (static, no macro
  feedback).
- `og+microsim` — dynamic scoring
  ([#11](https://github.com/PolicyEngine/macro/issues/11)): a two-run
  structure —

  ```
  reform ──> OG-UK steady states (baseline cached, reform solved)
                └─> EconomicAssumptions: earnings factor = w_reform/w_baseline
                      └─> DIRECT INPUT SCALING: the reform simulation's
                          employment-income arrays are multiplied by the
                          factor via the engine's supported
                          Dynamic(simulation_modifier=...) hook
  reform (+ modifier) ──> microsim vs the untouched stock baseline
  ```

  Input scaling — not a parameter overlay — because uprating-parameter
  overrides are empirically DEAD in population runs: the per-year microdata
  are pre-uprated at dataset build time, so overriding
  `gov.economic_assumptions.indices.obr.average_earnings` returns exactly
  zero everywhere (verified against the production engine; such reforms are
  refused in dynamic scoring rather than silently ignored). The overlay
  carries only the reform/baseline RATIO, applied to the reform side only,
  so the static effect embedded in the stock inputs is never
  double-counted; a null macro result attaches no modifier and reduces it
  exactly to `microsim`. Caveats (spelled out in every result): the
  steady-state factor is applied flat from the start year with no
  transition dynamics; the aggregate labour-supply change is reported but
  not distributionally allocated in v1; employment income only
  (self-employment and pension income are not adjusted, and the OG model is
  real, so there is no price-level overlay). UK-only and local-only (oguk
  is excluded from the hosted image); also exposed as
  `pe-macro dynamic-score` and the `dynamic_reform_impact` MCP tool.

  **Two-environment pipeline (required until
  [PSLmodels/OG-UK#68](https://github.com/PSLmodels/OG-UK/issues/68)):**
  oguk pins `policyengine-uk==2.88.0`, and importing the current
  `policyengine` wrapper alongside it raises a mixed-computation-mode
  error — the OG solve and the population microsim cannot share one
  process today. Run the OG solve in its own env and hand the payload
  across:

  ```bash
  # env A (.venv-og): the OG solve
  pe-macro og-score --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.21}' \
      --json > og.json
  # env B (main env): the dynamic score, consuming the payload
  pe-macro dynamic-score --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.21}' \
      --og-payload og.json
  ```

  The payload is validated against the reform and start year it was
  produced for; a mismatch is refused.

Every scoring result also carries a common `score` block — the `ScoreResult`
schema ([#10](https://github.com/PolicyEngine/macro/issues/10)): model id and
class, horizon, per-quantity deltas with units and basis, assumptions,
caveats, and an optional distributional block — so `pe-macro compare
--reform '...' --models microsim,obr` renders the same reform through
different model classes in one table. The rows are not automatically
like-for-like: each quantity retains its units, basis, and horizon, and the CLI
labels cross-class results as complementary unless their definitions align.

`src/policyengine_macro/core.py` holds the model adapters (single source of truth);
`cli.py` and `mcp_server.py` are thin wrappers over the same functions.

Every common `ScoreResult` can be converted into a stable JSON report envelope
or a Markdown report without losing units, time basis, uncertainty,
limitations, provenance, runtime, or reproduction instructions:

```bash
pe-macro score ... --json | pe-macro report --format markdown
pe-macro report saved-result.json --format json
```

MCP clients can use `format_score_report` with the same `json` or `markdown`
formats.

## Install

One pip install pulls the CLI, PolicyEngine, OBR emulator, SVAR, and FRB-US.
All model packages ship their required runtime data:

```bash
pip install "policyengine-macro[models] @ git+https://github.com/PolicyEngine/macro#subdirectory=integration"
```

A shorter `pip install policyengine-macro` will come with PyPI publication.

FRB-US packages `model.xml` and `LONGBASE.TXT` under `frbus/_data`; a normal
wheel installation can therefore execute a solve without a repository
checkout. `POLICYENGINE_MACRO_FRB_REPO` remains available as an explicit
override for older editable development checkouts.

For development, install with the available model set (PolicyEngine included
via the `[models]` extra), then override model packages with local
editable checkouts in a second step (`--no-deps`: mixing the extra's Git URLs
and editable paths in one resolution is a conflict):

```bash
uv venv && uv pip install -e "./integration[models]" pytest
uv pip install --no-deps -e ../obr-macroeconomic-model -e ../boe-var-model \
    -e ../us-frb-model
```

OG-UK (optional, for `--model og`) pins `policyengine-uk==2.88.0`, which
conflicts with the household/population stack — give it its own environment
until [PSLmodels/OG-UK#68](https://github.com/PSLmodels/OG-UK/issues/68)
lands:

```bash
uv venv .venv-og && uv pip install -p .venv-og/bin/python -e ./integration \
    "oguk @ git+https://github.com/PSLmodels/OG-UK"
```

(`-e ./integration` gives that env the `pe-macro` executable (and its legacy `policyengine-macro` alias); the base
package pins no policyengine version, so OG-UK's own pins win there.)

## CLI

```bash
pe-macro variables                                    # OBR shock variables + units
pe-macro score --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.21}' \
    --model og                                        # PolicyEngine reform -> OG-UK (slow)
pe-macro score --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.21}' \
    --model obr --years 5                             # static costing -> OBR second-round effects
pe-macro compare --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.21}' \
    --models microsim,obr                             # same reform, model classes side by side
pe-macro obr-shock --var CGG --shock 1250 --periods 4 # £5bn/yr spending, 1 year
pe-macro obr-shock --var TCPRO --shock -0.05          # 5pp corp tax cut (closure auto-on)
pe-macro frbus-variables                              # shockable FRB/US levers + units
pe-macro frbus-summary                                # FRB/US metadata + validation provenance
pe-macro frbus-shock --var rffintay_aerr --shock 1.0  # 100bp US monetary tightening
pe-macro frbus-shock --var egfe_aerr --shock 0.01 --periods 4 \
    --policy-rule fixed_funds_rate                    # fiscal shock, no monetary offset
pe-macro forecast --horizons 12 --draws 500           # YoY GDP & CPI, 68/90 bands
pe-macro shocks --draws 500                           # P(sign) of latest-quarter shocks
pe-macro summary                                      # instant, parses committed results
```

PolicyEngine tools:

```bash
pe-macro parameters                                   # curated reform parameters
pe-macro household --country uk \
    --people '[{"age":35,"employment_income":50000}]'
pe-macro household-impact --country uk \
    --people '[{"age":35,"employment_income":50000}]' \
    --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.25}'
pe-macro population-impact --country uk \
    --reform '{"gov.hmrc.cgt.basic_rate":0.20,"gov.hmrc.cgt.higher_rate":0.40}'
```

`--people`/`--benunit`/`--tax-unit`/`--household`/`--reform` take JSON.
Money amounts are annual GBP (uk) or USD (us); reform rates are decimals
(0.25 = 25%). The first PolicyEngine call in a process pays a ~20 s model
import (it is loaded lazily).

Add `--json` to any command for machine-readable output. Units: CGG/CGIPS
shocks are £m per quarter; TCPRO is a decimal rate change. FRB/US units differ
per lever and are not interchangeable — `rffintay_aerr` is in percentage
points, `trp_aerr`/`trci_aerr` are decimal rate changes, and the spending and
demand levers (`egfe_aerr`, `ecnia_aerr`, ...) are in log points of quarterly
growth, NOT dollars; run `pe-macro frbus-variables` first. SVAR estimation
results are cached in-process by draw count, so repeat calls are instant
within one process (each CLI invocation is a fresh process; the cache mainly
benefits the MCP server).

## MCP server

Runs over stdio via `python -m policyengine_macro.mcp_server`, exposing
eighteen tools:
`score_reform` (a PolicyEngine reform through a chosen macro model),
`dynamic_reform_impact` (the OG-UK overlay dynamic score; local-only —
the hosted server returns a "run locally" error),
`format_score_report` (stable JSON or Markdown reports),
`obr_shock` and `list_reform_variables` (raw OBR variable shocks),
`frbus_shock`, `frbus_list_variables` and `frbus_summary` (FRB/US),
`forecast_uk`, `latest_shocks`, `model_summary` (SVAR), and the PolicyEngine
tools `calculate_household`, `household_reform_impact`,
`list_reform_parameters`, `population_reform_impact`.

`score_reform` deliberately REFUSES `model='frbus'`: no mapping exists today
from a PolicyEngine US reform to FRB/US fiscal levers, and inventing one would
return plausible-looking wrong numbers. `frbus_shock` (raw shocks in model
units) is the supported FRB/US entry point.

The `frbus` package ships `model.xml` and `LONGBASE.TXT` as package data.
Editable checkouts are also supported, with
`POLICYENGINE_MACRO_FRB_REPO` available as an explicit path override.

Test locally with Claude Code:

```bash
claude mcp add policyengine-macro -- python -m policyengine_macro.mcp_server
```

Default `draws=500` keeps tool calls to tens of seconds; raise it (e.g. 2000+)
for smoother bands. Repeated calls with the same parameters hit an in-process
cache and return instantly.

## Deployment (Modal)

The MCP server is deployed on Modal (workspace `policyengine`) over
streamable HTTP:

```
https://policyengine--policyengine-macro-mcp-serve.modal.run/mcp
```

Defined in `modal_app.py`. `policyengine[models]` is installed in the image;
because it is imported lazily inside the adapters, cold starts stay fast and
only the first PolicyEngine tool call in a fresh container pays the ~20 s
model load. The private UK microdata credential comes from the Modal secret
`macromod-hf`, with derived datasets cached on the `policyengine-macro-pe-data` volume.

**Add it as a connector**

- claude.ai: Settings -> Connectors -> Add custom connector -> paste the URL
  above (including the trailing `/mcp`).
- Claude Code:

  ```bash
  claude mcp add --transport http policyengine-macro-remote \
      https://policyengine--policyengine-macro-mcp-serve.modal.run/mcp
  ```

**Cost profile** — `min_containers=0` (scales to zero, $0 idle),
`scaledown_window=300` (stays warm 5 min after the last call, so a chat
session pays at most one cold start), `cpu=2` / `memory=2048`,
`max_containers=3` (spend cap), `timeout=600` (high-draw forecasts). A
default forecast call costs on the order of $0.001–0.003; instant tools are
sub-cent. Cold start adds ~5–15 s.

**Redeploy** happens automatically on merge to `main` (and on
`repository_dispatch` from the model repos); manually:

```bash
modal deploy integration/modal_app.py
```

**Remote smoke test** (hits the live deployment; skipped without the env var):

```bash
POLICYENGINE_MACRO_REMOTE_TESTS=1 python -m pytest tests/test_remote_mcp.py -v
```

## Tests

New model integrations must implement the typed contract in
`policyengine_macro.adapters` and complete the
[new-model acceptance checklist](../docs/model-adapter-checklist.md) before
they are exposed through the CLI, MCP server, or website.

```bash
cd integration
python -m pytest tests -q     # includes an end-to-end stdio MCP client test
```
