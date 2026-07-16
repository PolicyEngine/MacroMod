# PolicyEngine Macro integration layer

A single Python package (`macromod`) exposing the suite's models behind
one CLI and one MCP server:

- **OBR emulator** (`obr_macro`): runs the OBR's published model equations —
  raw exogenous-variable shocks via `obr_shock`.
- **UK SVAR** (`boe_var`): sign-identified Bayesian VAR for UK GDP/CPI
  forecasts and structural shock readings.
- **PolicyEngine microsimulation** (`policyengine` from PyPI, v4): full
  UK/US tax-benefit rules — household calculator, baseline-vs-reform
  impacts, and population-level reform scoring over representative
  microdata (UK data is private: set `HUGGING_FACE_TOKEN`; the hosted
  deployment provisions it server-side).
- **OG-UK** (`oguk`, optional/local-only): overlapping-generations
  steady-state scoring through `macromod score --model og`.

`score_reform` is the one reform vocabulary across the suite: the same flat
`{parameter_path: value}` dict as the microsimulation tools, dispatched to a
macro model by its declared contract (OG via PolicyEngine-estimated tax
functions; the OBR static-costing bridge is
[#9](https://github.com/PolicyEngine/macro/issues/9) — until it lands the
OBR arm errors with a pointer to `obr_shock`).

`src/macromod/core.py` holds the model adapters (single source of truth);
`cli.py` and `mcp_server.py` are thin wrappers over the same functions.

## Install

No clone needed — one pip install pulls the CLI plus the hosted-model
packages (the OBR emulator and the SVAR ship their data as package data):

```bash
pip install "macromod[models] @ git+https://github.com/PolicyEngine/macro#subdirectory=integration"
```

A shorter `pip install macromod` will come with PyPI publication.

For development, install with the full model set (policyengine included via
the `[models]` extra), overriding the two model packages with local editable
checkouts; OG-UK is optional and local-only:

```bash
uv venv && uv pip install -e "./integration[models]" pytest \
    -e ../obr-macroeconomic-model -e ../boe-var-model
uv pip install "oguk @ git+https://github.com/PSLmodels/OG-UK"  # optional, for --model og
```

## CLI

```bash
macromod variables                                    # OBR shock variables + units
macromod score --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.21}' \
    --model og                                        # PolicyEngine reform -> OG-UK (slow)
macromod obr-shock --var CGG --shock 1250 --periods 4 # £5bn/yr spending, 1 year
macromod obr-shock --var TCPRO --shock -0.05          # 5pp corp tax cut (closure auto-on)
macromod forecast --horizons 12 --draws 500           # YoY GDP & CPI, 68/90 bands
macromod shocks --draws 500                           # P(sign) of latest-quarter shocks
macromod summary                                      # instant, parses committed results
```

PolicyEngine tools:

```bash
macromod parameters                                   # curated reform parameters
macromod household --country uk \
    --people '[{"age":35,"employment_income":50000}]'
macromod household-impact --country uk \
    --people '[{"age":35,"employment_income":50000}]' \
    --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.25}'
macromod population-impact --country uk \
    --reform '{"gov.hmrc.cgt.basic_rate":0.20,"gov.hmrc.cgt.higher_rate":0.40}'
```

`--people`/`--benunit`/`--tax-unit`/`--household`/`--reform` take JSON.
Money amounts are annual GBP (uk) or USD (us); reform rates are decimals
(0.25 = 25%). The first PolicyEngine call in a process pays a ~20 s model
import (it is loaded lazily).

Add `--json` to any command for machine-readable output. Units: CGG/CGIPS
shocks are £m per quarter; TCPRO is a decimal rate change. SVAR estimation
results are cached in-process by draw count, so repeat calls are instant
within one process (each CLI invocation is a fresh process; the cache mainly
benefits the MCP server).

## MCP server

Runs over stdio via `python -m macromod.mcp_server`, exposing ten tools:
`score_reform` (a PolicyEngine reform through a chosen macro model),
`obr_shock` and `list_reform_variables` (raw OBR variable shocks),
`forecast_uk`, `latest_shocks`, `model_summary` (SVAR), and the PolicyEngine
tools `calculate_household`, `household_reform_impact`,
`list_reform_parameters`, `population_reform_impact`.

Test locally with Claude Code:

```bash
claude mcp add macromod -- python -m macromod.mcp_server
```

Default `draws=500` keeps tool calls to tens of seconds; raise it (e.g. 2000+)
for smoother bands. Repeated calls with the same parameters hit an in-process
cache and return instantly.

## Deployment (Modal)

The MCP server is deployed on Modal (workspace `policyengine`) over
streamable HTTP:

```
https://policyengine--macromod-mcp-serve.modal.run/mcp
```

Defined in `modal_app.py`. `policyengine[models]` is installed in the image;
because it is imported lazily inside the adapters, cold starts stay fast and
only the first PolicyEngine tool call in a fresh container pays the ~20 s
model load. The private UK microdata credential comes from the Modal secret
`macromod-hf`, with derived datasets cached on the `macromod-pe-data` volume.

**Add it as a connector**

- claude.ai: Settings -> Connectors -> Add custom connector -> paste the URL
  above (including the trailing `/mcp`).
- Claude Code:

  ```bash
  claude mcp add --transport http macromod-remote \
      https://policyengine--macromod-mcp-serve.modal.run/mcp
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
MACROMOD_REMOTE_TESTS=1 python -m pytest tests/test_remote_mcp.py -v
```

## Tests

```bash
cd integration
python -m pytest tests -q     # includes an end-to-end stdio MCP client test
```
