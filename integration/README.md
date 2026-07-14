# MacroMod integration layer

A single Python package (`macromod`) exposing three models behind
one CLI and one MCP server:

- **OBR emulator** (`obr_macro`, at `~/obr-macroeconomic-model`): runs the
  OBR's published model equations to score policy reforms (spending, tax).
- **UK SVAR** (`boe_var`, at `~/boe-var-model`): sign-identified Bayesian VAR
  for UK GDP/CPI forecasts and structural shock readings.
- **PolicyEngine microsimulation** (`policyengine` from PyPI, v4): full
  UK/US tax-benefit rules for custom households — a household calculator
  plus baseline-vs-reform impact analysis. Population-level scoring (via
  `pe.uk.ensure_datasets` + `Simulation`) is planned, not yet wired in: the
  UK dataset lives in a private Hugging Face repo (needs
  `HUGGING_FACE_TOKEN`) and the downloads are large, so it doesn't fit the
  zero-auth, scale-to-zero deployment yet.

`src/macromod/core.py` holds the model adapters (single source of truth);
`cli.py` and `mcp_server.py` are thin wrappers over the same functions.

## Install

Into the conda `python313` env (unset `VIRTUAL_ENV` first if set):

```bash
PY=/Users/janansadeqian/anaconda3/envs/python313/bin/python
$PY -m pip install -e /Users/janansadeqian/obr-macroeconomic-model \
                   -e /Users/janansadeqian/boe-var-model \
                   -e /Users/janansadeqian/MacroMod/integration
```

## CLI

```bash
macromod variables                                   # OBR shock variables + units
macromod score --var CGG --shock 1250 --periods 4    # £5bn/yr spending, 1 year
macromod score --var TCPRO --shock -0.05 --investment-closure   # 5pp corp tax cut
macromod forecast --horizons 12 --draws 500          # YoY GDP & CPI, 68/90 bands
macromod shocks --draws 500                          # P(sign) of latest-quarter shocks
macromod summary                                     # instant, parses committed results
```

PolicyEngine household calculator:

```bash
macromod parameters                                  # curated, verified reform parameters
macromod household --country uk \
    --people '[{"age":35,"employment_income":50000}]'
macromod household --country us \
    --people '[{"age":35,"employment_income":60000}]' \
    --tax-unit '{"filing_status":"SINGLE"}' --household '{"state_code_str":"CA"}'
macromod household-impact --country uk \
    --people '[{"age":35,"employment_income":50000}]' \
    --reform '{"gov.hmrc.income_tax.rates.uk[0].rate":0.25}'
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

Runs over stdio via `python -m macromod.mcp_server`, exposing tools
`score_reform`, `list_reform_variables`, `forecast_uk`, `latest_shocks`,
`model_summary`, plus the PolicyEngine tools `calculate_household`,
`household_reform_impact`, and `list_reform_parameters`.

Test locally with Claude Code:

```bash
claude mcp add macromod -- /Users/janansadeqian/anaconda3/envs/python313/bin/python -m macromod.mcp_server
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

Defined in `modal_app.py`: both model repos are baked into the image at the
same absolute paths as on the laptop and installed with `pip install -e`, so
all `Path(__file__)`-relative data/results lookups (obr `data/`, boe_var
`data/boe_var_data.csv` and `results/*.md`) resolve unchanged. `policyengine`
is installed from PyPI in the image; because it is imported lazily inside the
adapters, cold starts stay fast and only the first household tool call in a
fresh container pays the ~20 s model load.

**Add it as a connector**

- claude.ai: Settings → Connectors → Add custom connector → paste the URL
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

**Redeploy** after changing any model repo or `macromod`:

```bash
modal deploy /Users/janansadeqian/MacroMod/integration/modal_app.py
```

**Remote smoke test** (hits the live deployment; skipped without the env var):

```bash
MACROMOD_REMOTE_TESTS=1 python -m pytest tests/test_remote_mcp.py -v
```

## Tests

```bash
cd /Users/janansadeqian/MacroMod/integration
python -m pytest tests -q     # includes an end-to-end stdio MCP client test
```
