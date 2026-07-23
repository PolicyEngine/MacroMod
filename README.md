# PolicyEngine Macro

**PolicyEngine Macro** is an open platform for answering economic and public-policy
questions with transparent, reproducible models. It brings household analysis,
macroeconomic scenarios, forecasts, empirical shock identification, and long-run
structural analysis behind a common discovery and run surface.

🌐 **[policyengine-macro.vercel.app](https://policyengine-macro.vercel.app)** · a PolicyEngine project

---

## The models

Each model answers a different class of economic question. MacroMod provides a
common way to discover, run, and interpret them while preserving the assumptions,
horizons, evidence, and outputs specific to each model. Results from different
model classes are often complementary rather than directly comparable.

| model | status | repo |
|-------|--------|------|
| **Overlapping generations (OG-UK)** | experimental; local steady-state scoring | [PSLmodels/OG-UK](https://github.com/PSLmodels/OG-UK) |
| **OBR macroeconometric model** | validated for selected hosted scenarios | [PolicyEngine/obr-macroeconomic-model](https://github.com/PolicyEngine/obr-macroeconomic-model) |
| **Bank of England structural VAR (boe-svar)** | validated replication for selected hosted outputs; does not score reforms | [PolicyEngine/boe-var-model](https://github.com/PolicyEngine/boe-var-model) |
| **PolicyEngine tax-benefit microsimulation** | hosted household and population analysis; no macro feedback | [PolicyEngine/policyengine.py](https://github.com/PolicyEngine/policyengine.py) |
| **FRB-US (US macroeconometric model)** | hosted raw-shock experiments; VAR expectations only | [PolicyEngine/us-frb-model](https://github.com/PolicyEngine/us-frb-model) |
| More model classes (incl. OG-USA) | planned | — |

PolicyEngine is the *micro* member: person/household-resolution taxes and
benefits for the UK and US — the same engine that powers
[policyengine.org](https://policyengine.org) — complementing the macro models.

The models live in their own repositories. This repo hosts the **PolicyEngine Macro
website** and the **integration layer** (`integration/`) — a `pe-macro` CLI
and MCP server over the models, with CI auto-deploying the hosted MCP server
to Modal on every merge — merges to the model repos
(obr-macroeconomic-model, boe-var-model) trigger the same redeploy via
`repository_dispatch` — so you can drive them from any AI workflow.

The OBR emulator also runs as a live dashboard:
[obr-macroeconomic-model.vercel.app](https://obr-macroeconomic-model.vercel.app/).

## Quickstart — score a reform

The OLG model is a Python package; pip installs it straight from GitHub, no
clone needed (Python 3.11–3.13).

```bash
pip install git+https://github.com/PSLmodels/OG-UK
```

```python
from datetime import datetime
from policyengine.core import ParameterValue, Policy
from policyengine.tax_benefit_models.uk import uk_latest
from oguk import solve_steady_state, map_to_real_world

# Build a reform from real PolicyEngine parameters (basic rate 20% → 21%)
param = uk_latest.get_parameter("gov.hmrc.income_tax.rates.uk[0].rate")
reform = Policy(name="Basic rate 21%", parameter_values=[
    ParameterValue(parameter=param, value=0.21,
                   start_date=datetime(2026, 1, 1))])

# Solve baseline and reform steady states (~5–15 min each)
baseline  = solve_steady_state(start_year=2026)
reform_ss = solve_steady_state(start_year=2026, policy=reform)

# Map model units → current-price £bn
impact = map_to_real_world(baseline, reform_ss)
print(f"GDP change: {impact.gdp_change:+.1f}bn ({impact.gdp_pct:+.3f}%)")
```

See the [OG-UK model page](https://policyengine-macro.vercel.app/olg/) for the full guide —
parameter paths, solver options, structural shocks, and the transition path —
the [OBR model page](https://policyengine-macro.vercel.app/obr/) for the macroeconometric
emulator, the [SVAR model page](https://policyengine-macro.vercel.app/svar/) for the
structural VAR, and the [documentation](https://policyengine-macro.vercel.app/docs/) for
how the model classes differ and when to use which.

## Connecting to an AI

The [connect page](https://policyengine-macro.vercel.app/connect/) covers three ways to use the
models:

- **MCP** — the hosted Model Context Protocol server is **live** at
  `https://policyengine--policyengine-macro-mcp-serve.modal.run/mcp`. Add it as a custom
  connector in Claude or ChatGPT, or in Claude Code:

  ```bash
  claude mcp add --transport http policyengine-macro https://policyengine--policyengine-macro-mcp-serve.modal.run/mcp
  ```

  The tools include `score_reform` (a PolicyEngine reform through a supported
  scoring adapter), `obr_shock` and `list_reform_variables` (raw OBR
  variable shocks in model units), `frbus_shock`, `frbus_list_variables` and
  `frbus_summary` (FRB/US impulse responses under a selectable monetary policy
  rule; `score_reform` refuses `model='frbus'` because there is deliberately no
  PolicyEngine-reform bridge for it), `forecast_uk`, `latest_shocks`,
  `model_summary` (SVAR), and the PolicyEngine microsimulation tools
  (`calculate_household`, `household_reform_impact`, `list_reform_parameters`,
  `population_reform_impact`). `score_reform` with `model='og'` works locally
  only: OG-UK is deliberately excluded from the hosted image (a score takes
  tens of minutes) — use `pe-macro score --model og` instead. The OBR reform
  bridge translates a static population costing into a household-disposable-
  income scenario; it is a demand-side approximation, not a general reform-
  incidence model. Direct OBR scenarios use `obr_shock`.
  The server runs serverless and scales to zero — the first call after idle
  may take ~10 s to wake.
- **CLI** — the `pe-macro` CLI (`score`, `obr-shock`, `variables`, `forecast`,
  `shocks`, `summary`, `household`, `household-impact`, `population-impact`,
  `parameters`, `og-score`) lives
  in [`integration/`](integration/); PyPI publish is planned. Install it with
  PolicyEngine, the OBR emulator, and the SVAR via:

  ```bash
  pip install "policyengine-macro[models] @ git+https://github.com/PolicyEngine/macro#subdirectory=integration"
  ```

  The same extra now installs FRB-US with its packaged model and LONGBASE
  runtime data; no separate checkout is required.
- **Code** — drive each model's Python API yourself.

## The site

A static site in the [populace.dev](https://populace.dev) design language — no
build step.

```bash
python3 -m http.server 8000   # then open http://localhost:8000/
```

| path | page |
|------|------|
| `index.html` | the suite — idea, models, pipeline, outputs |
| `models/` | model discovery — choose by question and browse all five models |
| `olg/` | the OG-UK model page — install, quickstart, options, shocks, outputs |
| `obr/` | the OBR macroeconometric model — quickstart, solver, levers, forecasting |
| `svar/` | the UK structural VAR — the model, quickstart, outputs, validation |
| `pe/` | PolicyEngine tax-benefit microsimulation — household calculator, reforms, population analysis |
| `docs/` | comparison — the model classes side by side and when to use which |
| `papers/` | working-paper library linked to models and validation evidence |
| `connect/` | connect it or code it — MCP / CLI setup and the Python API |

Deployed on Vercel (PolicyEngine team). `vercel.json` enables clean URLs.

## Adding a model

A new model touches a fixed set of places. Update all of them so the site
stays consistent (this is exactly the set the OBR model added):

1. **`<slug>/index.html`** — a new model reference page. Copy `olg/` or `obr/`
   as the template: `<body class="doc">`, the shared nav, and the section
   rhythm (what it is → quickstart → how it works → levers → calibration).
2. **`index.html`** — add a `.strategy-card` in the `#models` grid linking to
   `/<slug>/`.
3. **`docs/index.html`** — add a `.doc-index` card, a column in the comparison
   table, and a when-to-use bullet in `#choose`.
4. **`connect/index.html`** — add a `<div class="model-pane" data-model="<slug>">`
   in the `#code` section and a button in `#model-seg` (the model selector JS
   toggles on `data-model`).
5. **Nav** — every page's `.nav-links` is identical; no change needed unless you
   add a top-level section.
6. **`README.md`** — the models table, the quickstart links, and the site-paths
   table above.

Keep model copy grounded in the model's own repo/docs, and label any
non-real numbers as illustrative.

## Roadmap

- [x] `pe-macro` CLI (in `integration/`; PyPI publish still to come)
- [x] Local MCP server (`python -m policyengine_macro.mcp_server`)
- [x] Hosted MCP server (`https://policyengine--policyengine-macro-mcp-serve.modal.run/mcp`, auto-deployed by CI)
- [x] OG-UK steady-state scoring (`pe-macro score --model og` / `pe-macro og-score`, local only)
- [x] Population-level PolicyEngine reform scoring (`population_reform_impact`, hosted and local)
- [x] FRB/US Python implementation ([PolicyEngine/us-frb-model](https://github.com/PolicyEngine/us-frb-model)), wired into the CLI (`pe-macro frbus-shock`) and the hosted MCP server
- [ ] Additional macroeconomic model classes (incl. OG-USA)

---

Open source · a [PolicyEngine](https://policyengine.org) project.
