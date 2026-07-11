# MacroMod

**MacroMod** is PolicyEngine's suite of open-source **macroeconomic simulation
models** for scoring public policy. Where microsimulation tells you who pays
what the morning after a reform, MacroMod traces the second act — how people
work, save, and invest differently, how firms adjust capital, and how wages,
interest rates, and the revenue estimate move with them — in a structural,
general-equilibrium engine.

🌐 **[macromod.vercel.app](https://macromod.vercel.app)** · a PolicyEngine project

---

## The models

Every model in the suite scores the same PolicyEngine reform objects and reports
the same real-world quantities (GDP, consumption, investment, government,
revenue, debt in £bn), so results are comparable across model classes.

| model | status | repo |
|-------|--------|------|
| **Overlapping generations (OG-UK)** | shipped | [PSLmodels/OG-UK](https://github.com/PSLmodels/OG-UK) |
| **OBR macroeconometric model** | shipped | [PolicyEngine/obr-macroeconomic-model](https://github.com/PolicyEngine/obr-macroeconomic-model) |
| More model classes | planned | — |

The models live in their own repositories. This repo hosts the **MacroMod
website** and, over time, the **integration layer** (CLI, MCP server) that lets
you drive them from any AI workflow.

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

See the [OG-UK model page](https://macromod.vercel.app/olg/) for the full guide —
parameter paths, solver options, structural shocks, and the transition path —
the [OBR model page](https://macromod.vercel.app/obr/) for the macroeconometric
emulator, and the [documentation](https://macromod.vercel.app/docs/) for how the
two model classes differ and when to use which.

## Connecting to an AI

The [connect page](https://macromod.vercel.app/connect/) covers three ways to use the
models:

- **Code** — drive the Python API yourself (works today).
- **MCP** — a Model Context Protocol server for Claude and ChatGPT *(coming soon)*.
- **CLI** — a `macromod score` command *(coming soon)*.

## The site

A static site in the [populace.dev](https://populace.dev) design language — no
build step.

```bash
python3 -m http.server 8000   # then open http://localhost:8000/
```

| path | page |
|------|------|
| `index.html` | the suite — idea, models, pipeline, outputs |
| `olg/` | the OG-UK model page — install, quickstart, options, shocks, outputs |
| `obr/` | the OBR macroeconometric model — quickstart, solver, levers, forecasting |
| `docs/` | documentation — the two model classes compared and when to use which |
| `connect/` | connect it or code it — MCP / CLI setup and the Python API |

Deployed on Vercel (PolicyEngine team). `vercel.json` enables clean URLs.

## Adding a model

A new model touches a fixed set of places. Update all of them so the site
stays consistent (this is exactly the set the OBR model added):

1. **`<slug>/index.html`** — a new model reference page. Copy `olg/` or `obr/`
   as the template: `<body class="doc">`, the shared nav, and the section
   rhythm (what it is → quickstart → how it works → levers → calibration).
2. **`index.html`** — add a `.strategy-card` in the `#models` grid linking to
   `/<slug>/`, and append the model to the `.stack-note` line.
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

- [ ] `macromod` CLI + PyPI publish
- [ ] Local MCP server (`uvx macromod-mcp`)
- [ ] Hosted MCP server (`mcp.macromod.dev`)
- [ ] Additional macroeconomic model classes
- See [#1](https://github.com/PolicyEngine/MacroMod/issues/1) — Rust port of the solver core

---

Open source · a [PolicyEngine](https://policyengine.org) project.
