# MacroMod

MacroMod is PolicyEngine's suite of open-source macroeconomic simulation
models for scoring public policy — general-equilibrium dynamics for GDP,
investment, consumption, revenue, and debt, driven by the same tax-and-benefit
rules and microdata that power PolicyEngine.

## The models

| model | status | repo |
|-------|--------|------|
| Overlapping generations (OG-UK) | shipped | [PSLmodels/OG-UK](https://github.com/PSLmodels/OG-UK) |
| More model classes | planned | — |

The models live in their own repositories; this repo hosts the MacroMod
website and, over time, the integration layer (CLI, MCP server, agent skill).

## The site

A static site in the populace.dev design language — no build step.

```bash
python3 -m http.server 8642   # then open http://localhost:8642/
```

- `index.html` — the suite: idea, models, pipeline, outputs
- `olg/` — the OG-UK model page: install, quickstart, PolicyEngine connection, options, shocks, transition path, outputs
- `connect/` — connect it or code it: MCP / CLI / Skill setup for Claude and ChatGPT, plus the Python API walkthrough

## Quickstart (the model itself)

```bash
pip install git+https://github.com/PSLmodels/OG-UK   # Python 3.11–3.13
```

```python
from oguk import solve_steady_state, map_to_real_world

baseline = solve_steady_state(start_year=2026)
reform_ss = solve_steady_state(start_year=2026, policy=reform)
impact = map_to_real_world(baseline, reform_ss)
```

A PolicyEngine project. Open source.
