"""Deploy the PolicyEngine Macro MCP server to Modal over streamable HTTP.

    modal deploy integration/modal_app.py

Serves the FastMCP instance from `policyengine_macro.mcp_server` (14 tools:
score_reform, obr_shock, list_reform_variables, frbus_shock,
frbus_list_variables, frbus_summary, forecast_uk, latest_shocks, model_summary,
calculate_household, household_reform_impact, list_reform_parameters,
population_reform_impact, dynamic_reform_impact — the last returns an
actionable "run locally" error here, because oguk is excluded from this
image; see below) as an
ASGI app at  https://policyengine--policyengine-macro-mcp-serve.modal.run/mcp

All three model repos resolve their data files relative to their own repo root
(`Path(__file__)`-relative); `policyengine_macro.core`'s svar_summary falls back to a
checkout via POLICYENGINE_MACRO_BOE_VAR_REPO only when boe_var is absent (it is
installed here, so the fallback never fires), and _frbus_repo() falls back to
POLICYENGINE_MACRO_FRB_REPO likewise. We bake the repos into the
image at the SAME absolute paths and `pip install -e` them, so every path
resolves in the container with zero patching:
  - obr_macro:  /Users/janansadeqian/obr-macroeconomic-model  (+ data/)
  - boe_var:    /Users/janansadeqian/boe-var-model            (+ data/, results/)
  - frbus:      /Users/janansadeqian/us-frb-model             (+ vendor/, pruned)

COST PROFILE
------------
- min_containers=0  -> scales to zero: $0/hr while idle (no keep_warm).
- scaledown_window=300 -> container stays warm 5 min after the last request,
  so a chat session doesn't pay a cold start on every tool call, then sleeps.
- cpu=4, memory=8192 MiB -> ~$0.00023/s while running (4 physical cores +
  8GiB). Raised from 2/2048 for population_reform_impact: a UK population
  run measured ~1.8GB peak RSS on a laptop, so 8GiB gives 2 concurrent
  population runs headroom; 4 CPUs also cut OBR/SVAR latency. A population
  score (~10-40s warm) costs order $0.002-0.01; a default forecast
  (draws=500) similar; summary/list calls are sub-cent noise. Idle cost is
  exactly $0.
- max_containers=3 -> hard spend cap against abuse/fan-out.
- timeout=600 -> allows high-draw forecasts (e.g. draws=6000) and the
  first-ever population data download (~125MB + dataset build) to finish.

FRB/US (frbus_shock) COST AND RESOURCES
---------------------------------------
FRB/US is by far the CHEAPEST macro member here, and needed no change to
cpu/memory/timeout:
- Latency, measured on a laptop for the default 20-quarter window: ~0.1s to
  read LONGBASE, ~0.7s to parse model.xml and symbolically differentiate the
  284 equations into an analytic sparse Jacobian, ~2.3s for init_trac, and
  ~0.2-0.5s per solve (~4.4ms per quarter). A COLD frbus_shock call is
  therefore ~3s end to end. core._FRBUS_BASELINE_CACHE holds the compiled
  model and its add-factored baseline per (policy_rule, start, end), so every
  WARM call in the same container is just the shocked solve: ~0.3s.
  frbus_list_variables and frbus_summary are static and instant.
- Against the 600s timeout that is ~200x headroom. Even a 100-year window
  would land near 18s.
- Memory: the model state is a single ~860-column DataFrame over a quarterly
  index plus the sparse Jacobian — tens of MB, an order of magnitude under
  the population microsim's ~1.8GB peak that actually sizes this container.
  The cache holds one such baseline per policy rule (3 rules max).
- CPU: the solve is a scipy sparse LU on 284 equations and is effectively
  single-threaded; it does not benefit from cpu=4 the way the OBR/SVAR runs
  do, but it does not contend for it either.
- Image cost: the repo is 251MB on disk but only LONGBASE.TXT (4.4MB) and
  model.xml (533KB) are needed at runtime, so _FRB_IGNORE / _FRB_PRUNE cut it
  to ~5MB in the image (see the comment on those constants).

WHY FRB/US IS HOSTABLE AND OG-UK IS NOT
---------------------------------------
The distinction is not model size, it is what a single call has to compute.
FRB/US here uses VAR (backward-looking) expectations, so each quarter is one
Newton solve on a 284-equation system with a precomputed analytic Jacobian and
NO iteration over the future: cost is linear in the horizon at ~4.4ms per
quarter, and the data it needs is a 4.5MB text file that ships in the image.
OG-UK has to find a general-equilibrium STEADY STATE — a fixed point over the
whole lifecycle distribution — per scenario, measured at >17 minutes for one
baseline solve, two solves per reform score, plus a PolicyEngine microdata
calibration step. One is a bounded linear-time simulation, the other is an
unbounded nested fixed-point search; only the first fits a 600s request.

POPULATION DATA (population_reform_impact)
------------------------------------------
- Secret "macromod-hf" provides HUGGING_FACE_TOKEN for the private UK
  enhanced-FRS microdata on HuggingFace. The name predates the
  PolicyEngine Macro rename and is kept because Modal has no rename for
  Secrets: changing it means recreating the Secret with the token value.
- A modal.Volume ("policyengine-macro-pe-data") is mounted at
  /root/.cache/policyengine-macro;
  HF_HOME points the HuggingFace download cache inside it and
  POLICYENGINE_MACRO_PE_DATA_DIR puts the derived per-year .h5 files (~92MB/year)
  there too, so the ~125MB download + dataset build happens once and
  persists across containers.

OG-UK (oguk) IS DELIBERATELY NOT IN THIS IMAGE
----------------------------------------------
The OG-UK steady-state adapters (score_reform model='og' / og-score CLI) need a
PolicyEngine microdata calibration plus an OG-Core steady-state solve per
scenario: measured >17 minutes for ONE baseline solve at defaults on a laptop,
and a reform score needs two solves. That cannot fit the 600s Modal timeout
with any headroom, so oguk is excluded here; the score_reform MCP tool with
model='og' will return an ImportError on the hosted server. Use the local CLI
(`pe-macro score --model og`) or a local MCP server instead. If it is ever added,
`pip install git+https://github.com/PSLmodels/OG-UK` works (hatchling build;
heavy deps: ogcore, policyengine-uk==2.88.0), but calibration also downloads
the enhanced FRS dataset (HUGGING_FACE_TOKEN) and UN demographics at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HOME = "/Users/janansadeqian"
OBR_REPO = f"{HOME}/obr-macroeconomic-model"
BOE_REPO = f"{HOME}/boe-var-model"
FRB_REPO = f"{HOME}/us-frb-model"
INTEGRATION = str(Path(__file__).parent)

# POLICYENGINE_MACRO_IMAGE_SOURCE=github (used by CI) clones the model repos from
# GitHub main at image-build time instead of copying the local checkouts —
# same absolute paths, so all data-file resolution is unchanged.
GITHUB_SOURCE = os.environ.get("POLICYENGINE_MACRO_IMAGE_SOURCE") == "github"
OBR_URL = "https://github.com/PolicyEngine/obr-macroeconomic-model"
BOE_URL = "https://github.com/PolicyEngine/boe-var-model"
FRB_URL = "https://github.com/PolicyEngine/us-frb-model"

# Keep the image lean: skip the 412MB dashboard, VCS, caches, docs.
# "**/.venv" matters more than it looks: a local checkout that has been set up
# for development carries its own virtualenv (213MB in us-frb-model alone),
# which add_local_dir would otherwise copy into the image — shadowing the
# container's own site-packages with laptop-built wheels. Only the local-source
# branch can hit this; a git clone never has one.
_IGNORE = ["**/.git", "**/.github", "**/__pycache__", "**/*.egg-info",
           "**/.pytest_cache", "**/.venv", "**/.ruff_cache", "**/.mypy_cache"]

# us-frb-model is ~251MB on disk but needs only two files at RUNTIME:
#   vendor/data_only_package/LONGBASE.TXT            4.4MB  (the data vintage)
#   vendor/pyfrbus_package/models/model.xml          533KB  (the equations)
# Everything else under vendor/ is provenance material for the validation
# workflow: the original .zip archives sitting alongside their own unpacked
# contents (5.9MB of pure duplication), the Fed's EViews-era frbus_package
# (3MB of PDFs and .prg files), the pyfrbus reference implementation and its
# docs (only ever run by the model repo's own vendor-reference CI job, in a
# throwaway venv), the EViews database and HISTDATA, and the committed vendor
# reference CSV under tests/.
#
# Both files are resolved by policyengine_macro.core._frbus_repo() from
# `frbus.__file__` — with the editable install below that is
# <repo>/src/frbus/__init__.py, so the repo root is two levels up and the two
# vendor paths must stay at their original locations. They do; only their
# siblings are dropped. tests/test_frbus.py exercises exactly this resolution
# path, and the exclusion list is verified by installing a filtered copy.
_FRB_IGNORE = [
    "**/*.zip",                              # archives duplicating unpacked dirs
    "vendor/frbus_package/**",               # EViews package: docs, mods, programs
    "vendor/pyfrbus_package/docs/**",
    "vendor/pyfrbus_package/demos/**",
    "vendor/pyfrbus_package/pyfrbus/**",     # reference impl; not imported here
    "vendor/data_only_package/eviews_database/**",
    "vendor/data_only_package/HISTDATA.TXT",  # history; we simulate from LONGBASE
    "tests/**",
    "scripts/**",
    "uv.lock",
]
# The equivalent as shell `find -delete` for the github-clone branch, which
# has no per-path ignore hook. Kept beside _FRB_IGNORE so the two cannot drift.
_FRB_PRUNE = " && ".join([
    f"rm -f {FRB_REPO}/vendor/*.zip",
    f"rm -rf {FRB_REPO}/vendor/frbus_package",
    f"rm -rf {FRB_REPO}/vendor/pyfrbus_package/docs "
    f"{FRB_REPO}/vendor/pyfrbus_package/demos "
    f"{FRB_REPO}/vendor/pyfrbus_package/pyfrbus",
    f"rm -rf {FRB_REPO}/vendor/data_only_package/eviews_database "
    f"{FRB_REPO}/vendor/data_only_package/HISTDATA.TXT",
    f"rm -rf {FRB_REPO}/tests {FRB_REPO}/scripts",
    # Fail the BUILD, not the first user request, if pruning took a file the
    # adapters need. A missing model.xml would otherwise surface as a
    # FileNotFoundError from the first frbus_shock call in production.
    f"test -f {FRB_REPO}/vendor/data_only_package/LONGBASE.TXT",
    f"test -f {FRB_REPO}/vendor/pyfrbus_package/models/model.xml",
])

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "openpyxl",
        "requests",
        "click",
        "mcp[cli]>=1.9",  # needs FastMCP.streamable_http_app()
        # PolicyEngine microsimulation (household calculator tools). Importing
        # it loads the full UK+US country models (~20s), so policyengine_macro.core
        # imports it lazily inside the pe_* adapters — module import at
        # container start stays fast; only the first policyengine tool call
        # in a fresh container pays the load.
        "policyengine[models]>=4,<5",  # [models] extra REQUIRED: base install leaves pe.uk/pe.us as None
    )
)

if GITHUB_SOURCE:
    # force_build=True: the clone command string never changes, so without it
    # Modal caches this layer and every redeploy reuses a STALE checkout —
    # model fixes merged to OBR/BoE main would never reach production. Forcing a
    # rebuild re-clones the current main on each github-source deploy (~10s).
    image = image.apt_install("git").run_commands(
        f"git clone --depth 1 {OBR_URL} {OBR_REPO}",
        f"git clone --depth 1 {BOE_URL} {BOE_REPO}",
        f"git clone --depth 1 {FRB_URL} {FRB_REPO}",
        _FRB_PRUNE,
        force_build=True,
    )
else:
    image = (
        image
        .add_local_dir(
            OBR_REPO,
            remote_path=OBR_REPO,
            copy=True,
            ignore=_IGNORE + ["dashboard/**", "uv.lock", "outputs/**"],
        )
        .add_local_dir(BOE_REPO, remote_path=BOE_REPO, copy=True,
                       ignore=_IGNORE + ["docs/**"])
        .add_local_dir(FRB_REPO, remote_path=FRB_REPO, copy=True,
                       ignore=_IGNORE + _FRB_IGNORE)
    )

image = (
    image
    .add_local_dir(INTEGRATION, remote_path=f"{HOME}/macro/integration",
                   copy=True, ignore=_IGNORE + ["modal_app.py"])
    # Editable installs keep each package's __file__ inside its repo, so the
    # repos' data/ and results/ directories resolve exactly as on the laptop.
    # Editable installs keep frbus.__file__ inside its repo too, which is how
    # _frbus_repo() finds vendor/ — a wheel install would NOT work, because
    # vendor/ is not package data.
    .run_commands(
        f"pip install -e {OBR_REPO} -e {BOE_REPO} -e {FRB_REPO} "
        f"-e {HOME}/macro/integration"
    )
)

app = modal.App("policyengine-macro-mcp")

# Persistent cache for PolicyEngine population microdata: the HuggingFace
# download cache (HF_HOME) and the derived per-year .h5 datasets both live
# on this volume, so the first population_reform_impact call pays the
# download/build once and every later container reuses it.
# Legacy path kept deliberately: it matches the data already on the volume.
CACHE_DIR = "/root/.cache/policyengine-macro"
# Volume keeps its legacy name deliberately; renaming needs maintainer action.
pe_data_volume = modal.Volume.from_name("policyengine-macro-pe-data", create_if_missing=True)


@app.function(
    image=image.env({
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "POLICYENGINE_MACRO_PE_DATA_DIR": f"{CACHE_DIR}/policyengine-data",
    }),
    cpu=4,
    memory=8192,            # UK population run peaks ~1.8GB; headroom for 2+
    timeout=600,
    min_containers=0,       # scale to zero: no idle cost
    scaledown_window=300,   # stay warm 5 min between calls, then sleep
    max_containers=3,       # spend cap
    secrets=[modal.Secret.from_name("macromod-hf")],
    volumes={CACHE_DIR: pe_data_volume},
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def serve():
    import os
    # Accept either variable name; policyengine reads HUGGING_FACE_TOKEN.
    if "HUGGING_FACE_TOKEN" not in os.environ and os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_TOKEN"] = os.environ["HF_TOKEN"]

    from policyengine_macro import core
    from policyengine_macro.mcp_server import mcp

    # Warm the cheap in-process cache (parses committed results/*.md only —
    # NOT a model estimation, which would make cold starts take minutes).
    try:
        core.svar_summary()
    except Exception:
        pass

    # Stateless streamable HTTP: no session pinning, safe across autoscaled
    # containers and reconnecting clients.
    mcp.settings.stateless_http = True

    # The SDK's DNS-rebinding protection only allows localhost Hosts by
    # default and returns "421 Invalid Host header" behind Modal's proxy.
    # Allow our public hostname (rebinding is a non-issue for a TLS-only,
    # fixed public hostname).
    from mcp.server.transport_security import TransportSecuritySettings

    mcp.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=["policyengine--policyengine-macro-mcp-serve.modal.run"],
        allowed_origins=["*"],
    )
    return mcp.streamable_http_app()


@app.function(
    image=image.env({
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "POLICYENGINE_MACRO_PE_DATA_DIR": f"{CACHE_DIR}/policyengine-data",
    }),
    timeout=600,
    memory=8192,
    cpu=4,
    volumes={CACHE_DIR: pe_data_volume},
    secrets=[modal.Secret.from_name("macromod-hf")],
)
def check_overlay_gate() -> dict:
    """Empirical gate for the issue-#11 input-scaling overlay (maintenance
    entrypoint, not part of the served app): a 0.99 earnings scaling applied
    via Dynamic(simulation_modifier=...) must move the population aggregates.
    Run with:  modal run modal_app.py::check_overlay_gate
    """
    from policyengine_macro import core
    from policyengine_macro.assumptions import EconomicAssumptions

    ea = EconomicAssumptions(
        source="empirical-gate",
        start_year=2026,
        earnings_factor=0.99,
        labour_supply_factor=1.0,
        interest_rate_baseline=0.0,
        interest_rate_reform=0.0,
    )
    modifier = ea.input_scaling_modifier()
    assert modifier is not None, "factor 0.99 must produce a modifier"
    # No-op statutory reform (basic rate at its current 20%) so the only
    # active ingredient is the modifier; the tool rejects an empty reform.
    noop = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.20}
    scaled = core.pe_population_impact(
        country="uk", reform=noop, year=2026, reform_modifier=modifier
    )
    out = {
        "net_income_change_bn": scaled["household_net_income_change_bn"],
        "budgetary_impact_bn": scaled["budgetary_impact_bn"],
        "losers": scaled["losers"],
        "winners": scaled["winners"],
        "bites": abs(scaled["household_net_income_change_bn"]) > 1.0,
    }
    print("GATE RESULT:", out)
    return out
