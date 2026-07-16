"""Deploy the MacroMod MCP server to Modal over streamable HTTP.

    modal deploy integration/modal_app.py

Serves the FastMCP instance from `macromod.mcp_server` (tools: score_reform,
obr_shock, list_reform_variables, forecast_uk, latest_shocks, model_summary,
calculate_household, household_reform_impact, list_reform_parameters,
population_reform_impact) as an
ASGI app at  https://policyengine--macromod-mcp-serve.modal.run/mcp

Both model repos resolve their data files relative to their own repo root
(`Path(__file__)`-relative), and `macromod.core` hardcodes
/Users/janansadeqian/boe-var-model. We therefore bake the repos into the
image at the SAME absolute paths and `pip install -e` them, so every path
resolves in the container with zero patching:
  - obr_macro:  /Users/janansadeqian/obr-macroeconomic-model  (+ data/)
  - boe_var:    /Users/janansadeqian/boe-var-model            (+ data/, results/)

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

POPULATION DATA (population_reform_impact)
------------------------------------------
- Secret "macromod-hf" provides HUGGING_FACE_TOKEN for the private UK
  enhanced-FRS microdata on HuggingFace.
- A modal.Volume ("macromod-pe-data") is mounted at /root/.cache/macromod;
  HF_HOME points the HuggingFace download cache inside it and
  MACROMOD_PE_DATA_DIR puts the derived per-year .h5 files (~92MB/year)
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
(`macromod score --model og`) or a local MCP server instead. If it is ever added,
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
INTEGRATION = str(Path(__file__).parent)

# MACROMOD_IMAGE_SOURCE=github (used by CI) clones the model repos from
# GitHub main at image-build time instead of copying the local checkouts —
# same absolute paths, so all data-file resolution is unchanged.
GITHUB_SOURCE = os.environ.get("MACROMOD_IMAGE_SOURCE") == "github"
OBR_URL = "https://github.com/PolicyEngine/obr-macroeconomic-model"
BOE_URL = "https://github.com/PolicyEngine/boe-var-model"

# Keep the image lean: skip the 412MB dashboard, VCS, caches, docs.
_IGNORE = ["**/.git", "**/.github", "**/__pycache__", "**/*.egg-info",
           "**/.pytest_cache"]

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
        # it loads the full UK+US country models (~20s), so macromod.core
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
    )

image = (
    image
    .add_local_dir(INTEGRATION, remote_path=f"{HOME}/MacroMod/integration",
                   copy=True, ignore=_IGNORE + ["modal_app.py"])
    # Editable installs keep each package's __file__ inside its repo, so the
    # repos' data/ and results/ directories resolve exactly as on the laptop.
    .run_commands(
        f"pip install -e {OBR_REPO} -e {BOE_REPO} "
        f"-e {HOME}/MacroMod/integration"
    )
)

app = modal.App("macromod-mcp")

# Persistent cache for PolicyEngine population microdata: the HuggingFace
# download cache (HF_HOME) and the derived per-year .h5 datasets both live
# on this volume, so the first population_reform_impact call pays the
# download/build once and every later container reuses it.
CACHE_DIR = "/root/.cache/macromod"
pe_data_volume = modal.Volume.from_name("macromod-pe-data", create_if_missing=True)


@app.function(
    image=image.env({
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "MACROMOD_PE_DATA_DIR": f"{CACHE_DIR}/policyengine-data",
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

    from macromod import core
    from macromod.mcp_server import mcp

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
        allowed_hosts=["policyengine--macromod-mcp-serve.modal.run"],
        allowed_origins=["*"],
    )
    return mcp.streamable_http_app()
