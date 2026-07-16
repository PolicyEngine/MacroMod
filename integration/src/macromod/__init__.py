"""MacroMod integration layer.

Wraps the suite's models behind one API:

- ``score_reform``: a PolicyEngine reform (the shared ``{parameter_path:
  value}`` dict) through a chosen macro model.
- OBR macroeconomic model emulator (``obr_macro``): raw variable shocks via
  ``obr_shock``.
- UK SVAR / BVAR model (``boe_var``): forecasts and structural shock readings.

Same functions are exposed via a CLI (``macromod``) and an MCP server
(``python -m macromod.mcp_server``).
"""

from macromod.core import (
    score_reform,
    obr_shock,
    obr_list_variables,
    og_score_reform,
    svar_forecast,
    svar_latest_shocks,
    svar_summary,
)

__all__ = [
    "score_reform",
    "obr_shock",
    "obr_list_variables",
    "og_score_reform",
    "svar_forecast",
    "svar_latest_shocks",
    "svar_summary",
]

__version__ = "0.1.0"
