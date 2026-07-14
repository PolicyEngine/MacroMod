"""MacroMod MCP server (stdio transport).

Run with:  python -m macromod.mcp_server

Exposes the same adapter functions as the `macromod` CLI as MCP tools.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from macromod import core

mcp = FastMCP("macromod")


@mcp.tool()
def score_reform(
    var: str,
    shock: float,
    periods: int = 12,
    name: str | None = None,
    investment_closure: bool = False,
) -> dict:
    """Score a UK fiscal policy reform with the OBR macroeconomic model emulator.

    Runs a baseline and a shocked solve of the OBR model and returns per-quarter
    GDP deltas plus a headline cumulative GDP effect.

    Args:
        var: Policy variable to shock. Common choices (see list_reform_variables):
            CGG (real government consumption), TCPRO (corporation tax main rate),
            CGIPS (nominal central government investment).
        shock: Shock size, applied each quarter. UNITS DEPEND ON THE VARIABLE:
            CGG and CGIPS are in £ million per quarter (e.g. 1250 = £5bn/year);
            TCPRO is a rate change in decimal (e.g. -0.05 = 5 percentage point cut).
        periods: Number of quarters the shock is applied (default 12 = 3 years).
        name: Optional label for the reform.
        investment_closure: Set True for corporation tax (TCPRO) shocks — it
            activates the cost-of-capital investment channel; without it TCPRO
            shocks have no effect on business investment.

    Returns per-period rows (period, delta_gdp_bn, pct_gdp, delta_cons_m,
    delta_if_m), the cumulative GDP effect in £bn over the shocked periods, and
    the peak percent-of-GDP effect. Takes roughly 10-60 seconds.
    """
    return core.obr_score_reform(
        var=var, shock=shock, periods=periods, name=name,
        investment_closure=investment_closure,
    )


@mcp.tool()
def list_reform_variables() -> list[dict]:
    """List the OBR policy variables commonly shocked with score_reform.

    Returns each variable's code, description, shock units (£m per quarter vs
    decimal rate change), and whether it needs investment_closure=True.
    Instant; call this before score_reform if unsure of units.
    """
    return core.obr_list_variables()


@mcp.tool()
def forecast_uk(horizons: int = 12, draws: int = 500) -> dict:
    """Forecast UK YoY GDP growth and CPI inflation with the UK SVAR model.

    Estimates a Bayesian VAR (1992Q1-2023Q2 sample, sign-identified structural
    shocks) and simulates the predictive distribution from the latest data
    quarter. Returns, per future quarter, the median and 68%/90% bands for YoY
    GDP growth and YoY CPI inflation, both in percent.

    Args:
        horizons: Forecast horizon in quarters (default 12 = 3 years).
        draws: Posterior draws (default 500, responds in tens of seconds;
            can be raised, e.g. 2000-6000, for smoother bands at the cost of
            minutes of runtime). Results are cached in-process, so repeated
            calls with the same (horizons, draws) are instant.
    """
    return core.svar_forecast(horizons=horizons, draws=draws)


@mcp.tool()
def latest_shocks(draws: int = 500) -> dict:
    """Structural-shock reading for the latest data quarter from the UK SVAR.

    For each of the 6 identified shocks (world demand/energy/supply, UK
    demand/supply/monetary policy) returns the posterior probability the shock
    was positive vs negative in the latest quarter, plus a one-line reading.

    Args:
        draws: Posterior draws (default 500; can be raised for precision).
            Cached in-process, so repeat calls are instant.
    """
    return core.svar_latest_shocks(draws=draws)


@mcp.tool()
def model_summary() -> dict:
    """Headline results of the UK SVAR replication, parsed from committed files.

    Instant (no estimation): returns last-run metadata (draws, acceptance, ESS,
    sample) and headline 1-year FEVD shares (how much of UK GDP and CPI variance
    is explained by global vs domestic identified shocks), plus the latest
    forecast-revision exercise's shock-sign table.
    """
    return core.svar_summary()


@mcp.tool()
def calculate_household(
    country: str,
    people: list[dict],
    year: int = 2026,
    reform: dict | None = None,
    benunit: dict | None = None,
    tax_unit: dict | None = None,
    household: dict | None = None,
) -> dict:
    """Calculate taxes and benefits for a custom household with the
    PolicyEngine microsimulation model (full UK/US tax-benefit rules).

    Args:
        country: 'uk' or 'us'.
        people: List of person dicts. All money amounts are ANNUAL and in the
            country's currency (GBP for uk, USD for us), as plain numbers.
            Example: [{"age": 35, "employment_income": 50000}, {"age": 5}].
        year: Tax year (default 2026).
        reform: Optional parametric reform as {parameter_path: value}, e.g.
            {"gov.hmrc.income_tax.rates.uk[0].rate": 0.25} (UK basic rate to
            25%, a decimal) or {"gov.irs.credits.ctc.amount.base[0].amount":
            3000} (US CTC to $3,000). Call list_reform_parameters for verified
            paths and units.
        benunit: UK only, optional benefit-unit dict, e.g.
            {"would_claim_uc": true, "would_claim_child_benefit": true}.
        tax_unit: US only, optional, e.g. {"filing_status": "SINGLE"} or "JOINT".
        household: Optional household dict. US: {"state_code_str": "CA"}.
            UK: {"rent": 12000, "region": "NORTH_WEST"}.

    Returns a dict with a headline `summary` (UK: per-person income tax and
    National Insurance in £, household net income, tax, benefits, Universal
    Credit, Child Benefit; US: federal income tax, payroll tax, state income
    tax, CTC, EITC, household net income in $) plus full per-entity variable
    dicts. Takes a few seconds per call.
    """
    return core.pe_household(
        country=country, people=people, year=year, reform=reform,
        benunit=benunit, tax_unit=tax_unit, household=household,
    )


@mcp.tool()
def household_reform_impact(
    country: str,
    people: list[dict],
    reform: dict,
    year: int = 2026,
    benunit: dict | None = None,
    tax_unit: dict | None = None,
    household: dict | None = None,
) -> dict:
    """What does a tax/benefit reform do to a specific family? Runs the
    PolicyEngine household calculation twice (baseline and reform) and
    returns both summaries plus the change in each headline number.

    Args are as in calculate_household, but `reform` is REQUIRED:
    {parameter_path: value}, e.g. {"gov.hmrc.income_tax.allowances.
    personal_allowance.amount": 15000} raises the UK personal allowance to
    £15,000/year. Rates are decimals (0.25 = 25%); amounts are annual £/$
    unless list_reform_parameters says otherwise (e.g. Child Benefit is
    £/week). The `change` dict and `net_income_change` are reform minus
    baseline, so a positive net_income_change means the family gains.
    """
    return core.pe_household_impact(
        country=country, people=people, reform=reform, year=year,
        benunit=benunit, tax_unit=tax_unit, household=household,
    )


@mcp.tool()
def list_reform_parameters() -> list[dict]:
    """List curated, verified PolicyEngine reform parameters for use in the
    `reform` argument of calculate_household / household_reform_impact.

    Returns ~10 well-known UK and US parameters with their exact path,
    description, and units (decimal rates vs annual £/$ amounts). Every path
    has been verified to resolve. Instant. Other parameter paths from the
    policyengine-uk/-us parameter trees also work, but are not verified here.
    """
    return core.pe_list_common_parameters()


if __name__ == "__main__":
    mcp.run(transport="stdio")
