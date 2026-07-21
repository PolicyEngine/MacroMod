"""PolicyEngine Macro CLI. Human-readable tables by default; --json for machine output."""

from __future__ import annotations

import json

import click

from policyengine_macro import core
from policyengine_macro import capabilities


def _emit_json(obj) -> None:
    click.echo(json.dumps(obj, indent=2))


def _table(rows: list[dict], columns: list[str]) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    head = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join(
        "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows
    )
    return f"{head}\n{sep}\n{body}"


@click.group()
def main() -> None:
    """PolicyEngine Macro: unified CLI over the OBR emulator and the UK SVAR model."""


@main.command("model-status")
@click.argument("model_id", required=False)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def model_status(model_id, as_json):
    """Show supported uses, access, and limitations for one or all models."""
    try:
        rows = ([capabilities.get_status(model_id)] if model_id
                else capabilities.list_capabilities())
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if as_json:
        _emit_json(rows[0] if model_id else rows)
        return
    click.echo(_table([
        {
            "model": row["model_id"],
            "country": ",".join(row["geography"]),
            "status": row["status"],
            "access": "; ".join(row["access"]),
        }
        for row in rows
    ], ["model", "country", "status", "access"]))


@main.command()
@click.option("--country", type=click.Choice(["uk", "us"]), default="uk",
              show_default=True)
@click.option("--reform", required=True,
              help='PolicyEngine reform JSON, e.g. \'{"gov.hmrc.income_tax.rates.uk[0].rate":0.21}\' '
                   "(same shape as `pe-macro population-impact`).")
@click.option("--model", required=True,
              type=click.Choice(list(core.SCORE_MODELS)),
              help="Scoring model: og (OG-UK steady state; slow), obr (OBR "
                   "emulator via the microsim static-costing bridge) or "
                   "microsim (PolicyEngine population costing, no macro "
                   "feedback).")
@click.option("--year", default=2026, show_default=True, help="Reform start year.")
@click.option("--max-iter", default=250, show_default=True,
              help="og only: solver iteration cap per steady-state solve.")
@click.option("--years", default=5, show_default=True,
              help="obr only: costing window length in years.")
@click.option("--dataset", default=None,
              help="obr/microsim only: microdata dataset name override.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def score(country, reform, model, year, max_iter, years, dataset, as_json):
    """Score a PolicyEngine reform with a scoring model of the suite.

    One reform vocabulary: the same {parameter_path: value} dict as
    `pe-macro population-impact`. Every result carries a common `score`
    block for cross-model comparison (`pe-macro compare`). For raw OBR
    variable shocks in model units, use `pe-macro obr-shock`.
    """
    try:
        res = core.score_reform(
            country=country, reform=_json_opt(reform, "reform"), model=model,
            start_year=year, max_iter=max_iter, years=years, dataset=dataset,
        )
    except (NotImplementedError, ValueError, ImportError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    if as_json:
        _emit_json(res)
        return
    if model == "og":
        _echo_og_impact(res)
    else:
        _echo_score_block(res["score"])


def _echo_score_block(score: dict) -> None:
    """Render one common ScoreResult block as a table."""
    click.echo(f"{score['model']} ({score['model_class']}, "
               f"{score['country'].upper()}, {score['horizon']})")
    click.echo(f"Reform: {score['reform']}\n")
    rows = []
    for name, q in score["quantities"].items():
        rows.append({
            "quantity": name,
            "delta_bn": q.get("delta_bn"),
            "delta_pct": q.get("delta_pct"),
            "units": q["units"],
        })
    click.echo(_table(rows, ["quantity", "delta_bn", "delta_pct", "units"]))
    for label, items in (("Assumptions", score.get("assumptions") or []),
                         ("Caveats", score.get("caveats") or [])):
        if items:
            click.echo(f"\n{label}:")
            for it in items:
                click.echo(f"  - {it}")


@main.command()
@click.option("--country", type=click.Choice(["uk", "us"]), default="uk",
              show_default=True)
@click.option("--reform", required=True,
              help='PolicyEngine reform JSON (same shape as `pe-macro score`).')
@click.option("--models", default="microsim,obr", show_default=True,
              help="Comma-separated scoring models (og, obr, microsim).")
@click.option("--year", default=2026, show_default=True, help="Reform start year.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON list of ScoreResults.")
def compare(country, reform, models, year, as_json):
    """Run one reform through supported adapters, with comparability warnings.

    Runs `score` once per model and renders one table from the common
    ScoreResult blocks (PolicyEngine/macro#10)."""
    reform_dict = _json_opt(reform, "reform")
    scores = []
    for model in [m.strip() for m in models.split(",") if m.strip()]:
        try:
            res = core.score_reform(
                country=country, reform=reform_dict, model=model,
                start_year=year,
            )
        except (NotImplementedError, ValueError, ImportError, RuntimeError) as e:
            raise click.ClickException(f"{model}: {e}") from e
        scores.append(res["score"])
    if as_json:
        _emit_json(scores)
        return
    click.echo(f"Reform: {reform_dict}  ({country.upper()}, from {year})\n")
    rows = []
    for s in scores:
        for name, q in s["quantities"].items():
            rows.append({
                "model": s["model"],
                "class": s["model_class"],
                "horizon": s["horizon"],
                "quantity": name,
                "delta_bn": q.get("delta_bn"),
                "delta_pct": q.get("delta_pct"),
                "units": q["units"],
                "time_basis": q["time_basis"],
                "comparability": q["comparability"],
            })
    click.echo(_table(rows, ["model", "class", "horizon", "quantity",
                             "delta_bn", "delta_pct", "units", "time_basis",
                             "comparability"]))
    click.echo("\nThese results use different horizons and mechanisms. "
               "Treat related-not-like-for-like rows as complementary: they "
               "must not be added, averaged, or ranked.")


@main.command("obr-shock")
@click.option("--var", required=True, help="Policy variable to shock (see `pe-macro variables`).")
@click.option("--shock", required=True, type=float,
              help="Shock size; units depend on the variable (£m/quarter for CGG, decimal for TCPRO).")
@click.option("--periods", default=12, show_default=True, help="Quarters the shock is applied.")
@click.option("--name", default=None, help="Label for the reform.")
@click.option("--investment-closure/--no-investment-closure", default=None,
              help="Cost-of-capital investment channel; omit for the safe "
                   "per-variable default (on for TCPRO, off otherwise).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def obr_shock(var, shock, periods, name, investment_closure, as_json):
    """Shock one OBR variable directly, in model units (escape hatch)."""
    res = core.obr_shock(
        var=var, shock=shock, periods=periods, name=name,
        investment_closure=investment_closure,
    )
    if as_json:
        _emit_json(res)
        return
    click.echo(f"Reform: {res['name']}  (var={res['var']}, shock={res['shock']:+g}, "
               f"periods={res['periods']}, investment_closure={res['investment_closure']})")
    click.echo(_table(res["results"],
                      ["period", "delta_gdp_bn", "pct_gdp", "delta_cons_m", "delta_if_m"]))
    click.echo(f"\nCumulative GDP effect over shocked periods: "
               f"£{res['cumulative_delta_gdp_bn_over_shock_periods']}bn")
    click.echo(f"Peak GDP effect: {res['peak_pct_gdp']}% in {res['peak_period']}")


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def variables(as_json):
    """List commonly shocked OBR policy variables."""
    res = core.obr_list_variables()
    if as_json:
        _emit_json(res)
        return
    click.echo(_table(res, ["var", "description", "units", "investment_closure"]))


@main.command("frbus-shock")
@click.option("--var", required=True,
              help="Lever to shock (see `pe-macro frbus-variables`).")
@click.option("--shock", required=True, type=float,
              help="Shock size; UNITS DIFFER PER LEVER (pp for rffintay_aerr, "
                   "decimal rate for trp_aerr, log points of quarterly growth "
                   "for egfe_aerr/ecnia_aerr).")
@click.option("--start", default=core.FRBUS_DEFAULT_START, show_default=True,
              help="First shocked quarter, e.g. 2026Q1.")
@click.option("--periods", default=1, show_default=True,
              help="Quarters the shock is held (1 = single-quarter impulse).")
@click.option("--horizon", default=core.FRBUS_DEFAULT_HORIZON, show_default=True,
              help="Quarters simulated and reported.")
@click.option("--policy-rule", default="inertial_taylor", show_default=True,
              type=click.Choice(sorted(core.FRBUS_POLICY_RULES)),
              help="Monetary policy reaction; changes the answer materially.")
@click.option("--variable", "variables", multiple=True,
              help="Extra model variable to report (repeatable).")
@click.option("--name", default=None, help="Label for the experiment.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def frbus_shock(var, shock, start, periods, horizon, policy_rule, variables,
                name, as_json):
    """Shock one FRB/US variable, in model units (US escape hatch)."""
    res = core.frbus_shock(
        var=var, shock=shock, start=start, periods=periods, horizon=horizon,
        policy_rule=policy_rule, variables=list(variables) or None, name=name,
    )
    if as_json:
        _emit_json(res)
        return
    click.echo(f"Experiment: {res['name']}  (var={res['var']}, "
               f"shock={res['shock']:+g}, start={res['start']}, "
               f"periods={res['periods']}, rule={res['policy_rule']})")
    click.echo(f"Units: {res['units']}")
    columns = ["period"] + [k for k in res["results"][0] if k != "period"]
    click.echo(_table(res["results"], columns))
    click.echo("\nPeak absolute deviations:")
    for v, peak in res["peaks"].items():
        click.echo(f"  {v:10s} {peak['value']:+.4f} in {peak['period']}"
                   f"   ({res['series_meaning'][v]})")
    if res.get("warning"):
        click.echo(f"\nWARNING: {res['warning']}")


@main.command("frbus-variables")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def frbus_variables(as_json):
    """List the shockable FRB/US levers and their units."""
    res = core.frbus_list_variables()
    if as_json:
        _emit_json(res)
        return
    click.echo(_table(res, ["var", "description", "units", "typical_shock",
                            "requires_policy_rule"]))


@main.command("frbus-summary")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def frbus_summary(as_json):
    """FRB/US model metadata and validation provenance (instant)."""
    res = core.frbus_summary()
    if as_json:
        _emit_json(res)
        return
    click.echo(f"{res['model']} — {res['implementation']}")
    click.echo(f"  {res['equations']} endogenous equations, "
               f"{res['data_vintage']}, {res['expectations']}")
    click.echo(f"  source: {res.get('source', res.get('source_error'))}\n")
    click.echo("Policy rules:")
    for rule in res["policy_rules"]:
        click.echo(f"  {rule['rule']:18s} {rule['description']}")
    val = res["validation"]
    click.echo("\nValidation:")
    click.echo(f"  tracking invariant: {val['tracking_invariant']['value']:.1e} "
               f"(gate {val['tracking_invariant']['gate']:.0e})")
    click.echo(f"  vs pyfrbus 1.0.0:   {val['vs_vendor_pyfrbus']['value']:.1e} "
               f"(gate {val['vs_vendor_pyfrbus']['gate']:.0e})")
    mon = val["monetary_tightening_properties"]
    click.echo(f"  {mon['shock']}: xgdp trough {mon['xgdp_trough_pct']}%, "
               f"lur peak {mon['lur_peak_pp']}pp")
    click.echo(f"\nReform bridge: {res['reform_bridge']}")


@main.command()
@click.option("--horizons", default=12, show_default=True, help="Forecast horizon in quarters.")
@click.option("--draws", default=500, show_default=True, help="Posterior draws (more = slower, smoother).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def forecast(horizons, draws, as_json):
    """UK SVAR forecast: YoY GDP growth and CPI inflation with bands."""
    res = core.svar_forecast(horizons=horizons, draws=draws)
    if as_json:
        _emit_json(res)
        return
    click.echo(f"UK SVAR forecast from {res['forecast_origin']} "
               f"({res['draws']} draws, {res['accepted_draws']} accepted, ESS {res['ess']})")
    for key, label in [("gdp_growth_yoy", "YoY GDP growth (%)"),
                       ("cpi_inflation_yoy", "YoY CPI inflation (%)")]:
        click.echo(f"\n{label}")
        click.echo(_table(res[key], ["quarter", "median", "lo68", "hi68", "lo90", "hi90"]))


@main.command()
@click.option("--draws", default=500, show_default=True, help="Posterior draws.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def shocks(draws, as_json):
    """P(sign) of the identified structural shocks in the latest quarter."""
    res = core.svar_latest_shocks(draws=draws)
    if as_json:
        _emit_json(res)
        return
    click.echo(f"Structural shocks in {res['quarter']} "
               f"({res['draws']} draws, {res['accepted_draws']} accepted, ESS {res['ess']})\n")
    click.echo(_table(res["shocks"], ["shock", "p_positive", "p_negative"]))
    click.echo()
    for s in res["shocks"]:
        click.echo(f"- {s['reading']}")


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def summary(as_json):
    """Headline SVAR results parsed from the repo's committed summaries (instant)."""
    res = core.svar_summary()
    if set(res) == {"error"}:
        raise click.ClickException(res["error"])
    rep = res.get("replication", {})
    fr_check = res.get("forecast_revision", {})
    if "error" in rep and "error" in fr_check:
        raise click.ClickException(
            "no parseable SVAR results — replication: "
            f"{rep['error']}; forecast revision: {fr_check['error']}"
        )
    if as_json:
        _emit_json(res)
        return
    click.echo("Replication (results/summary.md)")
    if "error" in rep:
        click.echo(f"  error: {rep['error']}", err=True)
    for ln in rep.get("metadata", []):
        click.echo(f"  {ln}")
    fevd = rep.get("fevd_1yr_headline", [])
    if fevd:
        click.echo("\nFEVD at 1-year horizon (median shares)")
        click.echo(_table(fevd, list(fevd[0].keys())))
    fr = res.get("forecast_revision", {})
    click.echo("\nForecast-revision exercise (results/forecast_summary.md)")
    if "error" in fr:
        click.echo(f"  error: {fr['error']}", err=True)
    for ln in fr.get("metadata", []):
        click.echo(f"  {ln}")
    signs = fr.get("latest_shock_signs", [])
    if signs:
        click.echo("\nP(sign) of the identified shocks at T")
        click.echo(_table(signs, list(signs[0].keys())))
    for ln in fr.get("composite_irf", []):
        click.echo(f"  {ln}")


def _json_opt(value, name):
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"--{name} must be valid JSON: {e}") from e


_PEOPLE_HELP = 'JSON list of person dicts, e.g. \'[{"age":35,"employment_income":50000}]\'.'
_REFORM_HELP = (
    'JSON reform dict, e.g. \'{"gov.hmrc.income_tax.rates.uk[0].rate":0.25}\'. '
    'A value may also be a single-effective-date dict, e.g. '
    '\'{"gov.hmrc.income_tax.rates.uk[0].rate":{"2026-01-01":0.25}}\'; '
    'date ranges ("2026-01-01.2029-12-31") are not supported.'
)


def _pe_common_options(fn):
    for opt in reversed([
        click.option("--country", type=click.Choice(["uk", "us"]), required=True),
        click.option("--people", required=True, help=_PEOPLE_HELP),
        click.option("--year", default=2026, show_default=True),
        click.option("--benunit", default=None, help="UK only: JSON benunit dict."),
        click.option("--tax-unit", "tax_unit", default=None,
                     help='US only: JSON tax unit dict, e.g. \'{"filing_status":"SINGLE"}\'.'),
        click.option("--household", default=None,
                     help='JSON household dict, e.g. \'{"state_code_str":"CA"}\' (US).'),
        click.option("--json", "as_json", is_flag=True, help="Emit JSON."),
    ]):
        fn = opt(fn)
    return fn


def _echo_summary(label: str, summary: dict, sym: str) -> None:
    click.echo(label)
    for k, v in summary.items():
        if isinstance(v, list):
            v = ", ".join(f"{sym}{x:,.0f}" for x in v)
        elif isinstance(v, (int, float)):
            v = f"{sym}{v:,.0f}"
        click.echo(f"  {k:32} {v}")


@main.command()
@_pe_common_options
@click.option("--reform", default=None, help=_REFORM_HELP)
def household(country, people, year, benunit, tax_unit, household, as_json, reform):
    """Calculate taxes and benefits for a household (PolicyEngine)."""
    res = core.pe_household(
        country=country,
        people=_json_opt(people, "people"),
        year=year,
        reform=_json_opt(reform, "reform"),
        benunit=_json_opt(benunit, "benunit"),
        tax_unit=_json_opt(tax_unit, "tax-unit"),
        household=_json_opt(household, "household"),
    )
    if as_json:
        _emit_json(res)
        return
    sym = "£" if res["country"] == "uk" else "$"
    click.echo(f"PolicyEngine {res['country'].upper()} household, {res['year']}"
               + (f" (reform: {res['reform']})" if res["reform"] else ""))
    _echo_summary("Summary:", res["summary"], sym)


@main.command("household-impact")
@_pe_common_options
@click.option("--reform", required=True, help=_REFORM_HELP)
def household_impact(country, people, year, benunit, tax_unit, household, as_json, reform):
    """Baseline-vs-reform impact of a reform on one household (PolicyEngine)."""
    res = core.pe_household_impact(
        country=country,
        people=_json_opt(people, "people"),
        reform=_json_opt(reform, "reform"),
        year=year,
        benunit=_json_opt(benunit, "benunit"),
        tax_unit=_json_opt(tax_unit, "tax-unit"),
        household=_json_opt(household, "household"),
    )
    if as_json:
        _emit_json(res)
        return
    sym = "£" if res["country"] == "uk" else "$"
    click.echo(f"PolicyEngine {res['country'].upper()} reform impact, {res['year']}")
    click.echo(f"Reform: {res['reform']}\n")
    _echo_summary("Baseline:", res["baseline"], sym)
    _echo_summary("\nWith reform:", res["with_reform"], sym)
    _echo_summary("\nChange:", {k: v for k, v in res["change"].items() if v is not None}, sym)


@main.command("population-impact")
@click.option("--country", type=click.Choice(["uk", "us"]), default="uk",
              show_default=True)
@click.option("--reform", required=True, help=_REFORM_HELP)
@click.option("--year", default=2026, show_default=True)
@click.option("--dataset", default=None,
              help="Dataset name (default: enhanced_frs_2023_24 for UK).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def population_impact(country, reform, year, dataset, as_json):
    """Population-level reform score (PolicyEngine microsimulation).

    Budgetary impact in £bn/$bn per year plus decile impacts. First UK run
    downloads private microdata (set HUGGING_FACE_TOKEN); afterwards a score
    takes tens of seconds.
    """
    res = core.pe_population_impact(
        country=country, reform=_json_opt(reform, "reform"),
        year=year, dataset=dataset,
    )
    if as_json:
        _emit_json(res)
        return
    sym = "£" if res["country"] == "uk" else "$"
    click.echo(f"PolicyEngine {res['country'].upper()} population impact, "
               f"{res['year']} ({res['dataset']}, "
               f"{res['n_households']:,} households)")
    click.echo(f"Reform: {res['reform']}\n")
    click.echo(res["headline"])
    click.echo(f"Budgetary impact: {sym}{res['budgetary_impact_bn']}bn/year "
               f"({res['budgetary_impact_basis']})")
    click.echo(f"Household net income change: "
               f"{sym}{res['household_net_income_change_bn']}bn/year")
    click.echo(f"Winners: {res['winners']:,}   Losers: {res['losers']:,}\n")
    click.echo(_table(res["decile_impacts"],
                      ["decile", "avg_income_change", "relative_change_pct",
                       "count_better_off", "count_worse_off"]))


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def parameters(as_json):
    """List curated PolicyEngine reform parameters (verified paths)."""
    res = core.pe_list_common_parameters()
    if as_json:
        _emit_json(res)
        return
    cols = ["country", "path", "description", "unit"]
    for extra in ("baseline_value", "live"):
        if any(extra in r for r in res):
            cols.append(extra)
    click.echo(_table(res, cols))
    dead = [r for r in res if r.get("live") is False]
    if dead:
        click.echo(
            f"\nWARNING: {len(dead)} parameter(s) failed live resolution "
            "(static catalogue shown for them):", err=True,
        )
        for r in dead:
            click.echo(f"  {r['path']}: {r.get('live_error')}", err=True)


def _echo_og_impact(res: dict) -> None:
    click.echo("OG-UK steady-state reform score")
    click.echo(f"Reform: {res['reform']} (from {res['start_year']})")
    click.echo(f"Assumptions: {res['assumptions']}\n")
    imp = res["impact"]
    rows = []
    for k in ("gdp", "consumption", "investment", "government",
              "tax_revenue", "debt"):
        rows.append({
            "aggregate": k,
            "level (£bn)": imp["levels_bn"][k],
            "change (£bn)": imp["changes_bn"][f"{k}_change"],
            "change (%)": imp["changes_pct"][f"{k}_pct"],
        })
    click.echo(_table(rows, ["aggregate", "level (£bn)", "change (£bn)", "change (%)"]))
    ir = imp["interest_rate"]
    click.echo(f"\nInterest rate: {ir['baseline']} -> {ir['reform']}")


@main.command("og-score")
@click.option("--reform", required=True, help=_REFORM_HELP)
@click.option("--year", default=2026, show_default=True, help="Reform start year.")
@click.option("--max-iter", default=250, show_default=True,
              help="Max solver iterations for each steady-state solve.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def og_score(reform, year, max_iter, as_json):
    """Score a reform with the OG-UK model (alias for `score --model og`; slow: ~10 min)."""
    try:
        res = core.og_score_reform(
            reform=_json_opt(reform, "reform"), start_year=year,
            max_iter=max_iter,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if as_json:
        _emit_json(res)
        return
    _echo_og_impact(res)


@main.command("dynamic-score")
@click.option("--reform", required=True, help=_REFORM_HELP)
@click.option("--start-year", "start_year", default=2026, show_default=True,
              help="Reform start year (OG solve and microsim year).")
@click.option("--max-iter", default=250, show_default=True,
              help="Max solver iterations for each OG steady-state solve.")
@click.option("--dataset", default=None,
              help="Microdata dataset name override.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def dynamic_score(reform, start_year, max_iter, dataset, as_json):
    """Dynamic population score: OG-UK macro overlay on the microsim.

    Alias for `score --model og+microsim` (UK only; slow: two OG
    steady-state solves, baseline cached, plus one microsim run).
    """
    try:
        res = core.dynamic_population_reform_impact(
            country="uk", reform=_json_opt(reform, "reform"),
            year=start_year, max_iter=max_iter, dataset=dataset,
        )
    except (ValueError, ImportError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    if as_json:
        _emit_json(res)
        return
    ea = res["economic_assumptions"]
    click.echo("Dynamic score: OG-UK overlay + PolicyEngine microsim")
    click.echo(f"Earnings factor: {ea['earnings_factor']}   "
               f"Labour-supply factor: {ea['labour_supply_factor']}   "
               f"r: {ea['interest_rate_baseline']} -> "
               f"{ea['interest_rate_reform']}\n")
    _echo_score_block(res["score"])
    micro = res["microsim"]
    click.echo(f"\n{micro['headline']}")
    click.echo(_table(micro["decile_impacts"],
                      ["decile", "avg_income_change", "relative_change_pct",
                       "count_better_off", "count_worse_off"]))


@main.command("og-baseline")
@click.option("--year", default=2026, show_default=True, help="Start year.")
@click.option("--max-iter", default=250, show_default=True,
              help="Max solver iterations.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def og_baseline(year, max_iter, as_json):
    """Baseline OG-UK steady state (slow: ~5 min; model units)."""
    res = core.og_baseline(start_year=year, max_iter=max_iter)
    if as_json:
        _emit_json(res)
        return
    click.echo(f"OG-UK baseline steady state, start year {res['start_year']}")
    click.echo(f"Assumptions: {res['assumptions']}\n")
    for k, v in res["steady_state_model_units"].items():
        click.echo(f"  {k:12} {v}")


if __name__ == "__main__":
    main()
