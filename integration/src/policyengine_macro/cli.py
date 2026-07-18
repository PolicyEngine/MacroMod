"""PolicyEngine Macro CLI. Human-readable tables by default; --json for machine output."""

from __future__ import annotations

import json

import click

from policyengine_macro import core


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
    """Score the SAME reform through several model classes, side by side.

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
            })
    click.echo(_table(rows, ["model", "class", "horizon", "quantity",
                             "delta_bn", "delta_pct"]))
    click.echo("\nDeltas are each model's own concept (see `score --json` "
               "for units/basis per quantity): steady-state vs budget-window "
               "numbers are NOT directly additive.")


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
