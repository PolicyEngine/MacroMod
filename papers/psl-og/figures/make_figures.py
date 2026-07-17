"""Generate illustrative figures for the psl-og working paper.

All inputs are cheap to load: UN WPP UK demographic CSVs shipped with OG-UK,
the OG-Core default Specifications object (ability matrix e_{j,s}, ellipse
utility parameters), and published tax-function parameter values. No OG
steady-state solve is required or performed.

Colours: Okabe-Ito colorblind-safe palette.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
OGUK_DEMO = "/Users/janansadeqian/OG-UK/oguk/data/demographic"

# Okabe-Ito
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
      "#E69F00", "#56B4E9", "#000000", "#F0E442"]

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def fig_demographics():
    pop = pd.read_csv(f"{OGUK_DEMO}/un_uk_pop.csv", sep="|", skiprows=1)
    fert = pd.read_csv(f"{OGUK_DEMO}/un_uk_fert.csv", sep="|", skiprows=1)
    dth = pd.read_csv(f"{OGUK_DEMO}/un_uk_deaths.csv", sep="|", skiprows=1)

    yr = 2021
    p = (pop[(pop.TimeLabel == yr) & (pop.Sex == "Both sexes")]
         .groupby("AgeStart")["Value"].sum().sort_index())
    f = (fert[fert.TimeLabel == yr].groupby("AgeStart")["Value"].mean()
         .sort_index())
    d = (dth[(dth.TimeLabel == yr) & (dth.Sex == "Both sexes")]
         .groupby("AgeStart")["Value"].sum().sort_index())
    # crude mortality hazard by age: deaths / population
    ages = p.index.values
    mort = (d.reindex(ages).fillna(0) / p).values

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax = axes[0]
    ax.fill_between(ages, p.values / 1e3, color=OI[0], alpha=0.75, lw=0)
    ax.axvspan(20, 100, color=OI[4], alpha=0.12)
    ax.text(60, ax.get_ylim()[1] * 0.02 + max(p.values / 1e3) * 0.93,
            "model ages\n($E{+}1$ to $E{+}S$)", fontsize=7, ha="center")
    ax.set_xlabel("Age")
    ax.set_ylabel("Population (thousands)")
    ax.set_title(f"UK population by single year of age, {yr}", fontsize=9)

    ax = axes[1]
    ax.plot(f.index, f.values, color=OI[1], lw=1.8)
    ax.set_xlabel("Age of mother")
    ax.set_ylabel("Births per 1{,}000 women".replace("{,}", ","))
    ax.set_title(f"Fertility rates by age, {yr}", fontsize=9)

    ax = axes[2]
    ax.semilogy(ages, np.maximum(mort, 1e-6), color=OI[2], lw=1.8)
    ax.set_xlabel("Age")
    ax.set_ylabel(r"Mortality hazard $\rho_s$ (log scale)")
    ax.set_title(f"Implied mortality hazard, {yr}", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_demographics.pdf")
    plt.close(fig)


def fig_ability():
    from ogcore.parameters import Specifications
    spec = Specifications()
    e = np.array(spec.e)[0, :, :]  # S x J at t=0
    lambdas = np.array(spec.lambdas).flatten()
    ages = np.arange(21, 101)
    labels = [
        "0–25th pct", "25–50th", "50–70th", "70–80th",
        "80–90th", "90–99th", "Top 1%",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for j in range(e.shape[1]):
        axes[0].plot(ages, e[:, j], color=OI[j], lw=1.6, label=labels[j])
        axes[1].semilogy(ages, e[:, j], color=OI[j], lw=1.6)
    axes[0].set_xlabel("Age $s$")
    axes[0].set_ylabel(r"Effective labour units $e_{j,s}$")
    axes[0].set_title("Lifetime-ability profiles (levels)", fontsize=9)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2,
                   title=r"Ability type $j$ (share $\lambda_j$)",
                   title_fontsize=7)
    axes[1].set_xlabel("Age $s$")
    axes[1].set_ylabel(r"$e_{j,s}$ (log scale)")
    axes[1].set_title("Lifetime-ability profiles (log scale)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_ability.pdf")
    plt.close(fig)


def fig_ellipse():
    from ogcore.parameters import Specifications
    spec = Specifications()
    b, up, lt, frisch = spec.b_ellipse, spec.upsilon, spec.ltilde, spec.frisch
    n = np.linspace(1e-3, lt - 1e-3, 500)
    # elliptical disutility and its derivative
    g = b * (1.0 - (n / lt) ** up) ** (1.0 / up)
    v = -g
    vp = (b / lt) * (n / lt) ** (up - 1.0) \
        * (1.0 - (n / lt) ** up) ** (1.0 / up - 1.0)
    # CFE calibrated to same Frisch elasticity; scale chosen to match at n=0.4
    n0 = 0.4
    chi_cfe = vp[np.argmin(abs(n - n0))] / (n0 ** (1.0 / frisch))
    v_cfe = -chi_cfe * n ** (1.0 + 1.0 / frisch) / (1.0 + 1.0 / frisch)
    vp_cfe = chi_cfe * n ** (1.0 / frisch)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    axes[0].plot(n, v, color=OI[0], lw=1.8,
                 label=fr"Ellipse ($b={b:.3f}$, $\upsilon={up:.3f}$)")
    axes[0].plot(n, v_cfe, color=OI[1], lw=1.8, ls="--",
                 label=fr"CFE ($\theta={frisch}$, matched at $n=0.4$)")
    axes[0].set_xlabel(r"Labour supply $n/\tilde{l}$")
    axes[0].set_ylabel(r"Disutility $v(n)$")
    axes[0].set_title("Disutility of labour", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(n, vp, color=OI[0], lw=1.8)
    axes[1].plot(n, vp_cfe, color=OI[1], lw=1.8, ls="--")
    axes[1].set_ylim(0, 6)
    axes[1].set_xlabel(r"Labour supply $n/\tilde{l}$")
    axes[1].set_ylabel(r"Marginal disutility $v'(n)$")
    axes[1].set_title(r"Marginal disutility: ellipse $\to\infty$ at both "
                      "bounds", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_ellipse.pdf")
    plt.close(fig)


def gs_etr(inc, phi0, phi1, phi2):
    return phi0 * (1.0 - (inc ** (-phi1) + phi2) ** (-1.0 / phi1) / inc)


def gs_mtr(inc, phi0, phi1, phi2):
    return phi0 * (1.0 - (1.0 + phi2 * inc ** phi1) ** (-1.0 - 1.0 / phi1))


def fig_gs():
    # Illustrative Gouveia-Strauss fit shaped to the UK schedule: rates rise
    # from ~0 through the personal allowance towards an asymptotic maximum.
    # Income measured in GBP thousands; parameters chosen so the ETR rises from
    # a few per cent at low income towards the asymptote phi0 = 0.479.
    phi0, phi1, phi2 = 0.479, 0.85, 0.030  # illustrative values, GS form
    inc = np.linspace(1, 200, 600)          # income in GBP thousands
    etr = gs_etr(inc, phi0, phi1, phi2)
    mtr = gs_mtr(inc, phi0, phi1, phi2)
    pounds = inc

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(pounds, 100 * etr, color=OI[0], lw=1.9, label="ETR (fitted G–S)")
    ax.plot(pounds, 100 * mtr, color=OI[1], lw=1.9, ls="--",
            label="MTR (analytical derivative)")
    ax.axhline(100 * phi0, color="grey", lw=0.8, ls=":")
    ax.text(180, 100 * phi0 + 0.8, r"asymptote $\phi_0$", fontsize=7,
            color="grey", ha="right")
    ax.set_xlabel("Total income (£ thousands)")
    ax.set_ylabel("Tax rate (%)")
    ax.set_title("Gouveia–Strauss schedule: mutually consistent ETR and MTR",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_gs.pdf")
    plt.close(fig)


def dep_tau(x, y, A, B, C, D, mx, mn_x, my, mn_y, sx, sy, sh, phi):
    taux = (mx - mn_x) * (A * x ** 2 + B * x) / (A * x ** 2 + B * x + 1) + mn_x
    tauy = (my - mn_y) * (C * y ** 2 + D * y) / (C * y ** 2 + D * y + 1) + mn_y
    return ((taux + sx) ** phi) * ((tauy + sy) ** (1 - phi)) + sh


def fig_dep():
    # Illustrative DEP surface using parameter magnitudes in the range
    # reported by DeBacker, Evans and Phillips (2019).
    pars = dict(A=0.02, B=0.8, C=0.06, D=0.4, mx=0.45, mn_x=-0.05,
                my=0.42, mn_y=-0.02, sx=0.05, sy=0.02, sh=-0.12, phi=0.75)
    x = np.linspace(0.01, 4, 80)   # labour income, model units
    y = np.linspace(0.01, 4, 80)   # capital income
    X, Y = np.meshgrid(x, y)
    T = dep_tau(X, Y, **pars)

    fig = plt.figure(figsize=(9.2, 3.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X * 50, Y * 50, 100 * T, cmap=cm.viridis, lw=0,
                    antialiased=True, alpha=0.95)
    ax.set_xlabel("Labour income $x$ (£000)", fontsize=7)
    ax.set_ylabel("Capital income $y$ (£000)", fontsize=7)
    ax.set_zlabel("ETR (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("DEP effective-tax-rate surface", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(X * 50, Y * 50, 100 * T, levels=14, cmap=cm.viridis)
    fig.colorbar(cs, ax=ax2, label="ETR (%)")
    ax2.set_xlabel("Labour income $x$ (£000)")
    ax2.set_ylabel("Capital income $y$ (£000)")
    ax2.set_title("Contours: rates rise in both income dimensions",
                  fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_dep.pdf")
    plt.close(fig)


def fig_targets():
    """Calibration targets vs current official UK aggregates (normalised).

    Model values are the deployed OG-UK calibration settings/targets; official
    values are the mid-2026 prints cited in the comparison section. Bars show
    model as a percentage of the official value; annotations flag whether the
    quantity is imposed, targeted, or emergent (so an exact match is anchoring,
    not validation).
    """
    rows = [
        # (label, model value, official value, unit, treatment)
        ("Labour share of income", 60.0, 59.5, "%", "set from factor shares"),
        ("Depreciation / capital stock", 6.5, 6.5, "%", "imposed"),
        ("Potential growth $g_y$", 1.1, 1.1, "%/yr", "imposed"),
        ("Debt-to-GDP (target vs ONS)", 94.4, 95.1, "%", "imposed via $\\alpha_D$"),
        ("Household saving ratio", 8.9, 8.9, "%", "targeted via $\\beta$"),
    ]
    labels = [r[0] for r in rows]
    ratio = [100.0 * r[1] / r[2] for r in rows]
    ypos = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.barh(ypos, ratio, height=0.55, color=OI[0], alpha=0.85)
    ax.axvline(100, color=OI[6], lw=1.0, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(96, 103)
    ax.set_xlabel("Calibration value as % of current official UK value "
                  "(100 = exact match)")
    for y, r, row in zip(ypos, ratio, rows):
        ax.text(r + 0.15, y,
                f"{row[1]:g}{row[3]} vs {row[2]:g}{row[3]}  ({row[4]})",
                va="center", fontsize=7)
    ax.set_title("Calibration targets against official UK aggregates "
                 "(mid-2026 vintages)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_targets.pdf")
    plt.close(fig)


def fig_dep_compare():
    """Deployed-form UK ETR schedule vs the published DEP US fit.

    Left: the DEP ETR and MTRx schedules evaluated with the parameter vectors
    published in the OG-USA documentation for the DeBacker-Evans-Phillips
    specification (baseline current law, age 42, year 2017), at zero capital
    income. Right: the published US ETR curve against the paper's illustrative
    UK Gouveia-Strauss ETR (the deployed functional form).
    """
    # Published DEP parameters, s = 42, t = 2017 (OG-USA docs, DEP-2019).
    dep_etr = dict(A=6.28e-12, B=4.36e-05, C=1.04e-23, D=7.77e-09,
                   mx=0.80, mn_x=-0.14, my=0.80, mn_y=-0.15,
                   sx=0.15, sy=0.16, sh=-0.15, phi=0.84)
    dep_mtrx = dict(A=3.43e-23, B=4.50e-04, C=9.81e-12, D=5.30e-08,
                    mx=0.71, mn_x=-0.17, my=0.80, mn_y=-0.42,
                    sx=0.18, sy=0.43, sh=-0.42, phi=0.96)
    x = np.linspace(1e3, 200e3, 600)          # labour income, dollars

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.plot(x / 1e3, 100 * dep_tau(x, np.zeros_like(x), **dep_etr),
            color=OI[0], lw=1.9, label="ETR (published fit)")
    ax.plot(x / 1e3, 100 * dep_tau(x, np.zeros_like(x), **dep_mtrx),
            color=OI[1], lw=1.9, ls="--", label="MTRx (published fit)")
    ax.set_xlabel("Labour income $x$ (\\$ thousands, $y=0$)")
    ax.set_ylabel("Tax rate (%)")
    ax.set_title("Published DEP fits, US age 42, 2017\n"
                 "(DeBacker–Evans–Phillips parameters)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    ax = axes[1]
    etr_us = dep_tau(x, np.zeros_like(x), **dep_etr)
    ax.plot(x / 1e3, 100 * etr_us, color=OI[0], lw=1.9,
            label="US published DEP fit ($y=0$)")
    phi0, phi1, phi2 = 0.479, 0.85, 0.030     # illustrative UK GS values
    inc = np.linspace(1, 200, 600)            # GBP thousands
    ax.plot(inc, 100 * gs_etr(inc, phi0, phi1, phi2), color=OI[2],
            lw=1.9, ls="--", label="UK illustrative G–S (deployed form)")
    ax.set_xlabel("Total income (local currency, thousands)")
    ax.set_ylabel("ETR (%)")
    ax.set_title("Deployed-form UK schedule vs published US DEP fit",
                 fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_dep_compare.pdf")
    plt.close(fig)


def fig_gdp_range():
    """Published long-run GDP effects across OG-Core-family / OLG simulations.

    All values are as published: OBR WP 22 Table 5.1 (basic rate +/-1pp);
    DeBacker-Evans-Phillips (2019) TCJA steady state (preferred spec, with
    the +0.03 to +0.15 range across tax-function forms); Pomerleau-DeBacker-
    Evans (AEI, 2020) Biden platform long run.
    """
    rows = [
        # (label, value, lo, hi)
        ("OBR UK OLG: basic rate +1pp", -0.1, None, None),
        ("OBR UK OLG: basic rate −1pp", 0.1, None, None),
        ("OG-USA: TCJA (steady state)", 0.07, 0.03, 0.15),
        ("OG-USA: Biden 2020 platform\n(AEI, long run)", -0.2, None, None),
    ]
    ypos = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for y, (lab, v, lo, hi) in zip(ypos, rows):
        ax.barh(y, v, height=0.55, color=OI[0] if v >= 0 else OI[1],
                alpha=0.85)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=OI[6], lw=1.4)
            ax.plot([lo, lo], [y - 0.12, y + 0.12], color=OI[6], lw=1.4)
            ax.plot([hi, hi], [y - 0.12, y + 0.12], color=OI[6], lw=1.4)
    ax.axvline(0, color=OI[6], lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Published long-run GDP effect (%)")
    ax.set_xlim(-0.45, 0.45)
    ax.set_title("Long-run GDP effects of tax reforms in published "
                 "OG-family simulations", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_gdp_range.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_demographics()
    fig_ability()
    fig_ellipse()
    fig_gs()
    fig_dep()
    fig_targets()
    fig_dep_compare()
    fig_gdp_range()
    print("done")
