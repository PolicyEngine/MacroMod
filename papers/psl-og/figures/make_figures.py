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
    phi0, phi1, phi2 = 0.479, 0.022, 0.212  # illustrative values, GS form
    inc = np.linspace(1e3, 200e3, 600) / 50e3  # income in model-ish units
    etr = gs_etr(inc, phi0, phi1, phi2)
    mtr = gs_mtr(inc, phi0, phi1, phi2)
    pounds = inc * 50e3 / 1e3

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


if __name__ == "__main__":
    fig_demographics()
    fig_ability()
    fig_ellipse()
    fig_gs()
    fig_dep()
    print("done")
