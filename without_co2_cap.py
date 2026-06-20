"""
Consolidated figure script for the semester-project report sections
"Influence of the bias term" and "Policy evaluation".

Generates exactly the figures referenced in those two sections:

  fig1   bias weight vs. total system cost AND total capacity additions (no policy)
  fig3   capacity additions per technology vs. bias weight (no policy, small multiples)
  fig4   policy effect on the targeted technology (lambda in [0, 1])
  fig5   subsidy-instrument comparison: steering effect + system-cost side effect
  fig6a  clean year-0 profitability index pi per technology (lambda-invariant)
  fig7   profitability index vs. capacity additions (alignment + heat-sector switch)
  fig8a  endogeneity of pi: total pi vs. lambda by decision year (valid range)

Reads the case-study outputs (run_summary.csv + the profitability_*.csv files).
All cost axes are shown in billion EUR (= total_cost / 1e3), assuming the model
cost unit is MEUR (see the report text / the data appendix).

Usage:
    python report_figures.py --data-dir /path/to/run --out-dir /path/to/figures
"""
from __future__ import annotations

import argparse
import functools
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EPS = 1e-6           # |capacity| below this is treated as exact zero (solver noise)
VALID_MAX = 1.0      # principled valid bias range is omega in [0, VALID_MAX)
DEGEN_OMEGA = 3.0    # visible degeneration onset in this case (delayed by capacity limits)
TERMINAL_YEAR = 7    # last rolling-horizon decision step (NPV terminal spike)
EMIS_UNIT = "Mt CO$_2$"   # model emission unit of total_carbon_emissions (relabel if needed)

POLICY_MAP = {
    "none": "none",
    "photovoltaics@DE:remuneration=0.11/electricity": "pv_remun_DE",
    "heat_pump@DE:capex=350": "hp_capex_DE",
    "natural_gas_boiler@DE:variable_opex=0.00710412": "gasboiler_co2_DE",
    "natural_gas_turbine@IT:fixed_opex=70": "gasturbine_capmarket_IT",
}
POLICY_TARGET = {
    "pv_remun_DE": ("photovoltaics", "DE", "DE PV remuneration (0.11/MWh_el)"),
    "hp_capex_DE": ("heat_pump", "DE", "DE heat-pump CAPEX subsidy (-350 MEUR/GW)"),
    "gasboiler_co2_DE": ("natural_gas_boiler", "DE", "DE gas-boiler CO2 relief (var. OPEX, 0.007 MEUR/GWh_th)"),
    "gasturbine_capmarket_IT": ("natural_gas_turbine", "IT", "IT gas-turbine capacity market (fixed OPEX, 70 MEUR/GW)"),
}
TECH_ORDER = [
    "photovoltaics", "wind_onshore", "wind_offshore",
    "reservoir_hydro", "run-of-river_hydro", "heat_pump", "battery",
    "natural_gas_turbine", "natural_gas_boiler", "power_line", "natural_gas_pipeline",
]
TECH_COLOR = {
    "photovoltaics": "#F4B400", "wind_onshore": "#4C9BE8", "wind_offshore": "#1F5FA6",
    "reservoir_hydro": "#1B9E9E", "run-of-river_hydro": "#7FD1C4",
    "heat_pump": "#2CA02C", "battery": "#7B5EA7",
    "natural_gas_turbine": "#E8825A", "natural_gas_boiler": "#B23B3B",
    "power_line": "#9AA0A6", "natural_gas_pipeline": "#5F6368",
}
TECH_LABEL = {
    "photovoltaics": "Photovoltaics", "wind_onshore": "Wind onshore", "wind_offshore": "Wind offshore",
    "reservoir_hydro": "Reservoir hydro", "run-of-river_hydro": "Run-of-river hydro",
    "heat_pump": "Heat pump", "battery": "Battery",
    "natural_gas_turbine": "Gas turbine", "natural_gas_boiler": "Gas boiler",
    "power_line": "Power line", "natural_gas_pipeline": "Gas pipeline",
}
TECH_SECTOR = {
    "photovoltaics": "electricity", "wind_onshore": "electricity", "wind_offshore": "electricity",
    "reservoir_hydro": "electricity", "run-of-river_hydro": "electricity",
    "natural_gas_turbine": "electricity", "heat_pump": "heat", "natural_gas_boiler": "heat",
}

C_PF = "#2B3A55"       # perfect foresight
C_NOBIAS = "#C46A3F"   # rolling horizon, no bias (lambda = 0)
C_BIAS = "#1A1A1A"     # biased sweep curve
C_HEALTHY = "#E8F1E8"
C_DEGEN = "#FBEAEA"
C_POS = "#1B9E77"      # profitable
C_NEG = "#B23B3B"      # loss-making
POLICY_COLOR = {
    "pv_remun_DE": "#F4B400", "hp_capex_DE": "#2CA02C",
    "gasboiler_co2_DE": "#B23B3B", "gasturbine_capmarket_IT": "#E8825A",
}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

# Larger fonts for the figures that go into the printed report (fig1, fig3, fig4,
# fig7, figE1, figE2). Applied per-figure via @boosted_fonts so the remaining
# figures keep the compact defaults above.
FONT_BOOST = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
}


def boosted_fonts(fn):
    """Run a figure function with the enlarged report font sizes."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(FONT_BOOST):
            return fn(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------- #
# Loading & shaping
# --------------------------------------------------------------------------- #
def bias_of(opt: str) -> float:
    m = re.search(r"bias_weight=([0-9.]+)", str(opt))
    return float(m.group(1)) if m else 0.0


def load_summary(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "run_summary.csv")
    df["bias"] = df["optimization"].map(bias_of)
    df["policy"] = df["subsidies"].map(POLICY_MAP)
    if df["policy"].isna().any():
        raise ValueError(f"Unmapped subsidy strings: {df.loc[df.policy.isna(),'subsidies'].unique()}")
    # NOTE: the original script dropped run_summary rows older than the earliest
    # profitability-file timestamp, to guard against stale files from a *different*
    # run. In this dataset the profitability files only begin part-way through the
    # *same* run (the first ones are written at bias 0.3), so that filter would
    # discard the perfect-foresight reference, the no-bias (omega=0) run and the
    # low-bias runs. All files here come from one coherent batch, so the filter is
    # disabled and we only de-duplicate.
    if "run_timestamp" in df.columns:
        df = (df.sort_values("run_timestamp")
                .drop_duplicates(subset=["foresight_mode", "bias", "policy"], keep="last"))
    return df.reset_index(drop=True)


def cap_columns(df):
    return [c for c in df.columns if c.startswith("cap|")]


def additions_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        for c in cap_columns(df):
            _, tech, node, yr = c.split("|")
            v = row[c]
            v = 0.0 if abs(v) < EPS else float(v)
            rows.append((idx, row["foresight_mode"], row["bias"], row["policy"],
                         tech, node, int(yr), v))
    return pd.DataFrame(rows, columns=["run", "fm", "bias", "policy", "tech", "node", "yr", "add"])


def tech_node_totals(L: pd.DataFrame) -> pd.DataFrame:
    return L.groupby(["run", "fm", "bias", "policy", "tech", "node"],
                     as_index=False)["add"].sum()


def _parse_profitability_file(fn: str):
    base = Path(fn).name.split("profitability_")[1].replace(".csv", "")
    if base.startswith("perfect_foresight"):
        return "perfect_foresight", np.nan, "none"
    # no-bias (omega=0) runs: "no_bias" or "no_bias_<policy>". The original regex
    # only matched the bare "no_bias" file; the complete dataset also contains the
    # omega=0 *policy* runs (e.g. "no_bias_pv_remun_DE"), which must be tagged with
    # their actual policy (not "none") so they are not miscounted as no-policy.
    m = re.match(r"rolling_horizon_no_bias(?:_(.+))?$", base)
    if m:
        return "rolling_horizon", 0.0, (m.group(1) if m.group(1) else "none")
    # biased runs: "bias_<weight>" or "bias_<weight>_<policy>"
    m = re.match(r"rolling_horizon_bias_(.+)$", base)
    rest = m.group(1)  # e.g. "0.05", "5_0", or "0.05_pv_remun_DE"
    vm = re.match(r"(\d+(?:[._]\d+)?)(?:_(.+))?$", rest)
    lam = float(vm.group(1).replace("_", "."))   # "5_0" -> 5.0, "0.05" -> 0.05
    pol = vm.group(2) if vm.group(2) else "none"
    return "rolling_horizon", lam, pol


def load_profitability(data_dir: Path):
    rows = []
    for f in glob.glob(str(data_dir / "*profitability*.csv")):
        fm, lam, pol = _parse_profitability_file(f)
        d = pd.read_csv(f)
        d["fm"], d["lam"], d["policy"] = fm, lam, pol
        rows.append(d)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def clean_year0(A: pd.DataFrame) -> pd.DataFrame:
    """First-period (decision_year == 0) profitability of the no-policy rolling
    horizon. The first rolling step carries no bias, so this is omega-invariant.

    The net-zero run included an omega=0 profitability file; this no-CO2-cap run
    does not (its profitability files start at bias 0.3/0.4). Because year 0 is
    identical for every omega, we take it from the *lowest available* weight
    instead of hard-coding lam == 0. (Verified: the per-technology year-0 values
    are bit-identical across all available weights.)
    """
    base = A[(A.fm == "rolling_horizon") & (A.policy == "none") & (A.decision_year == 0)]
    if len(base) == 0:
        return base
    lam0 = sorted(base["lam"].unique())[0]
    return base[base["lam"] == lam0]


def additions_per_tech(row, cap_cols) -> dict:
    d = {}
    for c in cap_cols:
        _, t, _, _ = c.split("|")
        v = row[c]
        d[t] = d.get(t, 0.0) + (0.0 if abs(v) < EPS else float(v))
    return d


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def ordinal_axis(ax, lambdas):
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:g}" for l in lambdas], rotation=90)
    ax.set_xlim(-0.5, len(lambdas) - 0.5)


def shade_regimes(ax, lambdas):
    # green: valid bias range omega in [0, VALID_MAX)
    gi = next((i for i, l in enumerate(lambdas) if l >= VALID_MAX), len(lambdas))
    ax.axvspan(-0.5, gi - 0.5, color=C_HEALTHY, zorder=0)
    # red: visible degeneration (omega >= DEGEN_OMEGA); the [VALID_MAX, DEGEN_OMEGA)
    # transition stays white because capacity limits delay the breakdown
    di = next((i for i, l in enumerate(lambdas) if l >= DEGEN_OMEGA), len(lambdas))
    if di < len(lambdas):
        ax.axvspan(di - 0.5, len(lambdas) - 0.5, color=C_DEGEN, zorder=0)
    return gi, di


def savefig(fig, out_dir: Path, name: str):
    # PNG for quick viewing, PDF and SVG as vector formats for LaTeX
    # (the report uses \includesvg{./Graphs/...}; PDF works with \includegraphics)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf / .svg")


# --------------------------------------------------------------------------- #
# Figure 1 -- bias weight vs total cost AND total additions (no policy)
# --------------------------------------------------------------------------- #
@boosted_fonts
def fig1_bias_overview(df, TN, out_dir):
    pf_rows = df[df.foresight_mode == "perfect_foresight"]
    has_pf = len(pf_rows) > 0
    pf = pf_rows.iloc[0] if has_pf else None
    none = df[df.policy == "none"].copy()
    sweep = none[none.foresight_mode == "rolling_horizon"].sort_values("bias")
    lambdas = sweep["bias"].tolist()
    x = range(len(lambdas))

    cost = sweep["total_cost"].values / 1e3
    pf_cost = pf["total_cost"] / 1e3 if has_pf else None
    nobias_cost = sweep[sweep.bias == 0]["total_cost"].iloc[0] / 1e3

    add_by_run = (TN[TN.policy == "none"].groupby(["run", "bias", "fm"])["add"].sum().reset_index())
    sweep_add = add_by_run[add_by_run.fm == "rolling_horizon"].sort_values("bias")["add"].values
    pf_add = (add_by_run[add_by_run.fm == "perfect_foresight"]["add"].iloc[0] if has_pf else None)
    nobias_add = add_by_run[(add_by_run.fm == "rolling_horizon") & (add_by_run.bias == 0)]["add"].iloc[0]

    lam_star = lambdas[int(np.argmin(cost))]
    gap_closed = ((nobias_cost - cost.min()) / (nobias_cost - pf_cost) * 100) if has_pf else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    shade_regimes(ax, lambdas)
    ax.plot(x, cost, "-o", color=C_BIAS, lw=1.8, ms=4, zorder=3, label="Rolling horizon (biased)")
    if has_pf:
        ax.axhline(pf_cost, color=C_PF, ls=":", lw=2, label="Perfect foresight")
    ax.axhline(nobias_cost, color=C_NOBIAS, ls="--", lw=2, label="Rolling horizon, no bias ($\\omega=0$)")
    argmin = int(np.argmin(cost))
    if has_pf and argmin > 0 and gap_closed > 5:
        # the bias meaningfully reduces cost: point to the optimum
        ax.annotate(f"cost-optimal $\\omega\\approx{lam_star:g}$\n"
                    f"($\\approx${gap_closed:.0f}% of the\nmyopia gap recovered)",
                    xy=(argmin, cost.min()), xytext=(max(argmin - 5.0, 0.3), cost.min() + 85),
                    fontsize=11, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#1B7A1B", lw=1.2, connectionstyle="arc3,rad=-0.2"))
    ordinal_axis(ax, lambdas)
    ax.set_xlabel("Bias weight  $\\omega$")
    ax.set_ylabel("Total system cost  [$10^3$ MEUR]")
    ax.set_title("(a)  Bias weight vs. total system cost  (no policy)")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)

    ax = axes[1]
    shade_regimes(ax, lambdas)
    ax.plot(x, sweep_add, "-o", color=C_BIAS, lw=1.8, ms=4, zorder=3, label="Rolling horizon (biased)")
    if has_pf:
        ax.axhline(pf_add, color=C_PF, ls=":", lw=2, label="Perfect foresight")
    ax.axhline(nobias_add, color=C_NOBIAS, ls="--", lw=2, label="Rolling horizon, no bias ($\\omega=0$)")
    ordinal_axis(ax, lambdas)
    ax.set_xlabel("Bias weight  $\\omega$")
    ax.set_ylabel("Total capacity additions  [GW + GWh]")
    ax.set_title("(b)  Bias weight vs. total capacity additions  (no policy)")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)

    fig.suptitle("Influence of the profitability-bias weight (rolling horizon, no policy)", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig1_bias_cost_and_additions")


# --------------------------------------------------------------------------- #
# Figure 3 -- per-technology additions vs bias weight (small multiples)
# --------------------------------------------------------------------------- #
@boosted_fonts
def fig3_tech_vs_bias(TN, out_dir):
    none = TN[TN.policy == "none"]
    sweep = none[none.fm == "rolling_horizon"]
    lambdas = sorted(sweep["bias"].unique())
    pf = none[none.fm == "perfect_foresight"]
    techs = [t for t in TECH_ORDER if t in
             ["photovoltaics", "wind_onshore", "wind_offshore", "reservoir_hydro",
              "run-of-river_hydro", "heat_pump", "battery", "natural_gas_turbine", "natural_gas_boiler"]]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    has_pf = len(pf) > 0
    for ax, tech in zip(axes.ravel(), techs):
        s = sweep[sweep.tech == tech].groupby("bias")["add"].sum().reindex(lambdas).values
        x = range(len(lambdas))
        shade_regimes(ax, lambdas)
        ax.plot(x, s, "-o", color=TECH_COLOR[tech], lw=1.8, ms=3.5, zorder=3)
        if has_pf:
            pf_val = pf[pf.tech == tech]["add"].sum()
            ax.axhline(pf_val, color=C_PF, ls=":", lw=1.6, zorder=2)
        ax.scatter([0], [s[0]], s=45, facecolors="white", edgecolors=C_NOBIAS, lw=1.6, zorder=4)
        ax.set_title(TECH_LABEL[tech], fontsize=13)
        ordinal_axis(ax, lambdas)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Additions [GW/GWh]")
    for ax in axes[-1, :]:
        ax.set_xlabel("Bias weight $\\omega$")
    legend = [
        Line2D([0], [0], color=C_PF, ls=":", lw=1.8, label="Perfect foresight"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor=C_NOBIAS, markersize=8, lw=0, label="No bias ($\\omega=0$)"),
        Patch(facecolor=C_HEALTHY, label="valid range $\\omega\\in[0,1)$"),
        Patch(facecolor=C_DEGEN, label="degenerate (from $\\omega\\approx3$)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Capacity additions per technology vs. bias weight (rolling horizon, no policy)", fontsize=15, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    savefig(fig, out_dir, "fig3_tech_additions_vs_bias")


# --------------------------------------------------------------------------- #
# Figure 4 / 5 -- policy effects (lambda in [0, 1])
# --------------------------------------------------------------------------- #
def _target_series(df, policy, tech, node, lambdas):
    rh = df[df.foresight_mode == "rolling_horizon"]
    prefix = f"cap|{tech}|{node}|"
    cols = [c for c in df.columns if c.startswith(prefix)]
    base, treat = [], []
    for b in lambdas:
        rb = rh[(rh.bias == b) & (rh.policy == "none")].iloc[0]
        rt = rh[(rh.bias == b) & (rh.policy == policy)].iloc[0]
        base.append(sum(0.0 if abs(rb[c]) < EPS else rb[c] for c in cols))
        treat.append(sum(0.0 if abs(rt[c]) < EPS else rt[c] for c in cols))
    return np.array(base), np.array(treat)


@boosted_fonts
def fig4_policy_targets(df, out_dir):
    rh = df[df.foresight_mode == "rolling_horizon"]
    lambdas = sorted([b for b in rh["bias"].unique() if b <= 1.0])
    x = range(len(lambdas))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    for ax, (pol, (tech, node, name)) in zip(axes.ravel(), POLICY_TARGET.items()):
        base, treat = _target_series(df, pol, tech, node, lambdas)
        ax.plot(x, base, "-o", color="#888", lw=1.8, ms=4, label="No policy (baseline)")
        ax.plot(x, treat, "-o", color=POLICY_COLOR[pol], lw=2.0, ms=4, label="With policy")
        ax.fill_between(x, base, treat, color=POLICY_COLOR[pol], alpha=0.15)
        ax.set_xticks(list(x)); ax.set_xticklabels([f"{l:g}" for l in lambdas], fontsize=11)
        ax.set_xlabel("Bias weight $\\omega$")
        ax.set_ylabel(f"{TECH_LABEL[tech]} @ {node} additions [GW]")
        ax.set_title(name, fontsize=13)
        ax.legend(loc="upper left", fontsize=12)
    fig.suptitle("Policy effect on the targeted technology (rolling horizon, $\\omega\\in[0,1]$)\n"
                 "At $\\omega=0$ every instrument is inert — the subsidies act only through the bias channel.",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig4_policy_targeted_effects")


def fig5_policy_summary(df, out_dir):
    rh = df[df.foresight_mode == "rolling_horizon"]
    lambdas = sorted([b for b in rh["bias"].unique() if b <= 1.0])
    x = range(len(lambdas))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for pol, (tech, node, name) in POLICY_TARGET.items():
        base, treat = _target_series(df, pol, tech, node, lambdas)
        ax.plot(x, treat - base, "-o", color=POLICY_COLOR[pol], lw=2, ms=4, label=name.split(" (")[0])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l:g}" for l in lambdas], fontsize=8)
    ax.set_xlabel("Bias weight $\\omega$")
    ax.set_ylabel("Δ targeted additions  (with policy − baseline)  [GW]")
    ax.set_title("(a)  Steering effect on the targeted technology")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for pol, (tech, node, name) in POLICY_TARGET.items():
        d = []
        for b in lambdas:
            cb = rh[(rh.bias == b) & (rh.policy == "none")]["total_cost"].iloc[0]
            ct = rh[(rh.bias == b) & (rh.policy == pol)]["total_cost"].iloc[0]
            d.append((ct - cb) / 1e3)
        ax.plot(x, d, "-o", color=POLICY_COLOR[pol], lw=2, ms=4, label=name.split(" (")[0])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l:g}" for l in lambdas], fontsize=8)
    ax.set_xlabel("Bias weight $\\omega$")
    ax.set_ylabel("Δ total system cost  (with policy − baseline)  [$10^3$ MEUR]")
    ax.set_title("(b)  Side effect on total system cost")
    ax.legend(fontsize=8, loc="lower left")
    # Data-driven annotation: classify instruments by the SIGN of their cost
    # effect at the top of the valid range. (The original text was hard-coded for
    # the net-zero run, where the pattern is the opposite of the no-CO2-cap run.)
    b_top = max(lambdas)
    raisers, lowerers = [], []
    for pol, (tech, node, name) in POLICY_TARGET.items():
        cb = rh[(rh.bias == b_top) & (rh.policy == "none")]["total_cost"].iloc[0]
        ct = rh[(rh.bias == b_top) & (rh.policy == pol)]["total_cost"].iloc[0]
        short = name.split(" (")[0]
        (raisers if ct - cb > 0 else lowerers).append(short)
    note_lines = []
    if raisers:
        note_lines.append("raise cost: " + ", ".join(raisers))
    if lowerers:
        note_lines.append("lower cost: " + ", ".join(lowerers))
    ax.text(0.97, 0.95, "\n".join(note_lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444")
    fig.suptitle("Subsidy-instrument comparison (rolling horizon, $\\omega\\in[0,1]$)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig5_policy_summary")


# --------------------------------------------------------------------------- #
# Figure 6a -- clean year-0 profitability index per technology (standalone)
# --------------------------------------------------------------------------- #
def fig6a_profitability_index(A, out_dir):
    y0 = clean_year0(A)
    pi = y0.groupby("set_conversion_technologies").profitability.sum().sort_values()
    techs = pi.index.tolist()
    colors = [C_POS if v > 0 else C_NEG for v in pi.values]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.barh([TECH_LABEL[t] for t in techs], pi.values, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="#333", lw=1)
    for b, v in zip(bars, pi.values):
        ax.text(v + (250 if v >= 0 else -250), b.get_y() + b.get_height() / 2,
                f"{v:,.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlabel("Profitability index  $\\Pi$  [MEUR/GW]")
    ax.set_title("Clean year-0 profitability index per technology ($\\omega$-invariant)")
    ax.margins(x=0.18)
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", color=C_POS, label="profitable ($\\Pi>0$)"),
                       Line2D([0], [0], marker="s", ls="", color=C_NEG, label="loss-making ($\\Pi<0$)")],
              loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, out_dir, "fig6a_profitability_index")


# --------------------------------------------------------------------------- #
# Figure 7 -- profitability vs capacity additions (alignment)
# --------------------------------------------------------------------------- #
@boosted_fonts
def fig7_profitability_vs_additions(A, rs, out_dir):
    from scipy.stats import spearmanr
    y0 = clean_year0(A)
    pi0 = y0.groupby("set_conversion_technologies").profitability.sum()
    cap_cols = [c for c in rs.columns if c.startswith("cap|")]
    none = rs[(rs.foresight_mode == "rolling_horizon") & (rs.subsidies == "none")]
    biases = sorted(none.bias.unique())
    # capacity-additions response is measured against the largest valid omega
    # available (<= 1); falls back to the largest bias present
    valid = [b for b in biases if 0 < b < VALID_MAX]
    omega_hi = max(valid) if valid else max(biases)
    a0 = additions_per_tech(none[none.bias == 0].iloc[0], cap_cols)
    a2 = additions_per_tech(none[none.bias == omega_hi].iloc[0], cap_cols)
    ceiling = {"reservoir_hydro", "run-of-river_hydro"}
    conv = [t for t in pi0.index if t in TECH_SECTOR]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax = axes[0]
    free = [t for t in conv if t not in ceiling]
    rho, _ = spearmanr([pi0[t] for t in free], [a2.get(t, 0) - a0.get(t, 0) for t in free])
    LBL = {"photovoltaics": (8, -5, "left"), "natural_gas_turbine": (10, 8, "left"),
           "reservoir_hydro": (2, 12, "center"), "run-of-river_hydro": (-2, -18, "center"),
           "heat_pump": (8, -17, "left"), "wind_onshore": (8, 10, "left"),
           "wind_offshore": (8, -17, "left"), "natural_gas_boiler": (8, 7, "left")}
    for t in conv:
        d = a2.get(t, 0) - a0.get(t, 0)
        is_c = t in ceiling
        ax.scatter(pi0[t], d, s=160, color=TECH_COLOR[t], marker="o" if not is_c else "D",
                   edgecolor="#333", linewidth=1.2, zorder=3,
                   facecolor=TECH_COLOR[t] if not is_c else "white")
        if is_c:
            ax.scatter(pi0[t], d, s=160, marker="D", facecolor="none", edgecolor=TECH_COLOR[t], linewidth=2, zorder=4)
        ox, oy, ha = LBL[t]
        ax.annotate(TECH_LABEL[t], (pi0[t], d), textcoords="offset points", xytext=(ox, oy), fontsize=11, ha=ha)
    ax.axhline(0, color="#999", lw=0.8, ls="--"); ax.axvline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("Clean year-0 profitability $\\Pi$  [MEUR/GW]")
    ax.set_ylabel(f"$\\Delta$ capacity additions,  $\\omega{{:}}\\,0\\!\\to\\!{omega_hi:g}$  [GW]")
    ax.set_title(f"(a)  Do additions follow $\\Pi$?  $\\rho_{{\\mathrm{{Spearman}}}}={rho:.2f}$ (non-ceiling)")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="#888", mec="#333", label="can expand"),
                       Line2D([0], [0], marker="D", ls="", mfc="none", mec="#333", label="at potential ceiling")],
              loc="upper left", fontsize=12, framealpha=0.9)

    ax = axes[1]
    lambdas = sorted(none.bias.unique())
    lam_valid = [l for l in lambdas if l < VALID_MAX]
    hp = [additions_per_tech(none[none.bias == l].iloc[0], cap_cols)["heat_pump"] for l in lam_valid]
    gb = [additions_per_tech(none[none.bias == l].iloc[0], cap_cols)["natural_gas_boiler"] for l in lam_valid]
    xx = range(len(lam_valid))
    ax.plot(xx, hp, "-o", color=TECH_COLOR["heat_pump"], lw=2, ms=4, label="Heat pump (additions)")
    ax.plot(xx, gb, "-o", color=TECH_COLOR["natural_gas_boiler"], lw=2, ms=4, label="Gas boiler (additions)")
    ax.set_xticks(list(xx)); ax.set_xticklabels([f"{l:g}" for l in lam_valid], rotation=90, fontsize=11)
    ax.set_xlabel("Bias weight  $\\omega$  (valid range)")
    ax.set_ylabel("Capacity additions  [GW]")
    ax.set_title("(b)  Heat sector: heat pump vs. gas boiler additions")
    ax.legend(fontsize=12, loc="center left")
    fig.suptitle("Profitability indices vs. capacity additions  (rolling horizon, no policy)", fontsize=15, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_dir, "fig7_profitability_vs_additions")


# --------------------------------------------------------------------------- #
# Figure 8a -- endogeneity of pi: total pi vs lambda by decision year (standalone)
# --------------------------------------------------------------------------- #
def fig8a_profitability_contamination(A, out_dir):
    none = A[(A.fm == "rolling_horizon") & (A.policy == "none")]
    lambdas = sorted(none.lam.unique())
    lam_valid = [l for l in lambdas if l < VALID_MAX]
    piv = none[none.lam < VALID_MAX].pivot_table(index="lam", columns="decision_year",
                                                 values="profitability", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    gi = next((i for i, l in enumerate(lam_valid) if l >= VALID_MAX), len(lam_valid))
    ax.axvspan(-0.5, gi - 0.5, color=C_HEALTHY, zorder=0, label="_nolegend_")
    cmap = plt.cm.viridis(np.linspace(0, 0.92, TERMINAL_YEAR))
    for dy in range(TERMINAL_YEAR):  # 0..6
        ax.plot(range(len(lam_valid)), piv[dy].values / 1e3, "-o", ms=4, lw=1.8,
                color=cmap[dy], label=f"year {dy}")
    ax.axhline(piv[0].iloc[0] / 1e3, color="#222", ls=":", lw=1)
    ax.set_xticks(range(len(lam_valid)))
    ax.set_xticklabels([f"{l:g}" for l in lam_valid], rotation=90, fontsize=8)
    ax.set_xlim(-0.5, len(lam_valid) - 0.5)
    ax.set_xlabel("Bias weight  $\\omega$   (green: valid range $[0,1)$)")
    ax.set_ylabel("Total $\\Pi$ over technologies & nodes  [$10^3$ MEUR/GW]")
    ax.set_title("Endogeneity of $\\Pi$: $\\Pi$ drifts upward with $\\omega$ (years $\\geq 1$)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9, title="decision year")
    fig.tight_layout()
    savefig(fig, out_dir, "fig8a_profitability_contamination")


# --------------------------------------------------------------------------- #
# Figure E1 -- CO2 emissions vs bias weight (no policy) + cost-emission trade-off
# --------------------------------------------------------------------------- #
@boosted_fonts
def figE1_emissions_bias(df, out_dir):
    E = "total_carbon_emissions"
    none = df[(df.policy == "none") & (df.foresight_mode == "rolling_horizon")].sort_values("bias")
    lambdas = none["bias"].tolist()
    emis = none[E].values
    cost = none["total_cost"].values / 1e3
    e0 = none[none.bias == 0][E].iloc[0]
    x = range(len(lambdas))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    # (a) emissions vs omega
    ax = axes[0]
    shade_regimes(ax, lambdas)
    ax.plot(x, emis, "-o", color="#3A7D44", lw=1.8, ms=4, zorder=3)
    ax.axhline(e0, color=C_NOBIAS, ls="--", lw=2, label="no bias ($\\omega=0$)")
    for w in (1.0, lambdas[-1]):
        if w in lambdas:
            i = lambdas.index(w)
            ax.annotate(f"{100*(emis[i]-e0)/e0:+.0f}%", (i, emis[i]),
                        textcoords="offset points", xytext=(4, 7), fontsize=11, color="#2E6336")
    ordinal_axis(ax, lambdas)
    ax.set_xlabel("Bias weight  $\\omega$")
    ax.set_ylabel(f"Total CO$_2$ emissions  [{EMIS_UNIT}]")
    ax.set_title("(a)  Emissions vs. bias weight  (no policy)")
    ax.legend(loc="lower left", fontsize=12)

    # (b) cost-emission trade-off, parametric in omega
    ax = axes[1]
    cols = [C_POS if l < VALID_MAX else ("#999999" if l < DEGEN_OMEGA else C_NEG) for l in lambdas]
    ax.plot(cost, emis, "-", color="#ccc", lw=1, zorder=1)
    ax.scatter(cost, emis, c=cols, s=60, edgecolor="#333", lw=0.8, zorder=3)
    for l, cc, ee in zip(lambdas, cost, emis):
        if l in (0, 0.5, 1.0, 2.0, 5.0, 100.0):
            ax.annotate(f"$\\omega$={l:g}", (cc, ee), textcoords="offset points",
                        xytext=(6, 3), fontsize=10, color="#444")
    ax.set_xlabel("Total system cost  [$10^3$ MEUR]")
    ax.set_ylabel(f"Total CO$_2$ emissions  [{EMIS_UNIT}]")
    ax.set_title("(b)  Cost\u2013emission trade-off")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=C_POS, mec="#333", label="valid $\\omega<1$"),
                       Line2D([0], [0], marker="o", ls="", mfc="#999", mec="#333", label="transition $1\\leq\\omega<3$"),
                       Line2D([0], [0], marker="o", ls="", mfc=C_NEG, mec="#333", label="degenerate $\\omega\\geq3$")],
              loc="upper right", fontsize=12)
    fig.suptitle("CO$_2$ emissions under the profitability bias (no CO$_2$ cap)", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "figE1_emissions_bias")


# --------------------------------------------------------------------------- #
# Figure E2 -- CO2 emissions by policy instrument
# --------------------------------------------------------------------------- #
@boosted_fonts
def figE2_emissions_policy(df, out_dir):
    E = "total_carbon_emissions"
    rh = df[df.foresight_mode == "rolling_horizon"]
    lambdas = sorted([b for b in rh.bias.unique() if b <= 1.0])
    x = range(len(lambdas))

    def emis(pol, b):
        return rh[(rh.policy == pol) & (rh.bias == b)][E].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    ax = axes[0]
    ax.plot(x, [emis("none", b) for b in lambdas], "-o", color="#333", lw=2, ms=4, label="no policy")
    for pol, (tech, node, name) in POLICY_TARGET.items():
        ax.plot(x, [emis(pol, b) for b in lambdas], "-o", color=POLICY_COLOR[pol], lw=1.8, ms=3.5,
                label=name.split(" (")[0])
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l:g}" for l in lambdas], fontsize=11)
    ax.set_xlabel("Bias weight $\\omega$")
    ax.set_ylabel(f"Total CO$_2$ emissions  [{EMIS_UNIT}]")
    ax.set_title("(a)  Emissions by policy  ($\\omega\\in[0,1]$)")
    ax.legend(fontsize=12, loc="lower left")

    ax = axes[1]
    for pol, (tech, node, name) in POLICY_TARGET.items():
        d = [emis(pol, b) - emis("none", b) for b in lambdas]
        ax.plot(x, d, "-o", color=POLICY_COLOR[pol], lw=2, ms=4, label=name.split(" (")[0])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l:g}" for l in lambdas], fontsize=11)
    ax.set_xlabel("Bias weight $\\omega$")
    ax.set_ylabel(f"$\\Delta$ CO$_2$ emissions vs. no policy  [{EMIS_UNIT}]")
    ax.set_title("(b)  Policy emission effect")
    ax.legend(fontsize=12, loc="lower left")
    fig.suptitle("Effect of the policy instruments on CO$_2$ emissions (no CO$_2$ cap)", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "figE2_emissions_policy")


# --------------------------------------------------------------------------- #
def main():
    # Default to the bundled case-study run so the script "just works" when
    # launched directly from the IDE (no command-line arguments). Paths are
    # resolved relative to this file, not the current working directory, so the
    # run folder is found regardless of where Python is started from.
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir / "data" / "until2050_no_carbon_cap"
    default_out = default_data / "figures"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--out-dir", type=Path, default=default_out)
    args = ap.parse_args()

    if not (args.data_dir / "run_summary.csv").exists():
        raise SystemExit(
            f"run_summary.csv not found in {args.data_dir}\n"
            f"Pass the run folder explicitly, e.g.:\n"
            f"    python {Path(__file__).name} --data-dir /path/to/run --out-dir /path/to/figures"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {args.data_dir}")
    df = load_summary(args.data_dir)
    L = additions_long(df)
    TN = tech_node_totals(L)
    A = load_profitability(args.data_dir)

    print("Generating report figures:")
    fig1_bias_overview(df, TN, args.out_dir)
    fig3_tech_vs_bias(TN, args.out_dir)
    if (df["policy"] != "none").any():
        fig4_policy_targets(df, args.out_dir)
        fig5_policy_summary(df, args.out_dir)
    else:
        print("  (no policy runs present -> skipping fig4 / fig5)")
    if A is not None and len(A) > 0:
        fig6a_profitability_index(A, args.out_dir)
        fig7_profitability_vs_additions(A, df, args.out_dir)
        fig8a_profitability_contamination(A, args.out_dir)
    else:
        print("  (no profitability files present -> skipping fig6a / fig7 / fig8a)")
    if "total_carbon_emissions" in df.columns:
        figE1_emissions_bias(df, args.out_dir)
        figE2_emissions_policy(df, args.out_dir)
    else:
        print("  (no total_carbon_emissions column -> skipping figE1 / figE2)")
    print(f"Done -> {args.out_dir}")


if __name__ == "__main__":
    main()
    