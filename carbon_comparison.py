#!/usr/bin/env python3
"""Compare total CO2 emissions across the bias-weight sweep for two scenarios:
a 3 Gt cumulative CO2 budget vs. an unconstrained (no CO2 limit) run.

Only no-policy runs are considered (subsidies == 'none'). Emissions are the
cumulative carbon emissions over the modelling horizon (2022-2050).

The figure styling matches report_figures / without_co2_cap.py: a categorical
bias-weight axis with the green "valid range" and red "degenerate" shading, the
shared colour palette, and PNG/PDF/SVG output.

Usage (defaults to the bundled case-study runs when launched without arguments):
    python carbon_comparison.py \
        --budget  data/carbon-emission-comparison/with_cap \
        --nolimit data/carbon-emission-comparison/without_cap \
        --outdir  data/carbon-emission-comparison/figures
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Configuration (kept consistent with without_co2_cap.py)
# --------------------------------------------------------------------------- #
VALID_MAX = 1.0      # principled valid bias range is omega in [0, VALID_MAX)
DEGEN_OMEGA = 3.0    # visible degeneration onset (delayed by capacity limits)
MT_TO_GT = 1.0 / 1000.0

# shared palette / shading
C_PF = "#2B3A55"        # perfect foresight reference
C_BIAS = "#1A1A1A"      # biased sweep curve
C_HEALTHY = "#E8F1E8"   # valid-range shading
C_DEGEN = "#FBEAEA"     # degenerate-range shading

C_BUDGET = "#1f5fa6"    # 3 Gt budget sweep
C_NOLIM = "#c0392b"     # no CO2 limit sweep
C_CAP = "#2c7d4f"       # 3 Gt budget cap reference

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- #
# Loading & shaping
# --------------------------------------------------------------------------- #
def parse_omega(row: pd.Series) -> float | None:
    """Map a run to its bias weight omega. Perfect-foresight -> None.
    Rolling-horizon total_cost -> omega = 0 (no bias)."""
    if row["optimization"] == "total_cost":
        return 0.0 if row["foresight_mode"] == "rolling_horizon" else None
    m = re.search(r"bias_weight=([0-9.]+)", str(row["optimization"]))
    return float(m.group(1)) if m else None


def load(path: str | Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["subsidies"] == "none"].copy()          # exclude policy instruments
    df["omega"] = df.apply(parse_omega, axis=1)
    df["scenario"] = label
    df["emissions_Gt"] = df["total_carbon_emissions"] * MT_TO_GT
    df["is_pf"] = df["foresight_mode"] == "perfect_foresight"
    return df


def split(df: pd.DataFrame):
    pf = df[df["is_pf"]]
    sweep = df[~df["is_pf"]].sort_values("omega")
    pf_val = float(pf["emissions_Gt"].iloc[0]) if len(pf) else np.nan
    return sweep, pf_val


def resolve_summary(p: Path) -> Path:
    """Accept either a run folder or a direct run_summary.csv path."""
    p = Path(p)
    return p if p.is_file() else p / "run_summary.csv"


# --------------------------------------------------------------------------- #
# Shared helpers (mirrors without_co2_cap.py)
# --------------------------------------------------------------------------- #
def ordinal_axis(ax, lambdas):
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:g}" for l in lambdas], rotation=90, fontsize=8)
    ax.set_xlim(-0.5, len(lambdas) - 0.5)


def shade_regimes(ax, lambdas):
    # green: valid bias range omega in [0, VALID_MAX)
    gi = next((i for i, l in enumerate(lambdas) if l >= VALID_MAX), len(lambdas))
    ax.axvspan(-0.5, gi - 0.5, color=C_HEALTHY, zorder=0)
    # red: visible degeneration (omega >= DEGEN_OMEGA)
    di = next((i for i, l in enumerate(lambdas) if l >= DEGEN_OMEGA), len(lambdas))
    if di < len(lambdas):
        ax.axvspan(di - 0.5, len(lambdas) - 0.5, color=C_DEGEN, zorder=0)
    return gi, di


def savefig(fig, out_dir: Path, name: str):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf / .svg")


# --------------------------------------------------------------------------- #
def main() -> None:
    # Default to the bundled comparison runs so the script "just works" when
    # launched directly from the IDE (no command-line arguments). Paths are
    # resolved relative to this file, not the current working directory.
    script_dir = Path(__file__).resolve().parent
    base = script_dir / "data" / "carbon-emission-comparison"

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget",  type=Path, default=base / "with_cap",
                    help="run folder (or run_summary.csv) of the 3 Gt CO2 budget run")
    ap.add_argument("--nolimit", type=Path, default=base / "without_cap",
                    help="run folder (or run_summary.csv) of the no-CO2-limit run")
    ap.add_argument("--outdir",  type=Path, default=base / "figures")
    ap.add_argument("--cap-gt",  type=float, default=3.0, help="cumulative CO2 budget [Gt]")
    args = ap.parse_args()

    bud_csv = resolve_summary(args.budget)
    nol_csv = resolve_summary(args.nolimit)
    for tag, p in [("budget", bud_csv), ("nolimit", nol_csv)]:
        if not p.exists():
            raise SystemExit(
                f"run_summary.csv not found for --{tag}: {p}\n"
                f"Pass the run folder explicitly, e.g.:\n"
                f"    python {Path(__file__).name} --budget /path/to/with_cap "
                f"--nolimit /path/to/without_cap"
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading budget  from {bud_csv}")
    print(f"Loading nolimit from {nol_csv}")
    bud = load(bud_csv, "3 Gt CO$_2$ budget")
    nol = load(nol_csv, "No CO$_2$ limit")

    bud_sweep, bud_pf = split(bud)
    nol_sweep, nol_pf = split(nol)

    # Shared categorical omega axis (both scenarios run the same sweep).
    lambdas = sorted(set(bud_sweep["omega"]).union(nol_sweep["omega"]))
    pos = {w: i for i, w in enumerate(lambdas)}

    def xy(sweep):
        s = sweep.dropna(subset=["omega"])
        return [pos[w] for w in s["omega"]], s["emissions_Gt"].values

    print("Generating comparison figure:")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    shade_regimes(ax, lambdas)

    # sweep curves
    xn, yn = xy(nol_sweep)
    xb, yb = xy(bud_sweep)
    ax.plot(xn, yn, "-o", color=C_NOLIM, ms=5, lw=1.8, zorder=4, label="No CO$_2$ limit")
    ax.plot(xb, yb, "-s", color=C_BUDGET, ms=5, lw=1.8, zorder=4, label="3 Gt CO$_2$ budget")

    # perfect-foresight reference (no-limit) and the 3 Gt cap (= budget PF).
    # PF label -> empty top-right corner; cap label -> gap above the 3 Gt line
    # on the left (legend occupies the upper-left).
    ax.axhline(nol_pf, color=C_PF, ls=":", lw=1.8, zorder=2)
    ax.text(len(lambdas) - 0.6, nol_pf, f"PF (no limit) = {nol_pf:.2f} Gt",
            color=C_PF, va="bottom", ha="right", fontsize=8.5)
    ax.axhline(args.cap_gt, color=C_CAP, ls="--", lw=1.8, zorder=2)
    ax.text(0.2, args.cap_gt, "3 Gt cap (= budget PF)",
            color=C_CAP, va="bottom", ha="left", fontsize=8.5)

    ordinal_axis(ax, lambdas)
    # extend the y-range top & bottom so the legend (upper-left) clears the
    # no-limit peak and the labels clear the curves -- no overlaps
    ymin = min(yn.min(), yb.min(), args.cap_gt)
    ymax = max(yn.max(), yb.max(), nol_pf)
    ax.set_ylim(ymin - 0.35, ymax + 0.75)
    ax.set_xlabel(r"Bias weight  $\omega$")
    ax.set_ylabel(r"Cumulative CO$_2$ emissions 2022–2050  [Gt]")
    ax.set_title("CO$_2$ emissions vs. bias weight: 3 Gt budget vs. no CO$_2$ limit")

    # legend: curves + shading regimes (same wording as without_co2_cap.py)
    handles = [
        Line2D([0], [0], marker="o", color=C_NOLIM, lw=1.8, label="No CO$_2$ limit"),
        Line2D([0], [0], marker="s", color=C_BUDGET, lw=1.8, label="3 Gt CO$_2$ budget"),
        Line2D([0], [0], color=C_PF, ls=":", lw=1.8, label="Perfect foresight (no limit)"),
        Line2D([0], [0], color=C_CAP, ls="--", lw=1.8, label="3 Gt cap (= budget PF)"),
        Patch(facecolor=C_HEALTHY, label="valid range $\\omega\\in[0,1)$"),
        Patch(facecolor=C_DEGEN, label="degenerate (from $\\omega\\approx3$)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    fig.tight_layout()
    savefig(fig, outdir, "emissions_vs_bias")

    # --- compact numeric summary ---------------------------------------------
    def at(df, w):
        r = df[np.isclose(df["omega"], w)]
        return float(r["emissions_Gt"].iloc[0]) if len(r) else np.nan
    print("\nScenario            omega=0   omega=1   omega=10   omega=100   PF")
    for name, s, pf in [("3 Gt budget", bud_sweep, bud_pf),
                        ("No CO2 limit", nol_sweep, nol_pf)]:
        print(f"{name:18s} {at(s,0):7.3f}  {at(s,1):7.3f}  {at(s,10):8.3f}  "
              f"{at(s,100):8.3f}  {pf:6.3f}")
    print(f"\nDone -> {outdir}")


if __name__ == "__main__":
    main()
