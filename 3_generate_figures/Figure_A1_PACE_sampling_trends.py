#!/usr/bin/env python3
"""Appendix Figure A1: GLORYS random-sampling PACE annual trends."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_common import G_COLOR, LEGEND_FS, parse_project_root_arg, paths, require_file, save_figure
from trend_common import ar1_gls_trend


SAMPLING_COLORS = {
    5: "#7B3294",
    10: "#008837",
    20: "#E08214",
    50: "#0571B0",
    100: G_COLOR,
}


def sampling_dir(project_root: str | Path) -> Path:
    return paths(project_root)["root"] / "processed" / "2_fPCA" / "GLORYS_random_sampling"


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 8:
        return np.nan, np.nan
    trend = ar1_gls_trend(x[valid], y[valid])
    return float(trend.intercept - trend.slope * trend.x_mean), float(trend.slope)


def format_slope(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.2f} +/- {std:.2f}"


def write_latex_table(summary: pd.DataFrame, out_file: Path) -> None:
    rows = []
    for row in summary.itertuples(index=False):
        n_obs_std = 0.0 if not np.isfinite(row.n_observations_std) else row.n_observations_std
        slope_std = 0.0 if not np.isfinite(row.slope_std) else row.slope_std
        rows.append(
            rf"{int(row.percentage)}\% & {int(row.n_replicates)} & "
            rf"${row.n_observations_mean:.0f} \pm {n_obs_std:.0f}$ & "
            rf"${row.slope_mean:.2f} \pm {slope_std:.2f}$ \\"
        )

    table = "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            r"\caption{Annual global MLD-anomaly trend sensitivity to random GLORYS space-time sampling. Slopes are in $\mathrm{m\,yr^{-1}}$ and reported as mean $\pm$ standard deviation across pseudo-random replicates.}",
            r"\label{tableA1}",
            r"\small",
            r"\begin{tabular}{c|c|c|c}",
            r"\hline",
            r"Sampling & Replicates & Observations & Trend \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    out_file.write_text(table)


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 20})

    base_dir = sampling_dir(args.project_root)
    series_file = require_file(
        base_dir / "global_annual_series.csv",
        "Run `Rscript 2_compute_fPCA_R/script_PCA_GLORYS_random_sampling_2026.R` first.",
    )
    summary_file = require_file(base_dir / "global_trend_summary.csv")
    series = pd.read_csv(series_file)
    summary = pd.read_csv(summary_file).sort_values("percentage")

    fig, ax = plt.subplots(figsize=(15, 6), tight_layout=True)
    for percentage in sorted(series["percentage"].unique()):
        percentage = int(percentage)
        sub = series.loc[series["percentage"].eq(percentage)]
        if sub.empty:
            continue

        grouped = sub.groupby("year")["mld"]
        year = grouped.mean().index.to_numpy(dtype=float)
        mean = grouped.mean().to_numpy(dtype=float)
        std = grouped.std().fillna(0.0).to_numpy(dtype=float)
        color = SAMPLING_COLORS[percentage]

        if percentage != 100:
            ax.fill_between(year, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.plot(year, mean, marker="o", ms=4, lw=1.5 if percentage != 100 else 2.2, color=color, alpha=0.85)

        intercept, slope = fit_line(year, mean)
        if np.isfinite(slope):
            ax.plot(year, intercept + slope * year, color=color, lw=2.2, linestyle="-" if percentage == 100 else "--")

        stat = summary.loc[summary["percentage"].eq(percentage)]
        slope_text = ""
        if not stat.empty:
            slope_text = format_slope(float(stat["slope_mean"].iloc[0]), float(stat["slope_std"].iloc[0]))
        ax.plot([], [], color=color, lw=2.5, label=rf"{percentage}%: {slope_text} m yr$^{{-1}}$")

    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("MLD anomaly [m]")
    ax.grid(alpha=0.3)
    ax.text(
        0.01,
        0.98,
        "(a) Annual PACE anomaly reconstruction",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    ax.legend(fontsize=LEGEND_FS, loc="best", ncol=2)

    out_dir = paths(args.project_root)["figures"]
    write_latex_table(summary, out_dir / "Table_A1_PACE_sampling_trends.tex")
    save_figure(fig, out_dir / "Figure_A1_PACE_sampling_trends.png")


if __name__ == "__main__":
    main()
