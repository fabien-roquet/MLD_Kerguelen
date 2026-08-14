#!/usr/bin/env python3
"""Appendix Figure A2: annual trend maps from random-sampling PACE reconstructions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from figure_common import add_common_map_layers, cmo, parse_project_root_arg, paths, require_file, topo_fronts


MAP_PERCENTAGES = [5, 10, 20]


def sampling_dir(project_root: str | Path) -> Path:
    return paths(project_root)["root"] / "processed" / "2_fPCA" / "GLORYS_random_sampling"


def trend_map_to_dataarray(df: pd.DataFrame) -> xr.DataArray:
    grouped = df.groupby(["long", "lat"], as_index=False)["slope"].mean()
    return grouped.set_index(["long", "lat"]).to_xarray()["slope"]


def format_slope(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.2f} +/- {std:.2f}"


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 16})

    base_dir = sampling_dir(args.project_root)
    maps_file = require_file(
        base_dir / "annual_trend_maps.csv",
        "Run `Rscript 2_compute_fPCA_R/script_PCA_GLORYS_random_sampling_2026.R` first.",
    )
    summary_file = require_file(base_dir / "global_trend_summary.csv")
    maps = pd.read_csv(maps_file)
    summary = pd.read_csv(summary_file)
    elevation, ds_front = topo_fronts(args.project_root)

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(
        1,
        len(MAP_PERCENTAGES) + 1,
        width_ratios=[1, 1, 1, 0.045],
        left=0.06,
        right=0.94,
        bottom=0.14,
        top=0.94,
        wspace=0.12,
    )
    axes = [fig.add_subplot(gs[0, 0])]
    axes.extend(fig.add_subplot(gs[0, i], sharex=axes[0], sharey=axes[0]) for i in range(1, len(MAP_PERCENTAGES)))
    cax = fig.add_subplot(gs[0, -1])
    cbar_ticks = np.arange(-2.0, 2.5, 1)
    im = None
    for i, (ax, percentage) in enumerate(zip(axes, MAP_PERCENTAGES)):
        sub = maps.loc[maps["percentage"].eq(percentage)]
        if sub.empty:
            raise ValueError(f"Missing {percentage}% sampling trend maps. Rerun the sampling stage with --sampling-levels including {percentage}.")
        tr = trend_map_to_dataarray(sub)
        im = ax.pcolormesh(
            tr["long"],
            tr["lat"],
            tr.transpose("lat", "long"),
            shading="auto",
            cmap=cmo.balance,
            vmin=-2,
            vmax=2,
        )
        add_common_map_layers(ax, elevation, ds_front)
        ax.set_xlabel("Longitude [˚E]")
        ax.set_ylabel("Latitude [˚N]" if i == 0 else "")

        stat = summary.loc[summary["percentage"].eq(percentage)]
        slope_text = ""
        if not stat.empty:
            slope_text = format_slope(float(stat["slope_mean"].iloc[0]), float(stat["slope_std"].iloc[0]))
            slope_text = rf"{slope_text} m yr$^{{-1}}$"
        ax.text(
            0.01,
            0.98,
            f"({chr(97 + i)}) {percentage}%\n{slope_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
        )

    if im is not None:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical", ticks=cbar_ticks)
        cbar.set_label("MLD anomaly trend [m yr$^{-1}$]")

    out_file = paths(args.project_root)["figures"] / "Figure_A2_PACE_sampling_trend_maps.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
