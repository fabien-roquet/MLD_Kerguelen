#!/usr/bin/env python3
"""Python script version of Figure_8_trend_maps.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from figure_common import add_common_map_layers, cmo, load_fpca, parse_project_root_arg, paths, save_figure, topo_fronts
from trend_common import seasonal_trend_map


N_RECONSTRUCTION_MODES = 50


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 20})

    elevation, ds_front = topo_fronts(args.project_root)
    fpca = load_fpca(args.project_root, max_modes=N_RECONSTRUCTION_MODES)
    ds_map = {"GLORYS": fpca["GLORYS"][0], "GLORYS_CL": fpca["GLORYS_CL"][0], "CMA": fpca["CMA"][0]}
    periods = {"Annual": list(range(1, 13)), "Summer (JFM)": [1, 2, 3], "Winter (JAS)": [7, 8, 9]}
    dataset_order = ["GLORYS", "GLORYS_CL", "CMA"]
    trend_maps_period_dataset = {
        p_name: {d_name: seasonal_trend_map(ds_map[d_name], months) for d_name in dataset_order}
        for p_name, months in periods.items()
    }

    fig, axes = plt.subplots(
        nrows=len(periods),
        ncols=len(dataset_order),
        figsize=(18, 15),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    cbar_ticks_uniform = np.arange(-2.0, 2.5, 1)
    for i, period_name in enumerate(periods.keys()):
        for j, d_name in enumerate(dataset_order):
            ax = axes[i, j]
            tr = trend_maps_period_dataset[period_name][d_name]
            slope = tr["slope"]
            im = ax.pcolormesh(slope["long"], slope["lat"], slope.transpose("lat", "long"), shading="auto", cmap=cmo.balance, vmin=-2, vmax=2)
            significant = tr["significant"].transpose("lat", "long").values > 0
            lon2d, lat2d = np.meshgrid(slope["long"].values, slope["lat"].values)
            ax.scatter(lon2d[significant], lat2d[significant], s=3, c="k", marker=".", alpha=0.65, linewidths=0)
            add_common_map_layers(ax, elevation, ds_front)
            ax.set_xlabel("Longitude [deg E]" if i == len(periods) - 1 else "")
            ax.set_ylabel("Latitude [deg N]" if j == 0 else "")
        cbar = fig.colorbar(im, ax=axes[i, :], orientation="vertical", shrink=0.95, pad=0.02, ticks=cbar_ticks_uniform)
        cbar.set_label("MLD trend [m yr$^{-1}$]")

    for i, ax_i in enumerate(axes.ravel()):
        period_idx = i // len(dataset_order)
        dataset_idx = i % len(dataset_order)
        d_name = dataset_order[dataset_idx]
        label = r"GLORYS$_{\mathregular{CL}}$" if d_name == "GLORYS_CL" else d_name
        ax_i.text(
            0.01,
            0.98,
            f"({chr(97 + i)}) {label} - {list(periods.keys())[period_idx]}",
            transform=ax_i.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
        )

    save_figure(fig, paths(args.project_root)["figures"] / "Figure_8_2D_trends.png")


if __name__ == "__main__":
    main()
