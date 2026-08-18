#!/usr/bin/env python3
"""Python script version of Figure_3_seasonal_cycle.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_common import CL_COLOR, CMA_COLOR, G_COLOR, LEGEND_FS, apply_consistent_plot_style, open_gridded, parse_project_root_arg, paths, save_figure


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    apply_consistent_plot_style()

    ds_CMA = open_gridded(args.project_root, "CMA_anom.nc")
    ds_CMA_clim = open_gridded(args.project_root, "CMA_clim.nc")
    ds_G = open_gridded(args.project_root, "GLORYS_anom.nc")
    ds_G_clim = open_gridded(args.project_root, "GLORYS_clim.nc")
    ds_CL = open_gridded(args.project_root, "GLORYS_CL_anom.nc")
    ds_CL_clim = open_gridded(args.project_root, "GLORYS_CL_clim.nc")

    datasets = {"G": ds_G, "CL": ds_CL, "CMA": ds_CMA}
    datasets_clim = {"G": ds_G_clim, "CL": ds_CL_clim, "CMA": ds_CMA_clim}
    colors = {"G": G_COLOR, "CL": CL_COLOR, "CMA": CMA_COLOR}
    labels = {"G": "GLORYS", "CL": r"GLORYS$_{CL}$", "CMA": "CMA"}

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), tight_layout=True)
    for key in ["G", "CL", "CMA"]:
        ax[0].plot(datasets_clim[key].month, datasets_clim[key].mld.mean(dim=["latitude", "longitude"]), label=labels[key], color=colors[key])
        ax[1].plot(datasets[key].time, datasets[key].mld.mean(dim=["latitude", "longitude"]), label=labels[key], color=colors[key])

    ax[1].hlines(0, ds_CMA.time.min() - pd.Timedelta(days=365), ds_CMA.time.max() + pd.Timedelta(days=365), color="k", linestyle="--", alpha=0.5)
    ax[1].set_xlim(pd.to_datetime("2006-01-02"), pd.to_datetime("2025-01-01"))
    ax[0].set_xticks(np.arange(1, 13))
    ax[0].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax[0].set_xlabel("Month")
    ax[0].set_ylabel("MLD [m]")
    ax[0].legend(ncol=3, fontsize=LEGEND_FS, loc="upper left", bbox_to_anchor=(0.05, 1.01))
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("MLD Anomaly [m]")

    for j, axis in enumerate(ax):
        axis.grid(True, linestyle="--", alpha=0.8)
        axis.text(0.01, 0.98, f"({chr(97 + j)})", transform=axis.transAxes, ha="left", va="top", fontsize=20, fontweight="bold")
    fig.align_ylabels()
    save_figure(fig, paths(args.project_root)["figures"] / "Figure_3_seasonal_cycle.png")


if __name__ == "__main__":
    main()
