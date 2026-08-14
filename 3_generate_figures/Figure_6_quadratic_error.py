#!/usr/bin/env python3
"""Python script version of Figure_6_quadratic_error.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from figure_common import (
    CL_COLOR,
    CMA_COLOR,
    G_COLOR,
    add_common_map_layers,
    align_original_anomaly,
    cmo,
    load_fpca,
    parse_project_root_arg,
    paths,
    save_figure,
    topo_fronts,
)


N_RECONSTRUCTION_MODES = 50


def compute_qe(ds_pred, ds_ref, var="mld", mask=None):
    err = ds_pred[var] - ds_ref[var]
    if mask is not None:
        err = err.where(mask)
    qe = np.sqrt((err**2).mean(dim="time", skipna=True))
    return qe, qe.where(qe > 0)


def monthly_rmse(ds_pred, ds_ref, var="mld", mask=None):
    err = ds_pred[var] - ds_ref[var]
    if mask is not None:
        err = err.where(mask)
    return np.sqrt((err**2).groupby("time.month").mean(skipna=True).mean(["lat", "long"], skipna=True))


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 20})

    elevation, ds_front = topo_fronts(args.project_root)
    fpca = load_fpca(args.project_root, max_modes=N_RECONSTRUCTION_MODES)
    ds_G = fpca["GLORYS"][0]
    ds_CL = fpca["GLORYS_CL"][0]
    ds_CMA = fpca["CMA"][0]

    ds_G_og = align_original_anomaly(args.project_root, "GLORYS_anom.nc", ds_G)
    ds_CL_og = align_original_anomaly(args.project_root, "GLORYS_CL_anom.nc", ds_G)
    ds_CMA_og = align_original_anomaly(args.project_root, "CMA_anom.nc", ds_G)

    evaluation_mask = ds_CL_og["mld"].notnull()
    qe_g, qe_plot_g = compute_qe(ds_G, ds_G_og, mask=evaluation_mask)
    qe_cl, qe_plot_cl = compute_qe(ds_CL, ds_CL_og, mask=evaluation_mask)
    qe_cma, qe_plot_cma = compute_qe(ds_CMA, ds_CMA_og, mask=evaluation_mask)

    fig = plt.figure(figsize=(30, 20))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1], wspace=0.2, hspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    cax = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, :])

    map_items = [
        (ax1, ds_G, qe_plot_g, "(a) GLORYS", "Latitude [˚N]"),
        (ax2, ds_CL, qe_plot_cl, r"(b) GLORYS$_{\mathregular{CL}}$", ""),
        (ax3, ds_CMA, qe_plot_cma, "(c) CMA", ""),
    ]
    for ax, ds_i, qe_i, panel_txt, ylab in map_items:
        pcm = ax.pcolormesh(ds_i["long"], ds_i["lat"], qe_i.transpose("lat", "long"), shading="auto", cmap=cmo.amp, vmin=0, vmax=80)
        add_common_map_layers(ax, elevation, ds_front)
        ax.plot(ds_front.LonSAF.where(ds_front.LatSAF > -50), ds_front.LatSAF.where(ds_front.LatSAF > -50), c="k", lw=2.5, zorder=5)
        ax.plot(ds_front.LonPF, ds_front.LatPF, c="k", lw=2.5, zorder=5)
        ax.plot(ds_front.LonSACCF, ds_front.LatSACCF, c="k", lw=2.5, zorder=5)
        ax.tick_params(axis="x", pad=10)
        ax.set_xlabel("Longitude [˚E]")
        ax.set_ylabel(ylab)
        ax.text(
            0.01,
            0.98,
            panel_txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
        )
    ax2.tick_params(axis="y", labelleft=False)
    ax3.tick_params(axis="y", labelleft=False)
    fig.colorbar(pcm, cax=cax).set_label("RMSE [m]")

    for data_ts, color_ts, label_ts in zip(
        [
            monthly_rmse(ds_G, ds_G_og, mask=evaluation_mask),
            monthly_rmse(ds_CL, ds_CL_og, mask=evaluation_mask),
            monthly_rmse(ds_CMA, ds_CMA_og, mask=evaluation_mask),
        ],
        [G_COLOR, CL_COLOR, CMA_COLOR],
        ["GLORYS", "GLORYS_CL", "CMA"],
    ):
        ax4.plot(data_ts.month, data_ts, color=color_ts, lw=2, marker="o", label=label_ts)
    ax4.set_xlabel("Month")
    ax4.set_ylabel("RMSE [m]")
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 60)
    ax4.text(0.01, 0.98, "(d)", transform=ax4.transAxes, ha="left", va="top", fontsize=20, fontweight="bold")

    print(f"The mean RMSE of GLORYS: {qe_g.mean().item():.2f}")
    print(f"The standard deviation of the RMSE of GLORYS is: {qe_g.std().item():.2f}")
    print(f"The mean RMSE of GLORYS_CL: {qe_cl.mean().item():.2f}")
    print(f"The standard deviation of the RMSE of GLORYS_CL is: {qe_cl.std().item():.2f}")
    print(f"The mean RMSE of CMA: {qe_cma.mean().item():.2f}")
    print(f"The standard deviation of the RMSE of CMA is: {qe_cma.std().item():.2f}")
    save_figure(fig, paths(args.project_root)["figures"] / "Figure_6_RMSE_mask.png")


if __name__ == "__main__":
    main()
