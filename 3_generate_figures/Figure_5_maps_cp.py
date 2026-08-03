#!/usr/bin/env python3
"""Python script version of Figure_5_maps_cp.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from figure_common import add_common_map_layers, cmo, load_fpca, parse_project_root_arg, paths, save_figure, topo_fronts


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 20})

    elevation, ds_front = topo_fronts(args.project_root)
    fpca = load_fpca(args.project_root)
    datasets = [("GLORYS", fpca["GLORYS"][0]), ("GLORYS_CL", fpca["GLORYS_CL"][0]), ("CMA", fpca["CMA"][0])]

    def plot_mode_panel(ax, ds_i, mode_name, is_bottom=False, is_left=False):
        vlim = 250 if mode_name == "xi1" else 80
        im = ax.pcolormesh(
            ds_i["long"],
            ds_i["lat"],
            ds_i[mode_name].transpose("lat", "long"),
            shading="auto",
            cmap=cmo.balance,
            vmin=-vlim,
            vmax=vlim,
        )
        add_common_map_layers(ax, elevation, ds_front, front_color="white")
        ax.set_xlabel("Longitude [deg E]" if is_bottom else "")
        ax.set_ylabel("Latitude [deg N]" if is_left else "")
        ax.tick_params(axis="x", labelbottom=is_bottom)
        ax.tick_params(axis="y", labelleft=is_left)
        return im

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.08, wspace=0.05)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for j, (ax_i, (_, ds_i)) in enumerate(zip(axes_top, datasets)):
        im_top = plot_mode_panel(ax_i, ds_i, "xi1", is_bottom=False, is_left=(j == 0))

    axes_bottom = [fig.add_subplot(gs[1, i], sharex=axes_top[i], sharey=axes_top[i]) for i in range(3)]
    for j, (ax_i, (_, ds_i)) in enumerate(zip(axes_bottom, datasets)):
        im_bottom = plot_mode_panel(ax_i, ds_i, "xi2", is_bottom=True, is_left=(j == 0))

    cbar1 = fig.colorbar(im_top, ax=axes_top, orientation="vertical", shrink=0.9, pad=0.02)
    cbar1.set_label(r"c$_{p1}$")
    cbar2 = fig.colorbar(im_bottom, ax=axes_bottom, orientation="vertical", shrink=0.9, pad=0.02)
    cbar2.set_label(r"c$_{p2}$")

    panel_dataset_labels = [label for label, _ in datasets] * 2
    for i, ax_i in enumerate(axes_top + axes_bottom):
        label = r"GLORYS$_{\mathregular{CL}}$" if panel_dataset_labels[i] == "GLORYS_CL" else panel_dataset_labels[i]
        ax_i.text(
            0.01,
            0.98,
            f"({chr(97 + i)}) {label}",
            transform=ax_i.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
        )

    save_figure(fig, paths(args.project_root)["figures"] / "Figure_5_cp_maps.png")


if __name__ == "__main__":
    main()
