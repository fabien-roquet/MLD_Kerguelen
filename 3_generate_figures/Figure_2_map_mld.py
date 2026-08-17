#!/usr/bin/env python3
"""Python script version of Figure_2_map_mld.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from figure_common import add_common_map_layers, cmo, kerguelen_mask, open_gridded, parse_project_root_arg, paths, save_figure, topo_fronts


SECTION_START = (72.0, -52.5)
SECTION_END = (79.0, -47.0)


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 20})

    elevation, ds_front = topo_fronts(args.project_root)
    ds_CMA = open_gridded(args.project_root, "CMA_gridded.nc")
    ds_G = open_gridded(args.project_root, "GLORYS_gridded.nc")
    ds_CL = open_gridded(args.project_root, "GLORYS_CL_gridded.nc")

    if not (
        ds_CMA.longitude.identical(ds_G.longitude)
        and ds_CMA.latitude.identical(ds_G.latitude)
        and ds_CMA.longitude.identical(ds_CL.longitude)
        and ds_CMA.latitude.identical(ds_CL.latitude)
    ):
        raise ValueError(
            "Figure 2 requires CMA/GLORYS/GLORYS_CL on the same observation grid. "
            "Run the data stage to regenerate GLORYS products on the observation grid."
        )

    mask = kerguelen_mask(ds_CMA)
    ds1 = ds_CMA.where(~mask).mean("time")
    ds2 = ds_G.where(~mask).mean("time")
    ds3 = ds_CL.where(~mask).mean("time")

    fig, axs = plt.subplots(1, 3, figsize=(25, 7), gridspec_kw={"wspace": 0.2})
    for ax, img in zip(axs, [ds1.mld, ds2.mld, ds3.mld]):
        pcm = img.plot(x="longitude", cmap=cmo.deep, add_colorbar=False, ax=ax, vmin=0, vmax=200)
        add_common_map_layers(ax, elevation, ds_front, front_color="white")
        ax.plot([SECTION_START[0], SECTION_END[0]], [SECTION_START[1], SECTION_END[1]], color="black", lw=5, zorder=6)
        ax.plot([SECTION_START[0], SECTION_END[0]], [SECTION_START[1], SECTION_END[1]], color="#FFBE0B", lw=3, zorder=6)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.set_xlabel("Longitude [˚E]")
        ax.set_ylabel("")

    axs[0].set_ylabel("Latitude [˚N]")
    pos = axs[2].get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
    fig.colorbar(pcm, cax=cax, label="MLD [m]")

    labels = [("(a)", "CMA"), ("(b)", "GLORYS"), ("(c)", r"GLORYS$_{\mathregular{CL}}$")]
    for ax, (letter, label) in zip(axs, labels):
        ax.tick_params(axis="x",pad=10)
        ax.text(0.01, 0.93, letter, transform=ax.transAxes, size=25, color="white", weight="bold")
        ax.text(0.11, 0.93, label, transform=ax.transAxes, size=25, color="white", weight="bold")

    save_figure(fig, paths(args.project_root)["figures"] / "Figure_2_map_mld.png")


if __name__ == "__main__":
    main()
