#!/usr/bin/env python3
"""Python script version of Figure_4_MU_modes.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from figure_common import CL_COLOR, CMA_COLOR, G_COLOR, LEGEND_FS, load_fpca, parse_project_root_arg, paths, save_figure


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 20})

    fpca = load_fpca(args.project_root)
    lambda_map = {"GLORYS": fpca["GLORYS"][6], "GLORYS_CL": fpca["GLORYS_CL"][6], "CMA": fpca["CMA"][6]}
    var_exp = {name: (100 * df_l.iloc[:, 0] / df_l.iloc[:, 0].sum()).to_numpy() for name, df_l in lambda_map.items()}
    phi_nums = [1, 2]

    fig, axes = plt.subplots(
        len(phi_nums),
        1,
        figsize=(15, 10),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.05},
    )
    datasets = [
        ("GLORYS", fpca["GLORYS"][0], fpca["GLORYS"][5], G_COLOR),
        ("GLORYS_CL", fpca["GLORYS_CL"][0], fpca["GLORYS_CL"][5], CL_COLOR),
        ("CMA", fpca["CMA"][0], fpca["CMA"][5], CMA_COLOR),
    ]

    for j, n in enumerate(phi_nums):
        ax_i = axes[j]
        var = f"phi{n}"
        var_name = rf"$\xi_{{{n}}}$"
        for label, ds, _, color in datasets:
            if var in ds:
                ve = var_exp[label][n - 1] if n - 1 < len(var_exp[label]) else float("nan")
                dataset_label = r"GLORYS$_{\mathregular{CL}}$" if label == "GLORYS_CL" else label
                ax_i.plot(ds.time, ds[var], color=color, linewidth=2, label=f"{dataset_label}: {var_name} ({ve:.1f}%)")
        ax_i.legend(loc="upper center", fontsize=LEGEND_FS, ncol=3)
        ax_i.set_ylim(-1, 1)
        ax_i.set_ylabel(rf"$\xi_{{{n}}}$")
        ax_i.axhline(0, color="k", linestyle="--", linewidth=1)
        ax_i.text(0.01, 0.98, f"({chr(97 + j)})", transform=ax_i.transAxes, ha="left", va="top", fontsize=20, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_minor_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Time")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.8)
    fig.align_ylabels()
    save_figure(fig, paths(args.project_root)["figures"] / "Figure_4_MU_modes.png")


if __name__ == "__main__":
    main()
