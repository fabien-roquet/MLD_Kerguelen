#!/usr/bin/env python3
"""Python script version of Figure_7_1D_trends.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from figure_common import CL_COLOR, CMA_COLOR, G_COLOR, LEGEND_FS, apply_consistent_plot_style, load_fpca, parse_project_root_arg, paths, save_figure
from trend_common import ar1_gls_trend, format_number, format_trend_ci, seasonal_domain_series


N_RECONSTRUCTION_MODES = 50


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    apply_consistent_plot_style()

    fpca = load_fpca(args.project_root, max_modes=N_RECONSTRUCTION_MODES)
    season_months = {"Annual": list(range(1, 13)), "Summer (JFM)": [1, 2, 3], "Winter (JAS)": [7, 8, 9]}
    ds_map = {"GLORYS": fpca["GLORYS"][0], "GLORYS_CL": fpca["GLORYS_CL"][0], "CMA": fpca["CMA"][0]}
    colors = {"GLORYS": G_COLOR, "GLORYS_CL": CL_COLOR, "CMA": CMA_COLOR}

    fig, axes = plt.subplots(3, 1, figsize=(20, 22), sharex=True, tight_layout=True, gridspec_kw={"hspace": 0.15})
    for ax, (sname, months), letter in zip(axes, season_months.items(), ["a", "b", "c"]):
        for name, ds_in in ds_map.items():
            ts_y = seasonal_domain_series(ds_in, months)
            x = ts_y["season_year"].values.astype(float)
            y = ts_y.values
            valid = np.isfinite(x) & np.isfinite(y)
            ax.plot(x[valid], y[valid], marker="o", ms=4, lw=1.3, color=colors[name], alpha=0.5, label="_nolegend_")
            if valid.sum() >= 8:
                trend = ar1_gls_trend(x, y)
                yhat = trend.predict(x[valid])
                label = r"GLORYS$_{\mathregular{CL}}$" if name == "GLORYS_CL" else name
                ax.plot(
                    x[valid],
                    yhat,
                    lw=2.2,
                    color=colors[name],
                    label=rf"{label}: {format_trend_ci(trend)} m yr$^{{-1}}$ ($p_{{AR1}}$={format_number(trend.pvalue)})",
                )
        ax.set_ylabel("MLD [m]")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
        ax.text(0.01, 0.98, f"({letter}) {sname}", transform=ax.transAxes, ha="left", va="top", fontsize=20, fontweight="bold")

    axes[0].set_ylim(-8, 8)
    axes[0].legend(fontsize=LEGEND_FS, loc="upper right", ncol=1)
    axes[1].set_ylim(-15, 15)
    axes[1].legend(fontsize=LEGEND_FS, loc="lower right", ncol=1)
    axes[2].set_ylim(-30, 30)
    axes[2].legend(fontsize=LEGEND_FS, loc="lower left", ncol=1)
    axes[-1].set_xlabel("Time")
    fig.align_ylabels()
    save_figure(fig, paths(args.project_root)["figures"] / "Figure_7_1D_trends.png")


if __name__ == "__main__":
    main()
