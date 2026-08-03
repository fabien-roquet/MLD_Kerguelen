#!/usr/bin/env python3
"""Supporting Figure S1 and Table S1: fixed-region MLD anomaly trends."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from appendix_region_trends import N_RECONSTRUCTION_MODES, regional_masks
from figure_common import (
    CL_COLOR,
    CMA_COLOR,
    G_COLOR,
    LEGEND_FS,
    add_common_map_layers,
    align_original_anomaly,
    load_fpca,
    parse_project_root_arg,
    paths,
    save_figure,
    topo_fronts,
)
from trend_common import ar1_gls_trend, format_number, format_trend_ci, seasonal_domain_series


REGION_COLORS = {"A": "#D55E00", "B": "#0072B2", "C": "#009E73"}
PERIODS = {"Annual": list(range(1, 13)), "Summer": [1, 2, 3], "Winter": [7, 8, 9]}


def rmse(ds_pred: xr.Dataset, ds_ref: xr.Dataset, mask: xr.DataArray) -> float:
    err = (ds_pred["mld"] - ds_ref["mld"]).where(mask)
    return float(np.sqrt((err**2).mean(skipna=True)))


def trend_for_mask(ds_in: xr.Dataset, months: list[int], mask: xr.DataArray):
    ts_y = seasonal_domain_series(ds_in, months, mask=mask)
    return ar1_gls_trend(ts_y["season_year"].values.astype(float), ts_y.values)


def write_regional_table(
    out_file: Path,
    ds_map: dict[str, xr.Dataset],
    original_map: dict[str, xr.Dataset],
    masks: dict[str, tuple[str, xr.DataArray]],
    evaluation_mask: xr.DataArray,
) -> None:
    dataset_order = ["GLORYS", "GLORYS_CL", "CMA"]
    dataset_labels = {"GLORYS": "GLORYS", "GLORYS_CL": r"$\textrm{GLORYS}_{\textrm{CL}}$", "CMA": "CMA"}

    rows = []
    for region_code, (region_name, region_mask) in masks.items():
        region_mask = region_mask.fillna(False)
        n_cells = int(region_mask.sum())
        for dataset_name in dataset_order:
            mask = region_mask & evaluation_mask
            rmse_value = rmse(ds_map[dataset_name], original_map[dataset_name], mask)
            trends = [trend_for_mask(ds_map[dataset_name], months, region_mask) for months in PERIODS.values()]
            trend_cells = [
                rf"${format_number(tr.slope)}$ [{format_number(tr.ci_low)}, {format_number(tr.ci_high)}] ({format_number(tr.pvalue)})"
                for tr in trends
            ]
            rows.append(
                f"{region_code} & {region_name} & {dataset_labels[dataset_name]} & {n_cells} & "
                f"{format_number(rmse_value)} & " + " & ".join(trend_cells) + r" \\"
            )

    table = "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            r"\caption{Regional reconstruction skill and AR(1)-adjusted trend estimates. RMSE is evaluated on the common co-located sampling mask. Trend slopes are in $\mathrm{m\,yr^{-1}}$ and are reported as slope [95\% CI] with the AR(1)-adjusted $p$-value in parenthesis.}",
            r"\label{tableS1}",
            r"\small",
            r"\begin{tabular}{c|l|c|c|c|c|c|c}",
            r"\hline",
            r"Region & Description & Dataset & Cells & RMSE & Annual & Summer & Winter \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    out_file.write_text(table)
    print(f"Wrote {out_file}")


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 16})

    elevation, fronts = topo_fronts(args.project_root)
    fpca = load_fpca(args.project_root, max_modes=N_RECONSTRUCTION_MODES)
    ds_map = {"GLORYS": fpca["GLORYS"][0], "GLORYS_CL": fpca["GLORYS_CL"][0], "CMA": fpca["CMA"][0]}
    ds_g = ds_map["GLORYS"]
    masks = regional_masks(ds_g, fronts)

    original_map = {
        "GLORYS": align_original_anomaly(args.project_root, "GLORYS_anom.nc", ds_g),
        "GLORYS_CL": align_original_anomaly(args.project_root, "GLORYS_CL_anom.nc", ds_g),
        "CMA": align_original_anomaly(args.project_root, "CMA_anom.nc", ds_g),
    }
    evaluation_mask = original_map["GLORYS_CL"]["mld"].notnull()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.28, wspace=0.18)
    ax_map = fig.add_subplot(gs[0, :])
    ts_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    region_code = xr.full_like(ds_g["xi1"], np.nan, dtype=float)
    for idx, (code, (_, mask)) in enumerate(masks.items(), start=1):
        region_code = region_code.where(~mask, idx)
    cmap = ListedColormap([REGION_COLORS["A"], REGION_COLORS["B"], REGION_COLORS["C"]])
    ax_map.pcolormesh(ds_g["long"], ds_g["lat"], region_code.transpose("lat", "long"), shading="auto", cmap=cmap, vmin=0.5, vmax=3.5, alpha=0.72)
    add_common_map_layers(ax_map, elevation, fronts)
    ax_map.set_xlabel("Longitude [deg E]")
    ax_map.set_ylabel("Latitude [deg N]")
    ax_map.text(
        0.01,
        0.98,
        "(a) Fixed regional masks",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
    )
    ax_map.legend(
        handles=[Patch(facecolor=REGION_COLORS[code], label=f"{code}: {name}") for code, (name, _) in masks.items()],
        loc="lower left",
        fontsize=LEGEND_FS,
        framealpha=0.9,
    )

    colors = {"GLORYS": G_COLOR, "GLORYS_CL": CL_COLOR, "CMA": CMA_COLOR}
    dataset_labels = {"GLORYS": "GLORYS", "GLORYS_CL": r"GLORYS$_{\mathregular{CL}}$", "CMA": "CMA"}
    for ax, (region_code_key, (region_name, mask)), letter in zip(ts_axes, masks.items(), ["b", "c", "d"]):
        for dataset_name, ds_in in ds_map.items():
            ts_y = seasonal_domain_series(ds_in, PERIODS["Annual"], mask=mask)
            x = ts_y["season_year"].values.astype(float)
            y = ts_y.values
            valid = np.isfinite(x) & np.isfinite(y)
            trend = ar1_gls_trend(x, y)
            ax.plot(x[valid], y[valid], marker="o", ms=3.5, lw=1.1, color=colors[dataset_name], alpha=0.5, label="_nolegend_")
            ax.plot(
                x[valid],
                trend.predict(x[valid]),
                lw=2,
                color=colors[dataset_name],
                label=rf"{dataset_labels[dataset_name]}: {format_trend_ci(trend)}",
            )
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
        ax.set_title(f"({letter}) Region {region_code_key}: {region_name}", loc="left", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("MLD anomaly [m]")
        ax.legend(fontsize=10, loc="best")

    out_dir = paths(args.project_root)["figures"]
    write_regional_table(out_dir / "Table_S1_regional_trends.tex", ds_map, original_map, masks, evaluation_mask)
    for code, (_, mask) in masks.items():
        print(f"Region {code} mask contains {int(mask.sum())} grid cells.")
    save_figure(fig, out_dir / "Figure_S1_regions_trends.png")


if __name__ == "__main__":
    main()
