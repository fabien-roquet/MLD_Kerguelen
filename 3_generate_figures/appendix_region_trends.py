"""Shared helpers for appendix regional trend figures."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from figure_common import CL_COLOR, CMA_COLOR, G_COLOR, LEGEND_FS, load_fpca, paths, save_figure, topo_fronts
from trend_common import ar1_gls_trend, format_number, format_trend_ci, seasonal_domain_series


MaskFunction = Callable[[xr.Dataset, xr.Dataset], xr.DataArray]
N_RECONSTRUCTION_MODES = 50


def latitude_grid(ds: xr.Dataset) -> xr.DataArray:
    _, lat2d = xr.broadcast(ds["long"], ds["lat"])
    return lat2d


def saf_latitude_grid(ds: xr.Dataset, fronts: xr.Dataset) -> xr.DataArray:
    lon_saf = np.asarray(fronts["LonSAF"].values).ravel()
    lat_saf = np.asarray(fronts["LatSAF"].values).ravel()
    valid = np.isfinite(lon_saf) & np.isfinite(lat_saf)
    lon_saf = lon_saf[valid]
    lat_saf = lat_saf[valid]
    if len(lon_saf) == 0:
        raise ValueError("SAF coordinates are empty.")

    order = np.argsort(lon_saf)
    lon_saf = lon_saf[order]
    lat_saf = lat_saf[order]
    unique_lon = np.unique(lon_saf)
    unique_lat = np.array([np.nanmean(lat_saf[lon_saf == lon]) for lon in unique_lon])
    lat_on_grid = np.interp(ds["long"].values, unique_lon, unique_lat, left=np.nan, right=np.nan)
    saf_lat = xr.DataArray(lat_on_grid, coords={"long": ds["long"]}, dims=("long",))
    saf2d, _ = xr.broadcast(saf_lat, ds["lat"])
    return saf2d


def region_a_mask(ds_glorys: xr.Dataset, fronts: xr.Dataset) -> xr.DataArray:
    """Deep region north of the SAF inside the GLORYS cp1 positive lobe."""
    return (latitude_grid(ds_glorys) > saf_latitude_grid(ds_glorys, fronts)) & (ds_glorys["xi1"] > 0.0)


def region_b_mask(ds_glorys: xr.Dataset, fronts: xr.Dataset) -> xr.DataArray:
    """Deep-south region inside the GLORYS cp1 positive lobe."""
    _ = fronts
    return (latitude_grid(ds_glorys) < -50.0) & (ds_glorys["xi1"] > 0.0)


def region_c_mask(ds_glorys: xr.Dataset, fronts: xr.Dataset) -> xr.DataArray:
    """Shallow region inside the GLORYS cp1 negative lobe."""
    _ = fronts
    return ds_glorys["xi1"] < 0.0


def regional_masks(ds_glorys: xr.Dataset, fronts: xr.Dataset) -> dict[str, tuple[str, xr.DataArray]]:
    return {
        "A": ("Deep north of SAF", region_a_mask(ds_glorys, fronts)),
        "B": ("Deep south", region_b_mask(ds_glorys, fronts)),
        "C": ("Shallow cp1", region_c_mask(ds_glorys, fronts)),
    }


def plot_region_trends(project_root: str | Path, mask_func: MaskFunction, mask_label: str, output_name: str) -> None:
    plt.rcParams.update({"font.size": 20})

    fpca = load_fpca(project_root, max_modes=N_RECONSTRUCTION_MODES)
    _, fronts = topo_fronts(project_root)
    ds_map = {"GLORYS": fpca["GLORYS"][0], "GLORYS_CL": fpca["GLORYS_CL"][0], "CMA": fpca["CMA"][0]}
    colors = {"GLORYS": G_COLOR, "GLORYS_CL": CL_COLOR, "CMA": CMA_COLOR}
    mask = mask_func(ds_map["GLORYS"], fronts).fillna(False)
    mask_count = int(mask.sum())
    if mask_count == 0:
        raise ValueError(f"{mask_label} mask is empty.")

    season_months = {"Annual": list(range(1, 13)), "Summer (JFM)": [1, 2, 3], "Winter (JAS)": [7, 8, 9]}

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.07, top=0.98, hspace=0.06)
    for ax, (sname, months), letter in zip(axes, season_months.items(), ["a", "b", "c"]):
        for name, ds_in in ds_map.items():
            ts_y = seasonal_domain_series(ds_in, months, mask=mask)
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

    axes[0].legend(fontsize=LEGEND_FS, loc="lower center", ncol=1)
    axes[1].legend(fontsize=LEGEND_FS, loc="upper right", ncol=1)
    axes[2].legend(fontsize=LEGEND_FS, loc="upper right", ncol=1)
    axes[-1].set_xlabel("Time")
    fig.align_ylabels()

    print(f"{mask_label} mask contains {mask_count} grid cells.")
    save_figure(fig, paths(project_root)["figures"] / output_name)
