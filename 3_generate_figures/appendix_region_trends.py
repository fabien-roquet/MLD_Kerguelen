"""Shared helpers for appendix regional trend figures."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import linregress

from figure_common import CL_COLOR, CMA_COLOR, G_COLOR, LEGEND_FS, load_fpca, paths, save_figure, topo_fronts


MaskFunction = Callable[[xr.Dataset, xr.Dataset], xr.DataArray]


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


def seasonal_series_masked(ds_in: xr.Dataset, months: list[int], mask: xr.DataArray) -> xr.DataArray:
    ts = ds_in["mld"].where(mask).mean(dim=("long", "lat"), skipna=True)
    sub = ts.where(ts["time"].dt.month.isin(months), drop=True)
    if set(months) == {12, 1, 2}:
        season_year = xr.where(sub["time"].dt.month == 12, sub["time"].dt.year + 1, sub["time"].dt.year)
    else:
        season_year = sub["time"].dt.year
    return sub.groupby(season_year.rename("season_year")).mean("time", skipna=True)


def plot_region_trends(project_root: str | Path, mask_func: MaskFunction, mask_label: str, output_name: str) -> None:
    plt.rcParams.update({"font.size": 20})

    fpca = load_fpca(project_root)
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
            ts_y = seasonal_series_masked(ds_in, months, mask)
            x = ts_y["season_year"].values.astype(float)
            y = ts_y.values
            valid = np.isfinite(x) & np.isfinite(y)
            ax.plot(x[valid], y[valid], marker="o", ms=4, lw=1.3, color=colors[name], alpha=0.5, label="_nolegend_")
            if valid.sum() >= 8:
                lr = linregress(x[valid], y[valid])
                yhat = lr.intercept + lr.slope * x[valid]
                label = r"GLORYS$_{\mathregular{CL}}$" if name == "GLORYS_CL" else name
                ax.plot(
                    x[valid],
                    yhat,
                    lw=2.2,
                    color=colors[name],
                    label=f"{label}: {lr.slope:.2f} m yr$^{{-1}}$ (p={lr.pvalue:.2f})",
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
