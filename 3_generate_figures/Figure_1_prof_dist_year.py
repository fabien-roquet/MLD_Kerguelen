#!/usr/bin/env python3
"""Build Figure 1 from scratch using CMA grid and source-wise profile counts."""

from __future__ import annotations

import argparse

from matplotlib import transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xarray as xr

from figure_common import cmo, parse_project_root_arg, paths, save_figure, topo_fronts


KERFIX_LON = 68.4167
KERFIX_LAT = -50.6667
SECTION_START = (72.0, -52.5)
SECTION_END = (79.0, -47.0)
END_DATE = pd.Timestamp("2023-12-31")


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    if centers.size < 2:
        raise ValueError("At least two centers are required to build bin edges.")
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def datetime_edges_from_centers(centers: np.ndarray) -> pd.DatetimeIndex:
    if centers.size < 2:
        raise ValueError("At least two time centers are required to build bin edges.")
    dt = pd.DatetimeIndex(centers)
    mids = dt[:-1] + (dt[1:] - dt[:-1]) / 2
    first = dt[0] - (mids[0] - dt[0])
    last = dt[-1] + (dt[-1] - mids[-1])
    return pd.DatetimeIndex(np.concatenate(([first], mids.values, [last])))


def profile_counts_on_template_grid(ds_profiles: xr.Dataset, template: xr.Dataset) -> xr.DataArray:
    if "mld" not in ds_profiles:
        raise ValueError("Input profile dataset must contain an mld variable.")

    df = ds_profiles[["mld"]].to_dataframe().reset_index()
    rename = {}
    if "LONGITUDE" in df.columns:
        rename["LONGITUDE"] = "longitude"
    if "LATITUDE" in df.columns:
        rename["LATITUDE"] = "latitude"
    df = df.rename(columns=rename)

    required_cols = {"time", "longitude", "latitude", "mld"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in profile dataframe: {sorted(missing)}")

    df = df.dropna(subset=["time", "longitude", "latitude", "mld"])
    df = df[df["time"] <= END_DATE]

    lon_centers = template.longitude.values.astype(float)
    lat_centers = template.latitude.values.astype(float)
    time_centers = pd.DatetimeIndex(template.time.values)

    lon_edges = edges_from_centers(lon_centers)
    lat_edges = edges_from_centers(lat_centers)
    time_edges = datetime_edges_from_centers(time_centers.values)

    cut_time = pd.cut(df["time"], bins=time_edges, labels=time_centers, include_lowest=True)
    cut_lon = pd.cut(df["longitude"], bins=lon_edges, labels=lon_centers, include_lowest=True)
    cut_lat = pd.cut(df["latitude"], bins=lat_edges, labels=lat_centers, include_lowest=True)

    grouped = (
        df.groupby([cut_time, cut_lon, cut_lat], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    grouped = grouped.dropna(subset=["time", "longitude", "latitude"])
    grouped["time"] = grouped["time"].astype("datetime64[ns]")
    grouped["longitude"] = grouped["longitude"].astype(float)
    grouped["latitude"] = grouped["latitude"].astype(float)

    da = (
        grouped.set_index(["time", "longitude", "latitude"])["count"]
        .to_xarray()
        .reindex(
            time=time_centers,
            longitude=lon_centers,
            latitude=lat_centers,
            fill_value=0,
        )
        .astype(float)
    )
    da.name = "count"
    return da


def yearly_monthly_counts(da_count: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    yearly = da_count.sum(["latitude", "longitude"]).groupby("time.year").sum()
    monthly = da_count.sum(["latitude", "longitude"]).groupby("time.month").sum()
    return yearly, monthly


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 20})

    p = paths(args.project_root)
    elevation, ds_front = topo_fronts(args.project_root)

    ds_cma_grid = xr.open_dataset(p["gridded"] / "CMA_gridded.nc")
    ds_total = xr.open_dataset(p["data"] / "CORA_MEOP_ARGO_2026.nc")
    ds_cora = xr.open_dataset(p["data"] / "CORA_2026.nc")
    ds_argo = xr.open_dataset(p["data"] / "ARGO_2026.nc")
    ds_meop = xr.open_dataset(p["data"] / "MEOP_2026.nc")

    total_count = profile_counts_on_template_grid(ds_total, ds_cma_grid)
    cora_count = profile_counts_on_template_grid(ds_cora, ds_cma_grid)
    argo_count = profile_counts_on_template_grid(ds_argo, ds_cma_grid)
    meop_count = profile_counts_on_template_grid(ds_meop, ds_cma_grid)
    other_count = cora_count + argo_count

    map_count = total_count.sum("time")

    hist_other_y, hist_other_m = yearly_monthly_counts(other_count)
    hist_meop_y, hist_meop_m = yearly_monthly_counts(meop_count)

    all_years = np.arange(2007, 2024)
    months_idx = np.arange(1, 13)

    hist_other_y = hist_other_y.reindex(year=all_years, fill_value=0)
    hist_meop_y = hist_meop_y.reindex(year=all_years, fill_value=0)
    hist_other_m = hist_other_m.reindex(month=months_idx, fill_value=0)
    hist_meop_m = hist_meop_m.reindex(month=months_idx, fill_value=0)

    fig = plt.figure(figsize=(25, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.2)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_year = fig.add_subplot(gs[0, 1])
    ax_month = fig.add_subplot(gs[1, 1])

    meop_color = "#B5446E"
    other_color = "#F6BD60"
    kerfix_color = "#F1BE08"

    bounds_1 = np.arange(0, 100, 20)
    bounds_2 = np.arange(100, 1600, 200)
    bounds = np.concatenate((bounds_1, bounds_2))

    pcm = map_count.where(map_count > 0).plot(
        x="longitude",
        y="latitude",
        cmap=cmo.matter,
        levels=bounds,
        add_colorbar=False,
        ax=ax_map,
    )

    (elevation / elevation).where(elevation > 0).plot(add_colorbar=False, cmap="gist_yarg", ax=ax_map)
    cs = (-elevation).plot.contour(levels=(500, 1000, 2000), colors=["black"], linewidths=1, ax=ax_map)
    ax_map.plot(ds_front.LonSAF.where(ds_front.LatSAF > -50), ds_front.LatSAF.where(ds_front.LatSAF > -50), c="k", label="SAF",linewidth=2.5)
    ax_map.plot(ds_front.LonPF, ds_front.LatPF, c="k", label="PF",linewidth=2.5)
    ax_map.plot(ds_front.LonSACCF, ds_front.LatSACCF, c="k", label="SACCF",linewidth=2.5)

    ax_map.clabel(cs, inline=True, fmt="%1.0f", fontsize=10)
    ax_map.set_ylabel("Latitude [˚N]")
    ax_map.set_xlabel("Longitude [˚E]")
    ax_map.set_xticks([60, 65, 70, 75, 80])
    ax_map.set_yticks([-60, -55, -50, -45, -40])
    ax_map.tick_params(axis="x", which="major", pad=12)

    ax_map.plot(KERFIX_LON, KERFIX_LAT, marker="*", markersize=30, color=kerfix_color, zorder=7, mec="black", mew=2)
    ax_map.text(
        64.8,
        -50.55,
        "KERFIX",
        fontweight="bold",
        color=kerfix_color,
        va="center",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.2"},
        zorder=7,
    )
    ax_map.plot([SECTION_START[0], SECTION_END[0]], [SECTION_START[1], SECTION_END[1]], color="black", lw=5, zorder=6)
    ax_map.plot([SECTION_START[0], SECTION_END[0]], [SECTION_START[1], SECTION_END[1]], color="#FFBE0B", lw=3, zorder=6)
    ax_map.annotate(
        "A",
        xy=(SECTION_START[0] - 0.7, SECTION_START[1] - 0.5),
        weight="bold",
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#FFBE0B",
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.2"},
    )

    cb = plt.colorbar(pcm, orientation="vertical", ax=ax_map, ticks=bounds)
    cb.set_label(label="Number of profiles")

    ax_year.bar(
        all_years,
        hist_meop_y.values,
        align="center",
        width=0.8,
        alpha=0.8,
        bottom=hist_other_y.values,
        color=meop_color,
        edgecolor="black",
        label="MEOP",
    )
    ax_year.bar(
        all_years,
        hist_other_y.values,
        align="center",
        width=0.8,
        alpha=0.8,
        color=other_color,
        edgecolor="black",
        label="CORA+ARGO",
    )

    ax_year.yaxis.tick_right()
    ax_year.yaxis.set_label_position("right")
    ax_year.set_ylabel("Number of profiles")
    ax_year.xaxis.set_major_locator(ticker.FixedLocator(np.arange(2007, 2024, 2)))
    ax_year.legend(loc="upper left")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_month.bar(
        months_idx,
        hist_meop_m.values,
        align="center",
        width=0.8,
        alpha=0.8,
        bottom=hist_other_m.values,
        color=meop_color,
        edgecolor="black",
        label="MEOP",
    )
    ax_month.bar(
        months_idx,
        hist_other_m.values,
        align="center",
        width=0.8,
        alpha=0.8,
        color=other_color,
        edgecolor="black",
        label="CORA+ARGO",
    )

    ax_month.yaxis.tick_right()
    ax_month.yaxis.set_label_position("right")
    ax_month.set_ylabel("Number of profiles")
    ax_month.set_xlabel("")
    ax_month.set_xticks(months_idx)
    ax_month.set_xticklabels(month_labels)

    for axes, panel_label in zip([ax_map, ax_year, ax_month], ["(a)", "(b)", "(c)"]):
        if axes == ax_map:
            axes.text(0.01, 0.95, panel_label, transform=axes.transAxes, weight="bold", fontsize=30)
        else:
            axes.text(0.92, 0.9, panel_label, transform=axes.transAxes, weight="bold", fontsize=30)

    save_figure(fig, p["figures"] / "Figure_1_prof_dist.png")


if __name__ == "__main__":
    main()
