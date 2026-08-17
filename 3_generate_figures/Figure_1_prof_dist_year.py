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
from pandas.tseries.offsets import DateOffset

from figure_common import cmo, parse_project_root_arg, paths, save_figure, topo_fronts, kerguelen_mask


KERFIX_LON = 68.4167
KERFIX_LAT = -50.6667
SECTION_START = (72.0, -52.5)
SECTION_END = (79.0, -47.0)
END_DATE = pd.Timestamp("2023-12-31")
PROFILE_KEY_COLS = ["time", "longitude", "latitude", "mld"]


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


def profile_dataframe(ds_profiles: xr.Dataset) -> pd.DataFrame:
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
    return df.assign(
        time=lambda frame: frame["time"].astype("datetime64[ns]"),
        longitude=lambda frame: frame["longitude"].round(6),
        latitude=lambda frame: frame["latitude"].round(6),
        mld=lambda frame: frame["mld"].round(6),
    )


def observation_grid_bins(ds_profiles: xr.Dataset, template: xr.Dataset) -> tuple[pd.IntervalIndex, pd.IntervalIndex, pd.DatetimeIndex]:
    df = profile_dataframe(ds_profiles)
    lon_bins = pd.cut(df["longitude"], template.sizes["longitude"]).cat.categories
    lat_bins = pd.cut(df["latitude"], template.sizes["latitude"]).cat.categories
    time_bins = pd.date_range(
        start=df["time"].min() + DateOffset(months=-1),
        end=END_DATE + DateOffset(months=1),
        freq="ME",
    )
    return lon_bins, lat_bins, time_bins


def profile_counts_on_template_grid(
    profile_df: pd.DataFrame,
    template: xr.Dataset,
    lon_bins: pd.IntervalIndex,
    lat_bins: pd.IntervalIndex,
    time_bins: pd.DatetimeIndex,
) -> xr.DataArray:
    df = profile_df.copy()

    lon_centers = template.longitude.values.astype(float)
    lat_centers = template.latitude.values.astype(float)
    time_centers = pd.DatetimeIndex(template.time.values)

    cut_time = pd.cut(df["time"], bins=time_bins)
    cut_lon = pd.cut(df["longitude"], bins=lon_bins)
    cut_lat = pd.cut(df["latitude"], bins=lat_bins)

    grouped = (
        df.groupby([cut_time, cut_lon, cut_lat], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    grouped = grouped.dropna(subset=["time", "longitude", "latitude"])
    grouped["time"] = pd.IntervalIndex(grouped["time"]).mid.astype("datetime64[ns]")
    grouped["longitude"] = pd.IntervalIndex(grouped["longitude"]).mid.astype(float)
    grouped["latitude"] = pd.IntervalIndex(grouped["latitude"]).mid.astype(float)

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


def split_combined_sources(
    ds_total: xr.Dataset,
    ds_meop: xr.Dataset,
    ds_cora: xr.Dataset,
    ds_argo: xr.Dataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_df = profile_dataframe(ds_total).reset_index(drop=True)
    meop_df = profile_dataframe(ds_meop)
    cora_df = profile_dataframe(ds_cora)
    argo_df = profile_dataframe(ds_argo)

    total_keys = total_df[PROFILE_KEY_COLS].copy()
    meop_keys = meop_df[PROFILE_KEY_COLS].assign(is_meop=True).drop_duplicates()
    cora_keys = cora_df[PROFILE_KEY_COLS].assign(is_cora=True).drop_duplicates()
    argo_keys = argo_df[PROFILE_KEY_COLS].assign(is_argo=True).drop_duplicates()

    classified = (
        total_keys.merge(meop_keys, on=PROFILE_KEY_COLS, how="left")
        .merge(cora_keys, on=PROFILE_KEY_COLS, how="left")
        .merge(argo_keys, on=PROFILE_KEY_COLS, how="left")
        .fillna(False)
    )

    source_hits = classified[["is_meop", "is_cora", "is_argo"]].sum(axis=1)
    if not (source_hits == 1).all():
        bad = int((source_hits != 1).sum())
        raise ValueError(f"Expected each combined profile to map to exactly one source, found {bad} ambiguous rows.")

    meop_mask = classified["is_meop"]
    other_mask = classified["is_cora"] | classified["is_argo"]
    return total_df, total_df.loc[meop_mask].copy(), total_df.loc[other_mask].copy()


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

    lon_bins, lat_bins, time_bins = observation_grid_bins(ds_total, ds_cma_grid)
    total_df, meop_df, other_df = split_combined_sources(ds_total, ds_meop, ds_cora, ds_argo)

    total_count = profile_counts_on_template_grid(total_df, ds_cma_grid, lon_bins, lat_bins, time_bins)
    meop_count = profile_counts_on_template_grid(meop_df, ds_cma_grid, lon_bins, lat_bins, time_bins)
    other_count = profile_counts_on_template_grid(other_df, ds_cma_grid, lon_bins, lat_bins, time_bins)

    mask = kerguelen_mask(ds_cma_grid)
    ds_cma_grid = ds_cma_grid.where(~mask)
    total_count = total_count.where(~mask)
    meop_count = meop_count.where(~mask)
    other_count = other_count.where(~mask)

    map_count = total_count.sum("time")
    print(f"Occupied monthly CMA cells: {int(ds_cma_grid.mld.count().item())}")
    print(f"Raw combined profiles assigned to Figure 1 map: {int(total_count.sum().item())}")
    print(f"Raw MEOP profiles assigned to Figure 1 bars: {int(meop_count.sum().item())}")
    print(f"Raw CORA+ARGO profiles assigned to Figure 1 bars: {int(other_count.sum().item())}")

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
