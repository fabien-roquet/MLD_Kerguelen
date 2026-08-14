"""Shared utilities for the data-processing notebooks converted to scripts."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import DateOffset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_BINS = 39
END_DATE = pd.Timestamp("2023-12-31")
KERGUELEN_BOX = {
    "lon_min": 68.25,
    "lon_max": 70.75,
    "lat_min": -50.0,
    "lat_max": -48.0,
}
_WARNED_GSW_FALLBACK = False


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    gridded: Path
    r_input: Path


def project_paths(project_root: str | Path = PROJECT_ROOT) -> ProjectPaths:
    root = Path(project_root).resolve()
    gridded = root / "processed" / "1_gridded_data"
    r_input = gridded / "r_input"
    gridded.mkdir(parents=True, exist_ok=True)
    r_input.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, data=root / "data", gridded=gridded, r_input=r_input)


def rename_obs_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "LATITUDE" in ds:
        rename["LATITUDE"] = "latitude"
    if "LONGITUDE" in ds:
        rename["LONGITUDE"] = "longitude"
    return ds.rename(rename) if rename else ds


def make_ds_cut(
    df: pd.DataFrame,
    nb_bins: int = NB_BINS,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp = END_DATE,
    freq: str = "ME",
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Bin point MLD values onto the monthly 39 x 39 grid used in the notebooks."""
    lon_col = "longitude"
    lat_col = "latitude"
    time_col = "time"

    bins_dt = pd.date_range(
        start=start_date if start_date is not None else df[time_col].min() + DateOffset(months=-1),
        end=end_date + DateOffset(months=1),
        freq=freq,
    )

    cut_lat_label = pd.cut(df[lat_col], nb_bins)
    cut_lon_label = pd.cut(df[lon_col], nb_bins)
    cut_time_label = pd.cut(df[time_col], bins=bins_dt)

    df_cut_label = df.drop([lat_col, lon_col, time_col], axis=1)
    df_cut_label = df_cut_label.groupby(
        [cut_time_label, cut_lon_label, cut_lat_label],
        observed=False,
    ).mean()

    lat_mid = pd.IntervalIndex(df_cut_label.index.get_level_values(2)).mid.unique()
    lon_mid = pd.IntervalIndex(df_cut_label.index.get_level_values(1)).mid.unique()
    time_mid = pd.IntervalIndex(df_cut_label.index.get_level_values(0)).mid.unique()

    df_cut_label.index = df_cut_label.index.set_levels(time_mid.values, level=0)
    df_cut_label.index = df_cut_label.index.set_levels(lon_mid, level=1)
    df_cut_label.index = df_cut_label.index.set_levels(lat_mid.values, level=2)

    df_cut = df_cut_label.copy()
    df_cut.replace(0, np.nan, inplace=True)
    df_cut = df_cut.drop(columns=["uid"], errors="ignore")

    ds_cut = df_cut.to_xarray()
    ds_cut["latitude"] = sorted(lat_mid)
    ds_cut["longitude"] = sorted(lon_mid)
    ds_cut["time"] = time_mid
    return ds_cut, df_cut


def kerguelen_mask(ds: xr.Dataset, lon_name: str = "longitude", lat_name: str = "latitude") -> xr.DataArray:
    return (
        (ds[lon_name] >= KERGUELEN_BOX["lon_min"])
        & (ds[lon_name] <= KERGUELEN_BOX["lon_max"])
        & (ds[lat_name] >= KERGUELEN_BOX["lat_min"])
        & (ds[lat_name] <= KERGUELEN_BOX["lat_max"])
    )


def write_r_input(ds: xr.Dataset, out_file: Path) -> None:
    df = ds.where(ds.time.dt.year < 2024, drop=True).to_dataframe().reset_index()
    df["year"] = df.time.dt.year
    df["mth"] = df.time.dt.month
    df = (
        df.drop(columns="time")
        .rename(columns={"latitude": "lat", "longitude": "long"})
        [["long", "lat", "mth", "year", "mld"]]
        .sort_values(["year", "mth", "long", "lat"])
        .reset_index(drop=True)
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, sep=" ", index=False, na_rep="NA")
    print(f"Wrote {out_file}")


def write_anomaly_products(ds: xr.Dataset, mask: xr.DataArray, anom_file: Path, clim_file: Path) -> xr.Dataset:
    masked = ds.where(~mask)
    clim = masked.groupby("time.month").mean("time").mean(["latitude", "longitude"])
    anom = masked.groupby("time.month") - clim
    anom = anom.where(~mask, other=0)
    clim = clim.where(~mask, other=0)
    anom.to_netcdf(anom_file)
    clim.to_netcdf(clim_file)
    print(f"Wrote {anom_file}")
    print(f"Wrote {clim_file}")
    return anom


def same_grid_as_observations(
    ds: xr.Dataset,
    obs_grid: xr.Dataset,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
) -> bool:
    for coord in (lon_name, lat_name, time_name):
        if coord not in ds.coords or coord not in obs_grid.coords:
            return False
        if ds.sizes.get(coord) != obs_grid.sizes.get(coord):
            return False

    lon_ok = np.allclose(ds[lon_name].values, obs_grid[lon_name].values, rtol=0.0, atol=1e-10)
    lat_ok = np.allclose(ds[lat_name].values, obs_grid[lat_name].values, rtol=0.0, atol=1e-10)
    time_ok = np.array_equal(ds[time_name].values, obs_grid[time_name].values)
    return bool(lon_ok and lat_ok and time_ok)


def build_observation_grid(project_root: str | Path = PROJECT_ROOT, nb_bins: int = NB_BINS) -> xr.Dataset:
    paths = project_paths(project_root)
    ds_obs = xr.open_dataset(paths.data / "CORA_MEOP_ARGO_2026.nc")
    ds_obs = rename_obs_coords(ds_obs)
    df_obs = ds_obs.mld.to_dataframe().reset_index()
    ds_grid, _ = make_ds_cut(df_obs, nb_bins=nb_bins)
    return ds_grid


def sigma0_unesco(salinity: xr.DataArray, temperature: xr.DataArray) -> xr.DataArray:
    """EOS-80 density anomaly at atmospheric pressure, used only if gsw is unavailable."""
    t = temperature
    s = salinity
    rho_w = (
        999.842594
        + 6.793952e-2 * t
        - 9.095290e-3 * t**2
        + 1.001685e-4 * t**3
        - 1.120083e-6 * t**4
        + 6.536332e-9 * t**5
    )
    a = 0.824493 - 4.0899e-3 * t + 7.6438e-5 * t**2 - 8.2467e-7 * t**3 + 5.3875e-9 * t**4
    b = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t**2
    c = 4.8314e-4
    return (rho_w + a * s + b * s**1.5 + c * s**2 - 1000.0).rename("sigma0")


def compute_sigma0(ds: xr.Dataset) -> xr.DataArray:
    global _WARNED_GSW_FALLBACK
    try:
        import gsw  # type: ignore
    except ModuleNotFoundError:
        if not _WARNED_GSW_FALLBACK:
            warnings.warn(
                "gsw is not installed; using an EOS-80 sigma0 approximation for GLORYS MLD.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_GSW_FALLBACK = True
        return sigma0_unesco(ds["so"], ds["thetao"])

    return xr.apply_ufunc(
        gsw.sigma0,
        ds["so"],
        ds["thetao"],
        dask="allowed",
        output_dtypes=[float],
    ).rename("sigma0")


def mld_from_density_threshold(ds: xr.Dataset, density_threshold: float, z_ref: float = 10.0) -> xr.DataArray:
    depth = ds["depth"]
    sigma0 = compute_sigma0(ds)
    sigma0_ref = sigma0.interp(depth=z_ref)
    delta_sigma0 = sigma0 - sigma0_ref
    hit = delta_sigma0 >= density_threshold
    depth_index = hit.argmax("depth")
    has_mld = hit.any("depth")
    mld = xr.where(has_mld, depth.isel(depth=depth_index), np.nan)
    mld = mld.where(mld < depth.max())
    mld.name = "mld"
    mld.attrs["standard_name"] = "ocean_mixed_layer_thickness"
    mld.attrs["units"] = depth.attrs.get("units", "m")
    return mld


def compute_glorys_mld(
    ds_glorys: xr.Dataset,
    density_threshold: float,
    z_ref: float = 10.0,
    block_size: int = 12,
) -> xr.DataArray:
    chunks = []
    for start in range(0, ds_glorys.sizes["time"], block_size):
        stop = min(start + block_size, ds_glorys.sizes["time"])
        part = ds_glorys.isel(time=slice(start, stop))
        chunks.append(mld_from_density_threshold(part, density_threshold=density_threshold, z_ref=z_ref).load())
        print(f"Computed GLORYS MLD block {start + 1}-{stop} of {ds_glorys.sizes['time']}")
    return xr.concat(chunks, dim=ds_glorys["time"])


def grid_glorys_mld(mld: xr.DataArray, obs_grid: xr.Dataset, nb_bins: int = NB_BINS) -> xr.Dataset:
    df_glorys = mld.to_dataset(name="mld").mld.to_dataframe().reset_index()
    ds_glorys, _ = make_ds_cut(df_glorys, nb_bins=nb_bins)
    ds_glorys = ds_glorys.interp(
        latitude=obs_grid.latitude,
        longitude=obs_grid.longitude,
        time=obs_grid.time,
        method="nearest",
    )
    # Ensure output coordinates exactly match the observation-based analysis grid.
    return ds_glorys.assign_coords(latitude=obs_grid.latitude, longitude=obs_grid.longitude, time=obs_grid.time)
