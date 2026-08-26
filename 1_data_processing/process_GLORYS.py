#!/usr/bin/env python3
"""Python script version of 1_data_processing/GLORYS.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from processing_common import (
    NB_BINS,
    PROJECT_ROOT,
    build_observation_grid,
    compute_glorys_mld,
    grid_glorys_mld,
    kerguelen_mask,
    project_paths,
    rename_obs_coords,
    same_grid_as_observations,
    write_anomaly_products,
    write_r_input,
)


def process_glorys(
    project_root: str | Path = PROJECT_ROOT,
    force: bool = False,
    nb_bins: int = NB_BINS,
    block_size: int = 12,
) -> None:
    paths = project_paths(project_root)
    gridded_file = paths.gridded / "GLORYS_gridded.nc"
    anom_file = paths.gridded / "GLORYS_anom.nc"
    clim_file = paths.gridded / "GLORYS_clim.nc"
    r_file = paths.r_input / "GLORYS_masked.txt"

    # GLORYS.ipynb: observations define the analysis box and final grid.
    ds_obs_raw = xr.open_dataset(paths.data / "CORA_MEOP_ARGO_2026.nc")
    ds_obs_raw = rename_obs_coords(ds_obs_raw)
    ds_obs_grid = build_observation_grid(paths.root, nb_bins=nb_bins)

    if not force and gridded_file.exists() and anom_file.exists() and clim_file.exists():
        with xr.open_dataset(gridded_file) as ds_existing:
            grid_ok = same_grid_as_observations(ds_existing, ds_obs_grid)
            mask_applied = bool(ds_existing["mld"].where(kerguelen_mask(ds_existing)).isnull().all())
        if grid_ok and mask_applied:
            with xr.open_dataset(anom_file) as ds_anom:
                write_r_input(ds_anom, r_file)
            print("Reused existing GLORYS NetCDF products.")
            return
        print(
            "Existing GLORYS outputs are outdated (grid mismatch or missing Kerguelen mask); recomputing GLORYS products."
        )

    # GLORYS.ipynb: open GLORYS and select the observation domain.
    ds_g = xr.open_dataset(paths.data / "GLORYS_2026.nc")
    ds_g = ds_g.sel(
        latitude=slice(ds_obs_raw.latitude.min() - 0.5, ds_obs_raw.latitude.max() + 0.5),
        longitude=slice(ds_obs_raw.longitude.min() - 0.5, ds_obs_raw.longitude.max() + 0.5),
    )

    # GLORYS.ipynb: density-threshold MLD for the full GLORYS field.
    mld = compute_glorys_mld(ds_g, density_threshold=0.03, block_size=block_size)

    # GLORYS.ipynb: bin/interpolate onto observation grid, then apply Kerguelen mask.
    ds_g = grid_glorys_mld(mld, ds_obs_grid, nb_bins=nb_bins)
    mask = kerguelen_mask(ds_g)
    ds_g = ds_g.where(~mask)
    ds_g.to_netcdf(gridded_file)
    print(f"Wrote {gridded_file}")

    # GLORYS.ipynb: remove seasonal cycle from masked product and write R input.
    ds_anom = write_anomaly_products(ds_g, mask, anom_file, clim_file)
    write_r_input(ds_anom, r_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true", help="Recompute even if processed outputs exist.")
    parser.add_argument("--nb-bins", type=int, default=NB_BINS)
    parser.add_argument("--block-size", type=int, default=12, help="Number of GLORYS months per MLD block.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_glorys(
        project_root=args.project_root,
        force=args.force,
        nb_bins=args.nb_bins,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
