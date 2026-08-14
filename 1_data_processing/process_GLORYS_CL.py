#!/usr/bin/env python3
"""Python script version of 1_data_processing/GLORYS_CL.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


def process_glorys_cl(
    project_root: str | Path = PROJECT_ROOT,
    force: bool = False,
    nb_bins: int = NB_BINS,
    block_size: int = 12,
) -> None:
    paths = project_paths(project_root)
    gridded_file = paths.gridded / "GLORYS_CL_gridded.nc"
    anom_file = paths.gridded / "GLORYS_CL_anom.nc"
    clim_file = paths.gridded / "GLORYS_CL_clim.nc"
    r_file = paths.r_input / "GLORYS_CL_masked.txt"

    # GLORYS_CL.ipynb: observations define the analysis box, final grid, and co-location mask.
    ds_obs_raw = xr.open_dataset(paths.data / "CORA_MEOP_ARGO_2026.nc")
    ds_obs_raw = rename_obs_coords(ds_obs_raw)
    ds_obs_grid = build_observation_grid(paths.root, nb_bins=nb_bins)

    if not force and gridded_file.exists() and anom_file.exists() and clim_file.exists():
        with xr.open_dataset(gridded_file) as ds_existing:
            grid_ok = same_grid_as_observations(ds_existing, ds_obs_grid)
        if grid_ok:
            with xr.open_dataset(anom_file) as ds_anom:
                write_r_input(ds_anom, r_file)
            print("Reused existing GLORYS_CL NetCDF products.")
            return
        print("Existing GLORYS_CL grid differs from observation grid; recomputing GLORYS_CL products.")

    # GLORYS_CL.ipynb: open GLORYS and select the observation domain.
    ds_g = xr.open_dataset(paths.data / "GLORYS_2026.nc")
    ds_g = ds_g.sel(
        latitude=slice(ds_obs_raw.latitude.min() - 0.5, ds_obs_raw.latitude.max() + 0.5),
        longitude=slice(ds_obs_raw.longitude.min() - 0.5, ds_obs_raw.longitude.max() + 0.5),
    )

    # GLORYS_CL.ipynb: density-threshold MLD using the common study threshold.
    mld = compute_glorys_mld(ds_g, density_threshold=0.03, block_size=block_size)

    # GLORYS_CL.ipynb: bin/interpolate to the observation grid, then keep only observed cells.
    ds_cl = grid_glorys_mld(mld, ds_obs_grid, nb_bins=nb_bins)
    ds_cl["mld"] = ds_cl.mld.where(~np.isnan(ds_obs_grid.mld))
    ds_cl.to_netcdf(gridded_file)
    print(f"Wrote {gridded_file}")

    # GLORYS_CL.ipynb: mask Kerguelen, remove seasonal cycle, write R input.
    mask = kerguelen_mask(ds_cl)
    ds_anom = write_anomaly_products(ds_cl, mask, anom_file, clim_file)
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
    process_glorys_cl(
        project_root=args.project_root,
        force=args.force,
        nb_bins=args.nb_bins,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
