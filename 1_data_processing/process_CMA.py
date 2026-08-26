#!/usr/bin/env python3
"""Python script version of 1_data_processing/CMA.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from processing_common import (
    NB_BINS,
    PROJECT_ROOT,
    build_observation_grid,
    kerguelen_mask,
    project_paths,
    same_grid_as_observations,
    write_anomaly_products,
    write_r_input,
)


def process_cma(project_root: str | Path = PROJECT_ROOT, force: bool = False, nb_bins: int = NB_BINS) -> None:
    paths = project_paths(project_root)
    gridded_file = paths.gridded / "CMA_gridded.nc"
    anom_file = paths.gridded / "CMA_anom.nc"
    clim_file = paths.gridded / "CMA_clim.nc"
    r_file = paths.r_input / "CMA_masked.txt"

    # CMA.ipynb: open CORA/MEOP/ARGO observations and bin onto the 39 x 39 grid.
    ds = build_observation_grid(paths.root, nb_bins=nb_bins)

    if not force and gridded_file.exists() and anom_file.exists() and clim_file.exists():
        with xr.open_dataset(gridded_file) as ds_existing:
            grid_ok = same_grid_as_observations(ds_existing, ds)
        if grid_ok:
            with xr.open_dataset(anom_file) as ds_anom:
                write_r_input(ds_anom, r_file)
            print("Reused existing CMA NetCDF products.")
            return
        print("Existing CMA grid differs from observation grid; recomputing CMA products.")

    # CMA.ipynb: mask the Kerguelen island box.
    mask = kerguelen_mask(ds)
    ds_masked = ds.where(~mask)

    # CMA.ipynb: remove the mean seasonal cycle and write anomaly/climatology products.
    ds_anom = write_anomaly_products(ds_masked, mask, anom_file, clim_file)

    # CMA.ipynb: write masked gridded MLD for maps, and anomalies for R input.
    ds_masked.to_netcdf(gridded_file)
    print(f"Wrote {gridded_file}")
    write_r_input(ds_anom, r_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true", help="Recompute even if processed outputs exist.")
    parser.add_argument("--nb-bins", type=int, default=NB_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_cma(project_root=args.project_root, force=args.force, nb_bins=args.nb_bins)


if __name__ == "__main__":
    main()
