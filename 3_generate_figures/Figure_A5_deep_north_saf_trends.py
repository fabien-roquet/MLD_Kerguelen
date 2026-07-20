#!/usr/bin/env python3
"""Appendix Figure A5: Figure 7 trends within the GLORYS cp1 deep region north of the SAF."""

from __future__ import annotations

import argparse

import xarray as xr

from appendix_region_trends import latitude_grid, plot_region_trends, saf_latitude_grid
from figure_common import parse_project_root_arg


def deep_north_saf_cp1_mask(ds_glorys: xr.Dataset, fronts: xr.Dataset) -> xr.DataArray:
    """Region north of the SAF inside the GLORYS cp1 deep lobe delimited by cp1 = 0."""
    return (latitude_grid(ds_glorys) > saf_latitude_grid(ds_glorys, fronts)) & (ds_glorys["xi1"] > 0.0)


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plot_region_trends(
        args.project_root,
        deep_north_saf_cp1_mask,
        "Deep north-of-SAF cp1",
        "Figure_A5_deep_north_saf_trends.png",
    )


if __name__ == "__main__":
    main()
