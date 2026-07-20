#!/usr/bin/env python3
"""Appendix Figure A3: Figure 7 trends within the GLORYS cp1 deep-south region."""

from __future__ import annotations

import argparse

import xarray as xr

from appendix_region_trends import latitude_grid, plot_region_trends
from figure_common import parse_project_root_arg


def deep_south_cp1_mask(ds_glorys: xr.Dataset, fronts: xr.Dataset) -> xr.DataArray:
    """Region south of 50S inside the GLORYS cp1 deep lobe delimited by cp1 = 0."""
    _ = fronts
    return (latitude_grid(ds_glorys) < -50.0) & (ds_glorys["xi1"] > 0.0)


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plot_region_trends(
        args.project_root,
        deep_south_cp1_mask,
        "Deep-south cp1",
        "Figure_A3_deep_region_trends.png",
    )


if __name__ == "__main__":
    main()
