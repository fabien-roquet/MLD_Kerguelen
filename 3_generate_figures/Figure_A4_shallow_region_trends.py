#!/usr/bin/env python3
"""Appendix Figure A4: Figure 7 trends within the GLORYS cp1 shallow region."""

from __future__ import annotations

import argparse

import xarray as xr

from appendix_region_trends import plot_region_trends
from figure_common import parse_project_root_arg


def shallow_cp1_mask(ds_glorys: xr.Dataset, fronts: xr.Dataset) -> xr.DataArray:
    """GLORYS cp1 shallow lobe delimited by cp1 = 0."""
    _ = fronts
    return ds_glorys["xi1"] < 0.0


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plot_region_trends(
        args.project_root,
        shallow_cp1_mask,
        "Shallow cp1",
        "Figure_A4_shallow_region_trends.png",
    )


if __name__ == "__main__":
    main()
