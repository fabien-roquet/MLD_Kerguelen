#!/usr/bin/env python3
"""Appendix Figure A5: Figure 7 trends within the GLORYS cp1 deep region north of the SAF."""

from __future__ import annotations

import argparse

from appendix_region_trends import plot_region_trends, region_a_mask
from figure_common import parse_project_root_arg


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plot_region_trends(
        args.project_root,
        region_a_mask,
        "Deep north-of-SAF cp1",
        "Figure_A5_deep_north_saf_trends.png",
    )


if __name__ == "__main__":
    main()
