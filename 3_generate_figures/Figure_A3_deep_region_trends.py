#!/usr/bin/env python3
"""Appendix Figure A3: Figure 7 trends within the GLORYS cp1 deep-south region."""

from __future__ import annotations

import argparse

from appendix_region_trends import plot_region_trends, region_b_mask
from figure_common import parse_project_root_arg


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    plot_region_trends(
        args.project_root,
        region_b_mask,
        "Deep-south cp1",
        "Figure_A3_deep_region_trends.png",
    )


if __name__ == "__main__":
    main()
