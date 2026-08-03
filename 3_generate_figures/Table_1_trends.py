#!/usr/bin/env python3
"""Generate the manuscript trend table from the reconstructed products."""

from __future__ import annotations

import argparse

import numpy as np

from figure_common import load_fpca, parse_project_root_arg, paths
from trend_common import ar1_gls_trend, format_number, seasonal_domain_series


N_RECONSTRUCTION_MODES = 50


def trend_stats(ds_in, months):
    ts_y = seasonal_domain_series(ds_in, months)
    return ar1_gls_trend(ts_y["season_year"].values.astype(float), ts_y.values)


def fmt_cell(trend) -> str:
    return (
        rf"${format_number(trend.slope)}$ "
        rf"[{format_number(trend.ci_low)}, {format_number(trend.ci_high)}] "
        rf"({format_number(trend.pvalue)})"
    )


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    fpca = load_fpca(args.project_root, max_modes=N_RECONSTRUCTION_MODES)
    periods = {"Annual": list(range(1, 13)), "Summer": [1, 2, 3], "Winter": [7, 8, 9]}
    dataset_order = ["GLORYS", "GLORYS_CL", "CMA"]
    ds_map = {"GLORYS": fpca["GLORYS"][0], "GLORYS_CL": fpca["GLORYS_CL"][0], "CMA": fpca["CMA"][0]}

    rows = []
    for period_name, months in periods.items():
        cells = [fmt_cell(trend_stats(ds_map[dataset_name], months)) for dataset_name in dataset_order]
        rows.append(f"{period_name} & " + " & ".join(cells) + r" \\")

    table = "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            r"\caption{Summary of MLD annual and seasonal trend slope coefficients (in $\mathrm{m\,yr^{-1}}$), 95\% confidence intervals, and AR(1)-adjusted $p$-values in parenthesis.}",
            r"\label{table2}",
            r"\small",
            r"\begin{tabular}{c|c|c|c}",
            r"\hline",
            r"Period & GLORYS & $\textrm{GLORYS}_{\textrm{CL}}$ & CMA \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    out_file = paths(args.project_root)["figures"] / "Table_1_trends.tex"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(table)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
