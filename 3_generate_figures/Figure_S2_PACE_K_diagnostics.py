#!/usr/bin/env python3
"""Supporting Figure S2 and Table S2: PACE retained-mode diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from figure_common import (
    CL_COLOR,
    CMA_COLOR,
    G_COLOR,
    align_original_anomaly,
    apply_consistent_plot_style,
    load_fpca,
    parse_project_root_arg,
    paths,
    require_file,
    save_figure,
)
from trend_common import format_number


N_RECONSTRUCTION_MODES = 50
K_VALUES = [1, 2, 3, 5, 10, 20, 30, 40, 50, 75]
DATASET_META = {
    "GLORYS": ("GLORYS", "GLORYS_masked_dense", G_COLOR),
    "GLORYS_CL": (r"GLORYS$_{\mathregular{CL}}$", "GLORYS_CL_masked", CL_COLOR),
    "CMA": ("CMA", "CMA_masked", CMA_COLOR),
}


def lambda_file(project_root: str | Path, fpca_dir: str) -> Path:
    return require_file(Path(project_root) / "processed" / "2_fPCA" / fpca_dir / "PCA_LAMBDA.csv")


def read_lambdas(project_root: str | Path) -> dict[str, pd.Series]:
    out = {}
    for dataset_name, (_, fpca_dir, _) in DATASET_META.items():
        data = pd.read_csv(lambda_file(project_root, fpca_dir)).drop(columns="Unnamed: 0", errors="ignore")
        out[dataset_name] = data.iloc[:, 0].astype(float)
    return out


def cumulative_variance(lambdas: pd.Series) -> pd.Series:
    return lambdas.cumsum() / lambdas.sum()


def domain_rmse(ds_pred: xr.Dataset, ds_ref: xr.Dataset, mask: xr.DataArray) -> float:
    err = (ds_pred["mld"] - ds_ref["mld"]).where(mask)
    return float(np.sqrt((err**2).mean(skipna=True)))


def compute_rmse_by_k(project_root: str | Path, original_map: dict[str, xr.Dataset], evaluation_mask: xr.DataArray) -> pd.DataFrame:
    rows = []
    for k_value in K_VALUES:
        fpca = load_fpca(project_root, max_modes=k_value)
        ds_map = {"GLORYS": fpca["GLORYS"][0], "GLORYS_CL": fpca["GLORYS_CL"][0], "CMA": fpca["CMA"][0]}
        for dataset_name, ds_in in ds_map.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "K": k_value,
                    "rmse": domain_rmse(ds_in, original_map[dataset_name], evaluation_mask),
                }
            )
    return pd.DataFrame(rows)


def write_pace_parameter_table(out_file: Path, fve_at_50: dict[str, float], rmse_at_50: dict[str, float]) -> None:
    rows = [
        ("PACE input type", "DenseWithMV", "Sparse"),
        ("Mean/covariance estimator", "cross-sectional", "smooth"),
        ("Mean bandwidth", "not used", "0.05"),
        ("Covariance bandwidth", "not used", "0.25"),
        ("Kernel", "fdapace dense default", "epan"),
        ("Regular grid size", "204", "204"),
        ("Binned data", "OFF", "OFF"),
        ("Mode selection", "FVE", "FVE"),
        ("FVE threshold in fPCA export", "1", "1"),
        ("Score estimation", "IN", "CE"),
        ("Maximum exported modes", "min(nmonth - 2, ncell - 1)", "100"),
        ("Measurement-error option", "FALSE", "fdapace default TRUE"),
        ("Correlation estimator", "not used", "fdapace default vanilla"),
        (
            "Retained variance at K=50",
            f"{format_number(100 * fve_at_50['GLORYS'])}\\%",
            f"{format_number(100 * fve_at_50['GLORYS_CL'])}\\% (GLORYS$_{{CL}}$); "
            f"{format_number(100 * fve_at_50['CMA'])}\\% (CMA)",
        ),
        (
            "Domain RMSE at K=50",
            f"{format_number(rmse_at_50['GLORYS'])} m",
            f"{format_number(rmse_at_50['GLORYS_CL'])} m (GLORYS$_{{CL}}$); "
            f"{format_number(rmse_at_50['CMA'])} m (CMA)",
        ),
    ]
    body = "\n".join(f"{setting} & {dense} & {sparse} \\\\" for setting, dense, sparse in rows)
    table = "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            r"\caption{PACE settings used for the dense GLORYS reconstruction and the sparse profile-like reconstructions. Settings marked as fdapace defaults are not explicitly set in the fPCA scripts.}",
            r"\label{tableS2}",
            r"\small",
            r"\begin{tabular}{l|l|l}",
            r"\hline",
            r"Setting & Dense GLORYS & Sparse GLORYS$_{\textrm{CL}}$/CMA \\",
            r"\hline",
            body,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    out_file.write_text(table)
    print(f"Wrote {out_file}")


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    apply_consistent_plot_style()

    lambdas = read_lambdas(args.project_root)
    fpca_50 = load_fpca(args.project_root, max_modes=N_RECONSTRUCTION_MODES)
    template = fpca_50["GLORYS"][0]
    original_map = {
        "GLORYS": align_original_anomaly(args.project_root, "GLORYS_anom.nc", template),
        "GLORYS_CL": align_original_anomaly(args.project_root, "GLORYS_CL_anom.nc", template),
        "CMA": align_original_anomaly(args.project_root, "CMA_anom.nc", template),
    }
    evaluation_mask = original_map["GLORYS_CL"]["mld"].notnull()
    rmse_by_k = compute_rmse_by_k(args.project_root, original_map, evaluation_mask)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), tight_layout=True)
    ax_var, ax_rmse = axes

    fve_at_50 = {}
    for dataset_name, (label, _, color) in DATASET_META.items():
        fve = cumulative_variance(lambdas[dataset_name])
        fve_at_50[dataset_name] = float(fve.iloc[N_RECONSTRUCTION_MODES - 1])
        modes = np.arange(1, len(fve) + 1)
        ax_var.plot(modes, 100 * fve.to_numpy(), color=color, lw=2, label=label)
    ax_var.axvline(N_RECONSTRUCTION_MODES, color="0.25", lw=1.2, linestyle="--")
    ax_var.set_xlabel("Number of retained modes K")
    ax_var.set_ylabel("Cumulative explained variance [%]")
    ax_var.set_xlim(1, max(K_VALUES))
    ax_var.set_ylim(0, 101)
    ax_var.grid(alpha=0.3)
    ax_var.text(0.02, 0.96, "(a)", transform=ax_var.transAxes, ha="left", va="top", fontweight="bold")
    ax_var.legend(loc="lower right", fontsize=15)

    rmse_at_50 = {}
    for dataset_name, (label, _, color) in DATASET_META.items():
        sub = rmse_by_k.loc[rmse_by_k["dataset"].eq(dataset_name)]
        rmse_at_50[dataset_name] = float(sub.loc[sub["K"].eq(N_RECONSTRUCTION_MODES), "rmse"].iloc[0])
        ax_rmse.plot(sub["K"], sub["rmse"], marker="o", color=color, lw=2, label=label)
    ax_rmse.axvline(N_RECONSTRUCTION_MODES, color="0.25", lw=1.2, linestyle="--")
    ax_rmse.set_xlabel("Number of retained modes K")
    ax_rmse.set_ylabel("Domain RMSE [m]")
    ax_rmse.set_xlim(1, max(K_VALUES))
    ax_rmse.grid(alpha=0.3)
    ax_rmse.text(0.02, 0.96, "(b)", transform=ax_rmse.transAxes, ha="left", va="top", fontweight="bold")
    ax_rmse.legend(loc="best", fontsize=15)

    out_dir = paths(args.project_root)["figures"]
    write_pace_parameter_table(out_dir / "Table_S2_PACE_parameters.tex", fve_at_50, rmse_at_50)
    rmse_by_k.to_csv(out_dir / "Table_S2_PACE_K_RMSE_values.csv", index=False)
    print(f"Wrote {out_dir / 'Table_S2_PACE_K_RMSE_values.csv'}")
    save_figure(fig, out_dir / "Figure_S2_PACE_K_diagnostics.png")


if __name__ == "__main__":
    main()
