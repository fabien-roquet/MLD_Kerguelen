"""Shared helpers for figure scripts converted from notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERGUELEN_BOX = (68.25, 70.75, -50.0, -48.0)
G_COLOR = "#0D160B"
CMA_COLOR = "#008DD5"
CL_COLOR = "#CB152B"
BASE_FONT_SIZE = 20
LEGEND_FS = 15
AXIS_LABEL_FS = 20
TICK_LABEL_FS = 16
PANEL_LABEL_FS = 20
COLORBAR_LABEL_FS = 18


def apply_consistent_plot_style(
    base_font_size: int = BASE_FONT_SIZE,
    legend_font_size: int = LEGEND_FS,
    tick_font_size: int = TICK_LABEL_FS,
    label_font_size: int = AXIS_LABEL_FS,
    title_font_size: int = PANEL_LABEL_FS,
) -> None:
    """Apply a single manuscript-style baseline for all plots and allow easy rollback."""
    plt.rcParams.update(
        {
            "font.size": base_font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": label_font_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
            "legend.fontsize": legend_font_size,
            "figure.titlesize": title_font_size,
            "axes.labelpad": 8,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
        }
    )


try:
    import cmocean.cm as cmo
except ModuleNotFoundError:  # pragma: no cover - fallback for partial environments
    from matplotlib import colormaps

    class _CmoFallback:
        deep = colormaps["viridis"]
        matter = colormaps["plasma"]
        balance = colormaps["coolwarm"]
        amp = colormaps["magma"]
        thermal = colormaps["inferno"]

    cmo = _CmoFallback()


def paths(project_root: str | Path = PROJECT_ROOT) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "root": root,
        "data": root / "data",
        "gridded": root / "processed" / "1_gridded_data",
        "figures": root / "processed" / "3_figures",
    }


def save_figure(fig: plt.Figure, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_file}")


def open_gridded(project_root: str | Path, name: str) -> xr.Dataset:
    file = paths(project_root)["gridded"] / name
    if not file.exists():
        raise FileNotFoundError(f"Missing gridded product: {file}")
    return xr.open_dataset(file)


def require_file(file: Path, note: str = "") -> Path:
    if not file.exists():
        suffix = f" {note}" if note else ""
        raise FileNotFoundError(f"Missing required file: {file}.{suffix}")
    return file


def topo_fronts(project_root: str | Path = PROJECT_ROOT) -> tuple[xr.DataArray, xr.Dataset]:
    data_dir = paths(project_root)["data"]
    topo_file = require_file(data_dir / "GEBCO_ker_large.nc")
    fronts_file = require_file(data_dir / "fronts_62985.nc")
    elevation = xr.open_dataset(topo_file).elevation[::10, ::10]
    fronts = xr.open_dataset(fronts_file)
    return elevation, fronts


def add_common_map_layers(ax: plt.Axes, elevation: xr.DataArray, fronts: xr.Dataset, front_color: str = "k") -> None:
    (elevation / elevation).where(elevation > 0).plot(add_colorbar=False, cmap="gist_yarg", ax=ax)
    cs = (-elevation).plot.contour(levels=(500, 1000, 2000), colors=["black"], linewidths=1, ax=ax)
    ax.plot(fronts.LonSAF.where(fronts.LatSAF > -50), fronts.LatSAF.where(fronts.LatSAF > -50), c=front_color, lw=2.5)
    ax.plot(fronts.LonPF, fronts.LatPF, c=front_color, lw=2.5)
    ax.plot(fronts.LonSACCF, fronts.LatSACCF, c=front_color, lw=2.5)
    ax.clabel(cs, inline=True, fmt="%1.0f", fontsize=10)
    ax.set_xticks([60, 65, 70, 75, 80])
    ax.set_yticks([-60, -55, -50, -45, -40])


def kerguelen_mask(ds: xr.Dataset, lon_name: str = "longitude", lat_name: str = "latitude") -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = KERGUELEN_BOX
    return (
        (ds[lon_name] >= lon_min)
        & (ds[lon_name] <= lon_max)
        & (ds[lat_name] >= lat_min)
        & (ds[lat_name] <= lat_max)
    )


def apply_fpca_spatial_mask(ds: xr.Dataset) -> xr.Dataset:
    lon_min, lon_max, lat_min, lat_max = KERGUELEN_BOX
    lon2d, lat2d = xr.broadcast(ds["long"], ds["lat"])
    area_mask = (lon2d >= lon_min) & (lon2d <= lon_max) & (lat2d >= lat_min) & (lat2d <= lat_max)
    out = ds.copy()
    for var in out.data_vars:
        if {"long", "lat"}.issubset(out[var].dims):
            out[var] = out[var].where(~area_mask)
    return out


def load_fpca(project_root: str | Path = PROJECT_ROOT, max_modes: int | None = None):
    from create_rec_datasets import r_analysis_df

    ds_cma, df_cma, df_xiest_cma, df_phi_cma, df_grid_cma, df_mu_cma, df_lambda_cma = r_analysis_df(
        "CMA_masked", max_modes=max_modes, project_root=project_root
    )
    ds_g, df_g, df_xiest_g, df_phi_g, df_grid_g, df_mu_g, df_lambda_g = r_analysis_df(
        "GLORYS_masked", dense=True, max_modes=max_modes, project_root=project_root
    )
    ds_cl, df_cl, df_xiest_cl, df_phi_cl, df_grid_cl, df_mu_cl, df_lambda_cl = r_analysis_df(
        "GLORYS_CL_masked", max_modes=max_modes, project_root=project_root
    )
    return {
        "CMA": (apply_fpca_spatial_mask(ds_cma), df_cma, df_xiest_cma, df_phi_cma, df_grid_cma, df_mu_cma, df_lambda_cma),
        "GLORYS": (apply_fpca_spatial_mask(ds_g), df_g, df_xiest_g, df_phi_g, df_grid_g, df_mu_g, df_lambda_g),
        "GLORYS_CL": (apply_fpca_spatial_mask(ds_cl), df_cl, df_xiest_cl, df_phi_cl, df_grid_cl, df_mu_cl, df_lambda_cl),
    }


def align_original_anomaly(project_root: str | Path, filename: str, template: xr.Dataset) -> xr.Dataset:
    ds = open_gridded(project_root, filename)
    ds = ds.where(ds.time < pd.to_datetime("2024-01-01"), drop=True)
    ds = ds.rename({"longitude": "long", "latitude": "lat"})
    ds["time"] = template.time
    ds["lat"] = template.lat
    ds["long"] = template.long
    return ds


def parse_project_root_arg(parser):
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser
