#!/usr/bin/env python3
"""Generate Figure 9 from the GLORYS section product."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from figure_common import cmo, load_fpca, parse_project_root_arg, paths, require_file, save_figure


SECTION_START = (72.0, -52.5)
SECTION_END = (79.0, -47.0)


def zero_crossings_1d(x: np.ndarray, y: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.array([])

    idx = np.where(np.signbit(y[:-1]) != np.signbit(y[1:]))[0]
    den = y[idx + 1] - y[idx]
    ok = ~np.isclose(den, 0.0, atol=atol)
    x_interp = x[idx[ok]] - y[idx[ok]] * (x[idx[ok] + 1] - x[idx[ok]]) / den[ok]
    x_exact = x[np.isclose(y, 0.0, atol=atol)]
    return np.unique(np.concatenate([x_interp, x_exact]))


def interp_along_section(
    ds: xr.Dataset | xr.DataArray,
    start: tuple[float, float],
    end: tuple[float, float],
    n_points: int,
    lon_name: str,
    lat_name: str,
) -> xr.Dataset | xr.DataArray:
    lon = np.linspace(start[0], end[0], n_points)
    lat = np.linspace(start[1], end[1], n_points)
    out = ds.interp(
        {
            lon_name: xr.DataArray(lon, dims="section"),
            lat_name: xr.DataArray(lat, dims="section"),
        }
    )
    return out.assign_coords(longitude=("section", lon), latitude=("section", lat))


def nearest_front_longitude(ds_front: xr.Dataset, section_lon: np.ndarray, section_lat: np.ndarray) -> float:
    pf_lon = ds_front["LonPF"].values
    pf_lat = ds_front["LatPF"].values
    valid = np.isfinite(pf_lon) & np.isfinite(pf_lat)
    pf_lon = pf_lon[valid]
    pf_lat = pf_lat[valid]
    dist2 = (pf_lon[:, None] - section_lon[None, :]) ** 2 + (pf_lat[:, None] - section_lat[None, :]) ** 2
    i_pf, _ = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(pf_lon[i_pf])


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-points", type=int, default=240, help="Number of points along the plotted section.")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 15})

    project_paths = paths(args.project_root)
    section_file = require_file(
        project_paths["data"] / "GLORYS_1000m_section_timemean.nc",
        "Run 1_data_processing/process_GLORYS_section.py once while data/GLORYS_1000m_section_plot.nc is available.",
    )
    topo_file = require_file(project_paths["data"] / "GEBCO_ker_large.nc")
    fronts_file = require_file(project_paths["data"] / "fronts_62985.nc")

    ds_section_mean = xr.open_dataset(section_file)
    ds_section = interp_along_section(
        ds_section_mean[["thetao", "so", "mld"]],
        SECTION_START,
        SECTION_END,
        args.n_points,
        lon_name="longitude",
        lat_name="latitude",
    )

    fpca = load_fpca(args.project_root)
    ds_g = fpca["GLORYS"][0].mean("time").rename({"long": "longitude", "lat": "latitude"})
    section_lon = ds_section["longitude"].values
    section_lat = ds_section["latitude"].values
    cross_coef = interp_along_section(
        ds_g[["mld", "xi_phi1"]],
        SECTION_START,
        SECTION_END,
        args.n_points,
        lon_name="longitude",
        lat_name="latitude",
    )
    lon_intersection = zero_crossings_1d(section_lon, cross_coef["xi_phi1"].values)

    topo = -xr.open_dataset(topo_file).elevation[::10, ::10]
    topo_section = interp_along_section(
        topo,
        SECTION_START,
        SECTION_END,
        args.n_points,
        lon_name="lon",
        lat_name="lat",
    )
    ds_front = xr.open_dataset(fronts_file)
    pf_lon_section = nearest_front_longitude(ds_front, section_lon, section_lat)

    theta_mean = ds_section["thetao"]
    mld_mean = ds_section["mld"]
    depth_max = float(ds_section["depth"].max())

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    pcm = ax.pcolormesh(
        section_lon,
        ds_section["depth"],
        theta_mean.transpose("depth", "section"),
        shading="auto",
        cmap=cmo.thermal,
        vmin=0,
        vmax=5,
    )
    contours = ax.contour(
        section_lon,
        ds_section["depth"],
        theta_mean.transpose("depth", "section"),
        colors="black",
        levels=np.linspace(0, 4.8, 13),
        linewidths=0.8,
    )
    ax.clabel(contours, inline=True, fontsize=10)

    ax.plot(section_lon, mld_mean, color="lightgrey", lw=2)
    bathy = topo_section.values
    ax.fill_between(section_lon, bathy, depth_max, where=np.isfinite(bathy), color="0.65")
    ax.plot(section_lon, bathy, color="black", lw=1)

    if lon_intersection.size:
        ax.vlines(lon_intersection, depth_max, 0, ls="--", color="k", lw=2, label="$c_{p1} = 0$")
    ax.vlines(pf_lon_section, depth_max, 0, ls="-.", color="k", lw=2, label="PF")
    ax.set_ylim(depth_max, 0)
    ax.set_xlim(section_lon.min(), section_lon.max())
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Longitude [˚E]")
    ax.set_title("")
    ax.legend()
    fig.colorbar(pcm, cax=cax, label=r"$\theta \; [^\circ C]$")
    save_figure(fig, project_paths["figures"] / "Figure_9_sections.png")


if __name__ == "__main__":
    main()
