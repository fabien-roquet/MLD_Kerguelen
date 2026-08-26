#!/usr/bin/env python3
"""Python script version of Figure_10_KERFIX.ipynb."""

from __future__ import annotations

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_common import CMA_COLOR, apply_consistent_plot_style, open_gridded, parse_project_root_arg, paths, require_file, save_figure
from create_rec_datasets import r_analysis_df


def main() -> None:
    parser = parse_project_root_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    apply_consistent_plot_style()

    try:
        import gsw
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Figure 10 requires gsw. Install project dependencies with uv sync.") from exc

    df_ker = pd.read_csv(require_file(paths(args.project_root)["data"] / "kerfix.csv"), sep=";")
    profile_id = df_ker["profileID"].astype(str)
    date_prefix = profile_id.str[:2].apply(lambda value: "19" + value if int(value) > 50 else "20" + value)
    df_ker["TIME"] = pd.to_datetime(date_prefix + profile_id.str[2:], format="%Y%m")
    df_ker["DEPTH"] = -gsw.conversions.z_from_p(df_ker["PRES"], lat=-50.6667, geo_strf_dyn_height=0, sea_surface_geopotential=0)
    df_ker["SA"] = gsw.SA_from_SP(df_ker["PSAL"], df_ker["PRES"], lon=68.4167, lat=-50.6667)
    df_ker["CT"] = gsw.CT_from_t(df_ker["SA"], df_ker["TEMP"], df_ker["PRES"])
    df_ker["sigma0"] = gsw.density.sigma0(df_ker["SA"], df_ker["CT"])
    ds_ker = df_ker.reset_index(drop=True).set_index(["TIME", "DEPTH"]).to_xarray()

    sigma0_10m = ds_ker.sel(DEPTH=10, method="nearest").sigma0
    mask = ((ds_ker.sigma0 - sigma0_10m) >= 0.03) & (ds_ker.DEPTH >= 10)
    has = mask.any(dim="DEPTH")
    mld = ds_ker.DEPTH.isel(DEPTH=mask.argmax(dim="DEPTH")).where(has)
    ds_ker = ds_ker.assign({"mld": (("TIME"), mld.data)})

    lon_ker = 68.4167
    lat_ker = -50.6667
    ds_G = open_gridded(args.project_root, "GLORYS_gridded.nc").sel(longitude=lon_ker, latitude=lat_ker, method="nearest")
    ds_CMA_rec = r_analysis_df("CMA_masked", project_root=args.project_root)[0]
    ds_CMA_clim = open_gridded(args.project_root, "CMA_clim.nc").rename({"longitude": "long", "latitude": "lat"})
    ds_CMA_obs = open_gridded(args.project_root, "CMA_gridded.nc").sel(longitude=lon_ker, latitude=lat_ker, method="nearest")
    cma_anom = ds_CMA_rec["mld"].sel(long=lon_ker, lat=lat_ker, method="nearest")
    cma_clim = ds_CMA_clim["mld"].sel(long=lon_ker, lat=lat_ker, method="nearest")
    cma = cma_anom.groupby("time.month") + cma_clim

    ker = ds_ker["mld"].dropna(dim="TIME")
    glo = ds_G["mld"].dropna(dim="time")
    cma_obs = ds_CMA_obs["mld"].dropna(dim="time")
    ker_xlim = (ker["TIME"].min().values - np.timedelta64(30, "D"), ker["TIME"].max().values + np.timedelta64(30, "D"))
    glo_xlim = (glo["time"].min().values - np.timedelta64(90, "D"), glo["time"].max().values + np.timedelta64(90, "D"))
    ker_span = (ker_xlim[1] - ker_xlim[0]) / np.timedelta64(1, "D")
    glo_span = (glo_xlim[1] - glo_xlim[0]) / np.timedelta64(1, "D")

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 10), gridspec_kw={"width_ratios": [ker_span, glo_span], "wspace": 0.05})
    line_ker = ax1.plot(ker["TIME"].values, ker.values, color="#FF7733", lw=1.5, ms=4, label="KERFIX")[0]
    line_glo = ax2.plot(glo["time"].values, glo.values, color="black", lw=1.5, label="GLORYS")[0]
    line_cma = ax2.plot(cma["time"].values, cma.values, color=CMA_COLOR, lw=1.5, label="CMA reconstruction")[0]
    points_cma = ax2.scatter(
        cma_obs["time"].values,
        cma_obs.values,
        color=CMA_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=70,
        alpha=0.95,
        label="CMA data",
        zorder=5,
    )
    ax1.set_xlim(*ker_xlim)
    ax2.set_xlim(*glo_xlim)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(labelleft=False, left=False)

    fig.canvas.draw()
    pix = 10
    dx1, dy1 = pix / ax1.bbox.width, pix / ax1.bbox.height
    dx2, dy2 = pix / ax2.bbox.width, pix / ax2.bbox.height
    kwargs = dict(color="k", clip_on=False)
    ax1.plot((1 - dx1, 1 + dx1), (-dy1, +dy1), transform=ax1.transAxes, **kwargs)
    ax1.plot((1 - dx1, 1 + dx1), (1 - dy1, 1 + dy1), transform=ax1.transAxes, **kwargs)
    ax2.plot((-dx2, +dx2), (-dy2, +dy2), transform=ax2.transAxes, **kwargs)
    ax2.plot((-dx2, +dx2), (1 - dy2, 1 + dy2), transform=ax2.transAxes, **kwargs)

    ker_mean, ker_std = float(ker.mean().values), float(ker.std().values)
    glo_mean, glo_std = float(glo.mean().values), float(glo.std().values)
    cma_mean, cma_std = float(cma.mean().values), float(cma.std().values)
    mean_ker = ax1.axhline(ker_mean, color="#FF7733", ls="--", lw=2, alpha=0.9, label=f"Mean = {ker_mean:.1f} +/- {ker_std:.1f} m")
    mean_glo = ax2.axhline(glo_mean, color="dimgray", ls="--", lw=2, alpha=0.9, label=f"Mean = {glo_mean:.1f} +/- {glo_std:.1f} m")
    mean_cma = ax2.axhline(cma_mean, color=CMA_COLOR, ls="--", lw=2, alpha=0.9, label=f"Mean = {cma_mean:.1f} +/- {cma_std:.1f} m")
    ax1.set_ylabel("MLD [m]")
    fig.legend(
        handles=[line_ker, mean_ker, line_glo, mean_glo, line_cma, mean_cma, points_cma],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.87),
    )
    save_figure(fig, paths(args.project_root)["figures"] / "Figure_10_KERFIX_GLORYS_timeseries.png")


if __name__ == "__main__":
    main()
