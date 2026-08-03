"""Shared seasonal-series and AR(1)-aware trend helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy import stats


@dataclass(frozen=True)
class TrendResult:
    intercept: float
    slope: float
    ci_low: float
    ci_high: float
    pvalue: float
    phi: float
    n: int
    x_mean: float

    @property
    def significant(self) -> bool:
        return np.isfinite(self.ci_low) and np.isfinite(self.ci_high) and (self.ci_low > 0 or self.ci_high < 0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * (np.asarray(x, dtype=float) - self.x_mean)


def empty_trend() -> TrendResult:
    return TrendResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)


def seasonal_mean(da: xr.DataArray, months: list[int]) -> xr.DataArray:
    sub = da.where(da["time"].dt.month.isin(months), drop=True)
    if set(months) == {12, 1, 2}:
        season_year = xr.where(sub["time"].dt.month == 12, sub["time"].dt.year + 1, sub["time"].dt.year)
    else:
        season_year = sub["time"].dt.year
    return sub.groupby(season_year.rename("season_year")).mean("time", skipna=True)


def seasonal_domain_series(ds_in: xr.Dataset, months: list[int], mask: xr.DataArray | None = None) -> xr.DataArray:
    da = ds_in["mld"]
    if mask is not None:
        da = da.where(mask)
    ts = da.mean(dim=("long", "lat"), skipna=True)
    return seasonal_mean(ts, months)


def ar1_gls_trend(x: np.ndarray, y: np.ndarray, min_points: int = 8, alpha: float = 0.05) -> TrendResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(y)
    if n < min_points:
        return empty_trend()

    x_mean = float(x.mean())
    xc = x - x_mean
    design = np.column_stack((np.ones(n), xc))
    try:
        beta_ols = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return empty_trend()

    resid = y - design @ beta_ols
    denom = float(np.dot(resid[:-1], resid[:-1]))
    if denom > 0:
        phi = float(np.dot(resid[1:], resid[:-1]) / denom)
    else:
        phi = 0.0
    phi = float(np.clip(phi, -0.95, 0.95))

    index = np.arange(n)
    cov = phi ** np.abs(np.subtract.outer(index, index))
    try:
        cov_inv_design = np.linalg.solve(cov, design)
        cov_inv_y = np.linalg.solve(cov, y)
        xt_vinv_x = design.T @ cov_inv_design
        beta_cov_unit = np.linalg.inv(xt_vinv_x)
        beta = beta_cov_unit @ (design.T @ cov_inv_y)
    except np.linalg.LinAlgError:
        phi = 0.0
        beta = beta_ols
        beta_cov_unit = np.linalg.inv(design.T @ design)
        cov_inv_y = y
        cov_inv_design = design

    resid_gls = y - design @ beta
    dof = n - design.shape[1]
    if dof <= 0:
        return empty_trend()
    sigma2 = float(resid_gls.T @ (cov_inv_y - cov_inv_design @ beta) / dof)
    if not np.isfinite(sigma2) or sigma2 < 0:
        sigma2 = 0.0
    beta_cov = beta_cov_unit * sigma2
    slope_se = float(np.sqrt(max(beta_cov[1, 1], 0.0)))
    slope = float(beta[1])
    if slope_se == 0:
        pvalue = 0.0 if slope != 0 else 1.0
        margin = 0.0
    else:
        t_stat = slope / slope_se
        pvalue = float(2 * stats.t.sf(abs(t_stat), dof))
        margin = float(stats.t.ppf(1 - alpha / 2, dof) * slope_se)

    return TrendResult(
        intercept=float(beta[0]),
        slope=slope,
        ci_low=slope - margin,
        ci_high=slope + margin,
        pvalue=pvalue,
        phi=phi,
        n=n,
        x_mean=x_mean,
    )


def ar1_trend_values(y: np.ndarray, x: np.ndarray, min_points: int = 8) -> np.ndarray:
    trend = ar1_gls_trend(x, y, min_points=min_points)
    return np.array(
        [
            trend.slope,
            trend.ci_low,
            trend.ci_high,
            trend.pvalue,
            trend.phi,
            float(trend.n),
            float(trend.significant),
        ],
        dtype=float,
    )


def seasonal_trend_map(ds_in: xr.Dataset, months: list[int], min_years: int = 8) -> xr.Dataset:
    da_y = seasonal_mean(ds_in["mld"], months)
    x = da_y["season_year"].astype(float)
    result = xr.apply_ufunc(
        ar1_trend_values,
        da_y,
        x,
        input_core_dims=[["season_year"], ["season_year"]],
        output_core_dims=[["trend_stat"]],
        vectorize=True,
        dask="allowed",
        kwargs={"min_points": min_years},
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"trend_stat": 7}},
    )
    result = result.assign_coords(trend_stat=["slope", "ci_low", "ci_high", "pvalue", "phi", "n", "significant"])
    return result.to_dataset(dim="trend_stat")


def format_number(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def format_trend_ci(trend: TrendResult) -> str:
    return f"{format_number(trend.slope)} [{format_number(trend.ci_low)}, {format_number(trend.ci_high)}]"
