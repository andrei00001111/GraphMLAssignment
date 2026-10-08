"""Plotting helpers for the wind-following GNN project (Stage A and beyond)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

C.set_plot_style()

#: re-exported so notebooks can use ``V.save_figure`` next to the plotting helpers
from .config import save_figure  # noqa: E402,F401

import matplotlib.pyplot as plt  # noqa: E402  (after style/backend setup)


# --------------------------------------------------------------------------- #
# Farm layout
# --------------------------------------------------------------------------- #

def plot_layout(loc: pd.DataFrame, values: np.ndarray | None = None, ax=None,
                title: str = "Wind farm layout", cbar_label: str = "",
                cmap: str = "viridis", annotate_ids: bool = False):
    """Scatter the turbines in farm coordinates, optionally coloured by a value."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 7.2))
    xy = loc.loc[:, ["x", "y"]].to_numpy()
    if values is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=26, c="#2b6cb0", edgecolor="white", linewidth=0.4)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=30, c=values, cmap=cmap,
                        edgecolor="white", linewidth=0.3)
        cb = ax.figure.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label(cbar_label)
    if annotate_ids:
        for tid, (x, y) in zip(loc.index.to_numpy(), xy):
            if tid % 10 == 1:
                ax.annotate(str(tid), (x, y), fontsize=6, alpha=0.7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return ax


# --------------------------------------------------------------------------- #
# Time series / correlations
# --------------------------------------------------------------------------- #

def plot_power_curve(df: pd.DataFrame, turbine: int | None = None, ax=None,
                     max_points: int = 20000, title: str | None = None):
    """Scatter of wind speed against active power (10-minute records)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.0))
    sub = df if turbine is None else df[df["TurbID"] == turbine]
    sub = sub[sub["Wspd"].notna() & sub["Patv"].notna()]
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)
    ax.scatter(sub["Wspd"], sub["Patv"], s=3, alpha=0.25, color="#2b6cb0", linewidths=0)
    ax.axhline(C.RATED_POWER_KW, color="crimson", ls="--", lw=1,
               label=f"rated {C.RATED_POWER_KW:.0f} kW")
    ax.set_xlabel("wind speed [m/s]")
    ax.set_ylabel("active power [kW]")
    ax.set_title(title or ("Power curve" + (f" - turbine {turbine}" if turbine else " - all records")))
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_correlation_matrix(corr: np.ndarray, ax=None, title: str = "Inter-turbine power correlation",
                            vmin: float | None = None, vmax: float = 1.0):
    """Heatmap of a turbine x turbine correlation matrix, ordered by turbine ID."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(corr, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Pearson r")
    ax.set_xlabel("turbine ID")
    ax.set_ylabel("turbine ID")
    ax.set_title(title)
    ax.grid(False)
    return ax


def plot_correlation_vs_distance(dist: np.ndarray, corr: np.ndarray, ax=None,
                                 title: str = "Correlation vs. distance",
                                 max_points: int = 20000, seed: int = 0):
    """Scatter of pairwise distance against pairwise correlation, with a linear fit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.0))
    iu = np.triu_indices_from(dist, k=1)
    d, r = dist[iu], corr[iu]
    ok = np.isfinite(d) & np.isfinite(r)
    d, r = d[ok], r[ok]
    rng = np.random.default_rng(seed)
    if len(d) > max_points:
        sel = rng.choice(len(d), max_points, replace=False)
        d, r = d[sel], r[sel]
    ax.scatter(d / 1000.0, r, s=3, alpha=0.2, color="#4a5568", linewidths=0)
    slope, intercept = np.polyfit(d, r, 1)
    xs = np.linspace(d.min(), d.max(), 100)
    r2 = np.corrcoef(d, r)[0, 1] ** 2
    ax.plot(xs / 1000.0, slope * xs + intercept, color="crimson", lw=1.6,
            label=f"linear fit, $R^2$ = {r2:.3f}")
    ax.axhline(np.mean(r), color="#2b6cb0", ls="--", lw=1.2, label=f"mean r = {np.mean(r):.3f}")
    ax.set_xlabel("distance [km]")
    ax.set_ylabel("power correlation r")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------- #
# Wind direction
# --------------------------------------------------------------------------- #

def plot_wind_rose(wind_from_deg: np.ndarray, ax=None, bins: int = 36,
                   title: str = "Wind rose (direction the wind comes FROM)"):
    """Polar histogram of absolute wind direction.

    ``ax`` must be a *polar* axes (``fig.add_subplot(..., projection="polar")``); if none
    is given, one is created.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"projection": "polar"})
    elif not hasattr(ax, "set_theta_zero_location"):
        raise ValueError(
            "plot_wind_rose needs a polar axes: create it with "
            "fig.add_subplot(..., projection='polar')"
        )
    ang = np.deg2rad(np.mod(wind_from_deg[np.isfinite(wind_from_deg)], 360.0))
    counts, edges = np.histogram(ang, bins=bins, range=(0, 2 * np.pi))
    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align="edge", color="#2b6cb0", alpha=0.85)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=16)
    return ax


def plot_direction_hist(wind_from_deg: np.ndarray, ax=None, bins: int = 72,
                        title: str = "Absolute wind direction"):
    """Linear histogram of absolute wind direction."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(np.mod(wind_from_deg[np.isfinite(wind_from_deg)], 360.0), bins=bins,
            range=(0, 360), color="#2b6cb0", alpha=0.9)
    ax.set_xlabel("direction wind comes from [deg, true north]")
    ax.set_ylabel("records")
    ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- #
# Missingness
# --------------------------------------------------------------------------- #

def plot_missingness_by_turbine(rates: pd.Series, ax=None,
                                title: str = "Record validity by turbine"):
    """Bar chart of a per-turbine rate (e.g. invalid-target share)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(rates.index, rates.to_numpy(), color="#c05621", width=0.9)
    ax.set_xlabel("turbine ID")
    ax.set_ylabel("share of records")
    ax.set_title(title)
    return ax


def plot_gap_histogram(gap_lengths: np.ndarray, ax=None, title: str = "Gap length distribution",
                       max_gap: int = 200):
    """Histogram of contiguous invalid-gap lengths (in 10-minute steps)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.6))
    gl = gap_lengths[gap_lengths > 0]
    ax.hist(np.clip(gl, 0, max_gap), bins=min(60, max_gap), color="#805ad5", alpha=0.9)
    ax.set_xlabel(f"gap length [10-min steps, clipped at {max_gap}]")
    ax.set_ylabel("count")
    ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- #
# Baseline behaviour
# --------------------------------------------------------------------------- #

def plot_persistence_error(errors: dict, ax=None, title: str = "Persistence error vs. horizon"):
    """Plot MAE/RMSE of persistence as a function of the forecast horizon."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.8))
    for label, values in errors.items():
        ax.plot(list(values.keys()), list(values.values()), marker="o", label=label)
    ax.set_xlabel("horizon h [10-min steps]")
    ax.set_ylabel("error [kW]")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax
