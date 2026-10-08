"""Farm geometry and (Stage B) graph construction.

Stage A only needs the geometry helpers; the wind-following graph builders are added
in Stage B (notebook ``02_graph_construction.ipynb``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def coordinates(loc: pd.DataFrame) -> np.ndarray:
    """Return an ``(N_TURBINES, 2)`` array of relative ``(x, y)`` positions in metres."""
    return loc.loc[:, ["x", "y"]].to_numpy(dtype=np.float64)


def pairwise_distances(loc: pd.DataFrame) -> np.ndarray:
    """Euclidean distance matrix between turbines, in metres."""
    xy = coordinates(loc)
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def pairwise_offsets(loc: pd.DataFrame) -> np.ndarray:
    """``(N, N, 2)`` array of ``p_j - p_i`` offsets in metres."""
    xy = coordinates(loc)
    return xy[None, :, :] - xy[:, None, :]


def nearest_neighbour_stats(loc: pd.DataFrame) -> pd.DataFrame:
    """Per-turbine nearest-neighbour distance (excluding self), in metres."""
    d = pairwise_distances(loc)
    np.fill_diagonal(d, np.nan)
    nn = np.nanmin(d, axis=1)
    second = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)[:, 1]
    return pd.DataFrame({
        "TurbID": loc.index.to_numpy(),
        "x": loc["x"].to_numpy(),
        "y": loc["y"].to_numpy(),
        "nn_dist": nn,
        "second_nn_dist": second,
    })


def detect_columns(loc: pd.DataFrame, gap_threshold_m: float = 300.0) -> np.ndarray:
    """Label each turbine with its longitudinal column index.

    Turbine positions cluster into narrow strips along ``y``; columns are separated by
    x-gaps far larger than the within-column scatter (~50 m). Splitting on gaps above
    ``gap_threshold_m`` is therefore robust. Note: this yields **6** columns of ~22
    turbines, whereas Zhang et al. (2026) report five -- a discrepancy worth flagging.
    """
    x = loc["x"].to_numpy()
    order = np.argsort(x)
    xs = x[order]
    breaks = np.where(np.diff(xs) > gap_threshold_m)[0] + 1
    groups = np.split(np.arange(len(xs)), breaks)
    col = np.empty(len(x), dtype=int)
    for i, g in enumerate(groups):
        col[order[g]] = i
    return col


def spacing_summary(loc: pd.DataFrame, gap_threshold_m: float = 300.0) -> pd.DataFrame:
    """Nearest-neighbour spacing within a column vs. across columns, in metres.

    For every turbine the nearest neighbour is taken separately inside its own column and
    among the other columns; the table summarises those per-turbine distances.
    """
    d = pairwise_distances(loc)
    np.fill_diagonal(d, np.nan)
    col = detect_columns(loc, gap_threshold_m)
    same = col[:, None] == col[None, :]

    def summarize(mask):
        vals = np.where(mask, d, np.nan)
        nn = np.nanmin(vals, axis=1)
        nn = nn[np.isfinite(nn)]
        if nn.size == 0:
            return dict(count=0, min=np.nan, p25=np.nan, median=np.nan, p75=np.nan, max=np.nan)
        return dict(count=int(nn.size), min=float(nn.min()),
                    p25=float(np.percentile(nn, 25)), median=float(np.median(nn)),
                    p75=float(np.percentile(nn, 75)), max=float(nn.max()))

    rows = []
    for name, mask in (("within_column", same), ("across_columns", ~same)):
        rows.append({"neighbour_group": name, **summarize(mask)})
    allnn = np.where(np.isfinite(d), d, np.nan)
    nn_all = np.nanmin(allnn, axis=1)
    rows.append({"neighbour_group": "any",
                 "count": int(np.isfinite(nn_all).sum()),
                 "min": float(np.nanmin(nn_all)), "p25": float(np.nanpercentile(nn_all, 25)),
                 "median": float(np.nanmedian(nn_all)), "p75": float(np.nanpercentile(nn_all, 75)),
                 "max": float(np.nanmax(nn_all))})
    return pd.DataFrame(rows)


def extent(loc: pd.DataFrame) -> dict:
    """Bounding box of the farm plus the turbine count."""
    return {
        "n_turbines": int(len(loc)),
        "x_span_m": float(loc["x"].max() - loc["x"].min()),
        "y_span_m": float(loc["y"].max() - loc["y"].min()),
        "x_range": (float(loc["x"].min()), float(loc["x"].max())),
        "y_range": (float(loc["y"].min()), float(loc["y"].max())),
    }


# --------------------------------------------------------------------------- #
# Static graphs (the "fixed graph" baselines)
# --------------------------------------------------------------------------- #

def static_distance_knn(loc: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Symmetric binary adjacency of a k-nearest-neighbour graph on Euclidean distance."""
    d = pairwise_distances(loc)
    np.fill_diagonal(d, np.inf)
    n = len(loc)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, np.argsort(d[i])[:k]] = True
    return adj | adj.T


def static_radius_graph(loc: pd.DataFrame, radius_m: float) -> np.ndarray:
    """Symmetric binary adjacency connecting all pairs closer than ``radius_m``."""
    d = pairwise_distances(loc)
    np.fill_diagonal(d, np.inf)
    return d <= radius_m


def static_full_graph(loc: pd.DataFrame) -> np.ndarray:
    """Fully connected graph (paper 1's setup: six turbines, every pair connected)."""
    n = len(loc)
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    return adj


# --------------------------------------------------------------------------- #
# Farm orientation and the wind-following graph
# --------------------------------------------------------------------------- #

def flow_unit_vectors(wind_from_deg, alpha_deg: float) -> np.ndarray:
    """Unit flow vectors in the farm frame.

    ``wind_from_deg`` is the compass bearing the wind blows *from*; the flow direction is
    therefore ``wind_from + 180``. ``alpha_deg`` is the compass azimuth of the farm's +y
    axis (the unknown between the true-north SCADA angles and the relative ``(x, y)``
    coordinates): with ``alpha = 0`` the +y axis points north and +x points east.

    Returns an ``(n, 2)`` array of ``(u_x, u_y)``.
    """
    theta = np.deg2rad(np.asarray(wind_from_deg, dtype=np.float64) + 180.0)
    a = np.deg2rad(float(alpha_deg))
    return np.stack([np.sin(theta - a), np.cos(theta - a)], axis=-1)


def candidate_pairs(loc: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """All *ordered* turbine pairs within ``radius_m`` (both directions listed).

    This is the static candidate edge set: geometry is computed once, and a wind-following
    graph is then just a direction-dependent subset of it, which keeps per-snapshot graph
    construction cheap.
    """
    xy = coordinates(loc)
    ids = loc.index.to_numpy()
    i_idx, j_idx = np.meshgrid(np.arange(len(xy)), np.arange(len(xy)), indexing="ij")
    delta = xy[j_idx] - xy[i_idx]
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    keep = (dist > 0) & (dist <= radius_m)
    return pd.DataFrame({
        "i_idx": i_idx[keep],
        "j_idx": j_idx[keep],
        "i_turb": ids[i_idx[keep]],
        "j_turb": ids[j_idx[keep]],
        "dx": delta[..., 0][keep],
        "dy": delta[..., 1][keep],
        "dist": dist[keep],
    })


def pair_alignment(pairs: pd.DataFrame, wind_from_deg, alpha_deg: float) -> pd.DataFrame:
    """Add the along-flow cosine (``along``), cross-flow sine (``cross``) and angle (deg).

    ``along = 1`` means the pair axis points exactly downwind (so ``i`` is directly upwind
    of ``j``); ``along = -1`` means ``j`` is upwind of ``i``.
    """
    u = np.atleast_2d(flow_unit_vectors(wind_from_deg, alpha_deg))
    if u.shape[0] == 1:
        u = np.repeat(u, len(pairs), axis=0)
    dist = pairs["dist"].to_numpy()
    along = (pairs["dx"].to_numpy() * u[:, 0] + pairs["dy"].to_numpy() * u[:, 1]) / dist
    cross = (pairs["dx"].to_numpy() * u[:, 1] - pairs["dy"].to_numpy() * u[:, 0]) / dist
    out = pairs.copy()
    out["along"] = along
    out["cross"] = cross
    out["align_deg"] = np.rad2deg(np.arccos(np.clip(along, -1, 1)))
    return out


def edges_for_direction(pairs: pd.DataFrame, wind_from_deg, alpha_deg: float,
                        cone_deg: float, radius_m: float | None = None) -> pd.DataFrame:
    """Wind-following edges for one wind direction.

    Edge ``i -> j`` exists when ``j`` lies downwind of ``i``: the pair axis is within
    ``cone_deg`` of the flow direction and the distance is at most ``radius_m``.
    """
    a = pair_alignment(pairs, wind_from_deg, alpha_deg)
    mask = np.cos(np.deg2rad(cone_deg)) <= a["along"].to_numpy()
    if radius_m is not None:
        mask &= a["dist"].to_numpy() <= radius_m
    return a[mask].reset_index(drop=True)


def sector_index(direction_deg, n_sectors: int = 12):
    """Discretise directions into equal sectors; returns the sector index (or -1 for NaN)."""
    d = np.asarray(direction_deg, dtype=np.float64)
    out = np.full(d.shape, -1, dtype=int)
    good = np.isfinite(d)
    out[good] = (np.mod(d[good], 360.0) // (360.0 / n_sectors)).astype(int)
    return out


def sector_centers(n_sectors: int = 12) -> np.ndarray:
    """Center bearing of each sector, in degrees."""
    width = 360.0 / n_sectors
    return np.arange(n_sectors) * width + width / 2.0


def orientation_scan(
    loc: pd.DataFrame,
    power_wide: pd.DataFrame,
    farm_dir: pd.Series,
    alphas: np.ndarray,
    radius_m: float = 1000.0,
    cone_deg: float = 45.0,
    n_sectors: int = 12,
    min_snapshots_per_sector: int = 30,
) -> pd.DataFrame:
    """Score candidate farm orientations ``alpha`` by the observed wake deficit.

    **The physics.** Turbines in the same flow line sit in each other's wake, so for a pair
    aligned with the wind the upwind turbine produces slightly *more* than the downwind one.
    If the farm's axis orientation ``alpha`` is correct, aligned pairs show a positive power
    difference; at ``alpha + 180`` the same pairs are reversed and the difference flips sign.

    **The estimator.** The naive mean difference is swamped by *site quality*: per-turbine
    mean power spans ~244-463 kW, so a pair's intrinsic difference is tens of kW while the
    wake deficit is a few kW. We therefore use a within-pair (fixed-effects) estimator, which
    removes each pair's own offset:

    .. math:: D_{ij}(s) = \\beta x_{ij}(s) + \\gamma_{ij} + \\varepsilon

    where ``D_ij(s) = m_s[i] - m_s[j]`` is the sector-``s`` mean-power difference,
    ``x_ij(s) ∈ {+1, 0, -1}`` marks whether ``i`` is upwind of ``j`` (``+1``), ``j`` is
    upwind of ``i`` (``-1``) or the pair is outside the cone (``0``), and ``gamma_ij`` is the
    pair's site-quality offset. ``beta`` is then estimated from pair-demeaned quantities and
    reported with a t-statistic. The true orientation is the ``alpha`` that maximises
    ``beta``; ``alpha + 180`` should give ``-beta``.

    **Why not lead-lag?** Advection across the farm is fast: 474 m of column spacing at
    10 m/s takes ~47 s, which is 0.08 of a 10-minute sample, so a cross-correlation lag test
    cannot resolve it. The *magnitude* of the deficit is the identifiable signal.
    """
    dirs = farm_dir.reindex(power_wide.index).to_numpy()
    sec = sector_index(dirs, n_sectors)
    mat = power_wide.to_numpy()
    n_turb = mat.shape[1]

    sector_means = np.full((n_sectors, n_turb), np.nan)
    sector_counts = np.zeros(n_sectors, dtype=int)
    for s in range(n_sectors):
        rows = sec == s
        sector_counts[s] = int(rows.sum())
        if rows.sum() >= min_snapshots_per_sector:
            with np.errstate(invalid="ignore"):
                sector_means[s] = np.nanmean(mat[rows], axis=0)

    usable = np.flatnonzero(sector_counts >= min_snapshots_per_sector)

    # Unordered pairs only: (i, j) and (j, i) carry the same information, mirrored.
    pairs = candidate_pairs(loc, radius_m)
    pairs = pairs[pairs["i_idx"] < pairs["j_idx"]].reset_index(drop=True)
    ii = pairs["i_idx"].to_numpy()
    jj = pairs["j_idx"].to_numpy()
    centers = sector_centers(n_sectors)

    # Observation matrix: rows = (pair, sector) with both sector means finite.
    D = np.full((len(pairs), len(usable)), np.nan)
    for col, s in enumerate(usable):
        m = sector_means[s]
        D[:, col] = m[ii] - m[jj]
    obs_ok = np.isfinite(D)

    rows = []
    for alpha in np.atleast_1d(alphas):
        X = np.zeros_like(D)
        for col, s in enumerate(usable):
            aligned = edges_for_direction(pairs, centers[s], alpha, cone_deg, radius_m)
            if aligned.empty:
                continue
            a_i = aligned["i_idx"].to_numpy()
            a_j = aligned["j_idx"].to_numpy()
            X[a_i, col] += 1.0     # i is upwind of j
            X[a_j, col] -= 1.0     # j is upwind of i

        mask = obs_ok
        if mask.sum() == 0:
            rows.append({"alpha_deg": float(alpha), "beta_kW": np.nan, "t_stat": np.nan,
                         "n_obs": 0, "n_pairs": 0})
            continue

        x = X[mask]
        d = D[mask]
        # pair index for each observation, for demeaning
        pair_of_obs = np.broadcast_to(np.arange(len(pairs))[:, None], D.shape)[mask]
        n_obs = x.size
        # within-pair demeaning (removes gamma_ij)
        df_pair = pd.DataFrame({"p": pair_of_obs, "x": x, "d": d})
        g = df_pair.groupby("p", observed=True)
        xd = x - g["x"].transform("mean").to_numpy()
        dd = d - g["d"].transform("mean").to_numpy()
        sxx = float((xd ** 2).sum())
        if sxx <= 0:
            rows.append({"alpha_deg": float(alpha), "beta_kW": np.nan, "t_stat": np.nan,
                         "n_obs": n_obs, "n_pairs": int(df_pair["p"].nunique())})
            continue
        beta = float((xd * dd).sum() / sxx)
        resid = dd - beta * xd
        n_pairs = int(df_pair["p"].nunique())
        dof = max(n_obs - n_pairs - 1, 1)
        sigma2 = float((resid ** 2).sum() / dof)
        se = float(np.sqrt(sigma2 / sxx))
        rows.append({"alpha_deg": float(alpha), "beta_kW": beta,
                     "t_stat": beta / se if se > 0 else np.nan,
                     "n_obs": n_obs, "n_pairs": n_pairs})
    return pd.DataFrame(rows)


def downstream_gradient_scan(
    df: pd.DataFrame,
    loc: pd.DataFrame,
    farm_dir: pd.Series,
    alphas: np.ndarray,
    target_col: str = "Patv",
    valid_col: str = "target_valid",
) -> pd.DataFrame:
    """Estimate the farm orientation from the along-wind power gradient (high power, confounded).

    Regresses power on the turbine's projection onto the flow direction, with a per-turbine
    intercept:

    .. math:: P_{it} = \\mu_i + \\beta\\,(p_i \\cdot u_t) + \\varepsilon_{it}

    ``beta`` is expressed in kW per km of downstream distance. Because ``u(alpha + 180) =
    -u(alpha)``, ``beta`` is exactly antisymmetric in ``alpha``, so the scan always shows two
    extrema: the farm *axis* is identified, but the sign of ``beta`` fixes which end of that
    axis is downstream. Under the physical prior that a turbine downstream of others pro-
    duces *less*, the correct orientation is the one with ``beta < 0``.

    **Caveat.** This estimator is very precise (millions of records) but confounded: any
    static or wind-direction-dependent resource gradient along the farm produces the same
    functional form as the wake effect. Use :func:`orientation_scan` (pair fixed effects) to
    check that the chosen sign is the wake-consistent one.
    """
    use = df[[ "TurbID", "t", target_col]].copy()
    if valid_col in df.columns:
        use = use[df[valid_col].to_numpy()]
    use["dir"] = farm_dir.reindex(use["t"]).to_numpy()
    use = use[use["dir"].notna()]
    use = use[use[target_col].notna()]

    xy = loc.loc[use["TurbID"].to_numpy(), ["x", "y"]].to_numpy()
    y = use[target_col].to_numpy(dtype=np.float64)
    y = y - pd.Series(y).groupby(use["TurbID"].to_numpy()).transform("mean").to_numpy()
    dirs = use["dir"].to_numpy()

    rows = []
    for alpha in np.atleast_1d(alphas):
        u = flow_unit_vectors(dirs, alpha)
        proj = xy[:, 0] * u[:, 0] + xy[:, 1] * u[:, 1]
        xd = proj - proj.mean()
        sxx = float((xd ** 2).sum())
        beta = float((xd * y).sum() / sxx)
        resid = y - beta * xd
        se = float(np.sqrt((resid ** 2).sum() / max(len(y) - 2, 1) / sxx))
        rows.append({"alpha_deg": float(alpha), "beta_kW_per_km": beta * 1000.0,
                     "t_stat": beta / se if se > 0 else np.nan, "n_records": int(len(y))})
    return pd.DataFrame(rows)


def alignment_deficit_profile(
    loc: pd.DataFrame,
    power_wide: pd.DataFrame,
    farm_dir: pd.Series,
    alpha_deg: float,
    radius_m: float = 1000.0,
    n_sectors: int = 36,
    min_snapshots_per_sector: int = 20,
    n_angle_bins: int = 12,
) -> pd.DataFrame:
    """Wake deficit as a function of the pair's angle to the wind (the validation figure).

    Uses the same within-pair demeaning as :func:`orientation_scan`, then bins by the
    absolute angle between the pair axis and the flow direction. A physically correct
    graph should show the largest positive (upwind-minus-downwind) difference at 0 deg and
    none at 90 deg.
    """
    dirs = farm_dir.reindex(power_wide.index).to_numpy()
    sec = sector_index(dirs, n_sectors)
    mat = power_wide.to_numpy()
    n_sectors = n_sectors
    sector_means = np.full((n_sectors, mat.shape[1]), np.nan)
    usable = []
    for s in range(n_sectors):
        rows = sec == s
        if rows.sum() >= min_snapshots_per_sector:
            with np.errstate(invalid="ignore"):
                sector_means[s] = np.nanmean(mat[rows], axis=0)
            usable.append(s)

    pairs = candidate_pairs(loc, radius_m)
    pairs = pairs[pairs["i_idx"] < pairs["j_idx"]].reset_index(drop=True)
    ii = pairs["i_idx"].to_numpy()
    jj = pairs["j_idx"].to_numpy()
    centers = sector_centers(n_sectors)

    recs = []
    for s in usable:
        m = sector_means[s]
        a = pair_alignment(pairs, centers[s], alpha_deg)
        signed = np.sign(a["along"].to_numpy())         # +1: i upwind of j
        diff = (m[ii] - m[jj]) * signed                  # positive = upwind produces more
        ang = np.degrees(np.arccos(np.clip(np.abs(a["along"].to_numpy()), 0, 1)))
        for k in range(len(pairs)):
            if np.isfinite(diff[k]):
                recs.append({"pair": k, "sector": s, "angle_deg": ang[k], "deficit": diff[k]})

    obs = pd.DataFrame(recs)
    if obs.empty:
        return obs
    obs["deficit"] = obs["deficit"] - obs.groupby("pair", observed=True)["deficit"].transform("mean")
    obs["angle_bin"] = pd.cut(obs["angle_deg"], np.linspace(0, 90, n_angle_bins + 1),
                              include_lowest=True)
    prof = obs.groupby("angle_bin", observed=True)["deficit"].agg(
        mean_deficit_kW="mean", sem_kW="sem", n="size").reset_index()
    prof["angle_center_deg"] = [iv.mid for iv in prof["angle_bin"]]
    return prof



def upwind_census(
    loc: pd.DataFrame,
    farm_dir: pd.Series,
    alpha_deg: float,
    radius_m: float = 1000.0,
    cone_deg: float = 45.0,
    n_sectors: int = 12,
) -> pd.DataFrame:
    """Count upwind neighbours per turbine and per wind sector.

    The grouping variable for H2 ("benefit is largest for turbines with several upwind
    neighbours, ~zero for front-row turbines") and for identifying front-row turbines.
    """
    pairs = candidate_pairs(loc, radius_m)
    centers = sector_centers(n_sectors)
    dirs = farm_dir.to_numpy()
    sec = sector_index(dirs, n_sectors)
    _, counts = np.unique(sec[sec >= 0], return_counts=True)
    sector_n = np.zeros(n_sectors, dtype=int)
    for s, c in zip(*np.unique(sec[sec >= 0], return_counts=True)):
        sector_n[s] = c

    rows = []
    for s in range(n_sectors):
        if sector_n[s] == 0:
            continue
        aligned = edges_for_direction(pairs, centers[s], alpha_deg, cone_deg, radius_m)
        deg_in = np.bincount(aligned["j_idx"].to_numpy(), minlength=len(loc))
        deg_out = np.bincount(aligned["i_idx"].to_numpy(), minlength=len(loc))
        for k, tid in enumerate(loc.index.to_numpy()):
            rows.append({
                "TurbID": int(tid), "sector": s, "sector_center_deg": float(centers[s]),
                "n_upwind": int(deg_in[k]), "n_downwind": int(deg_out[k]),
                "n_snapshots": int(sector_n[s]),
                "share_of_time": float(sector_n[s] / max(int((sec >= 0).sum()), 1)),
            })
    return pd.DataFrame(rows)


def graph_stats(edges: pd.DataFrame, n_turbines: int = None) -> dict:
    """Basic descriptors of one snapshot graph."""
    n = n_turbines or int(max(edges["i_idx"].max(), edges["j_idx"].max()) + 1)
    deg_in = np.bincount(edges["j_idx"].to_numpy(), minlength=n)
    deg_out = np.bincount(edges["i_idx"].to_numpy(), minlength=n)
    return {
        "n_edges": int(len(edges)),
        "density": float(len(edges) / (n * (n - 1))) if n > 1 else np.nan,
        "isolated_nodes": int(((deg_in + deg_out) == 0).sum()),
        "mean_degree_in": float(deg_in.mean()),
        "mean_degree_out": float(deg_out.mean()),
        "max_degree_in": int(deg_in.max()),
        "median_distance_m": float(edges["dist"].median()) if len(edges) else np.nan,
    }
