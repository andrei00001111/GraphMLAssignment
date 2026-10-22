"""Stage C: masked-value recovery (imputation) - masks, floors, and the three-rung ladder.

The comparison requested by the team is deliberately narrow:

* **rung 0** - floors: per-turbine mean, last-observation-carried-forward, farm mean of the
  visible turbines;
* **rung 1** - **linear regression on the dataset itself**: the turbine's own features, its
  own lags and the time of day, with no neighbour information at all;
* **rung 2** - the same information **plus a kNN graph** over geometric distance;
* **rung 3** - the same information **plus the wind-following graph** (wind direction and
  angle), i.e. the project's own contribution.

Everything here runs on the wide ``(time x turbine)`` matrices produced by Stage A, so the
masking, the floors and the neighbour aggregation share one representation. The cheap
neighbour-mean rungs (2 and 3) need no neural network, which gives an early, interpretable
answer; the GNN version of the same two rungs is trained in notebook 04 with an identical body.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import graphs as G

# --------------------------------------------------------------------------- #
# Wide matrices and the design matrix for rung 1
# --------------------------------------------------------------------------- #

#: Per-turbine covariates used by the rung-1 regression (the target itself is excluded).
COVARIATES = ["Wspd", "Etmp", "Itmp", "Pab1", "Prtv"]

#: Lags of the target and of wind speed (in 10-minute steps).
TARGET_LAGS = (1, 2, 3, 6)
COVARIATE_LAGS = (1, 6)


def wide_from_long(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot a per-turbine column into a ``(time x turbine)`` matrix (slow fallback)."""
    m = df.pivot_table(index="t", columns="TurbID", values=column, aggfunc="first")
    return m.reindex(columns=np.arange(1, C.N_TURBINES + 1))


def is_turbine_major_grid(df: pd.DataFrame) -> bool:
    """True when ``df`` is the full 134 x 35 280 grid stored turbine by turbine.

    The KDD Cup file is ordered turbine-major (all 35 280 timestamps of turbine 1, then
    turbine 2, ...), so the wide matrices everyone needs are a plain reshape. That turns
    eight ``pivot_table`` calls over 4.7 M rows (~6 minutes) into a memory view.
    """
    if len(df) != C.N_RECORDS:
        return False
    codes = df["TurbID"].to_numpy().reshape(C.N_TURBINES, C.N_STEPS)
    tvals = df["t"].to_numpy().reshape(C.N_TURBINES, C.N_STEPS)
    expected_codes = np.arange(1, C.N_TURBINES + 1)[:, None]
    expected_t = np.arange(C.N_STEPS)[None, :]
    return bool((codes == expected_codes).all() and (tvals == expected_t).all())


def wide_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All ``(time x turbine)`` matrices Stage C needs, built once.

    Uses a reshape when the frame is the full turbine-major grid (the normal case) and falls
    back to ``pivot_table`` otherwise, so the function is correct either way.
    """
    columns = ["Patv", "Wspd", "Etmp", "Itmp", "Ndir", "Wdir", "Pab1", "Prtv", "target_valid"]
    if is_turbine_major_grid(df):
        index = pd.Index(np.arange(C.N_STEPS), name="t")
        cols = pd.Index(np.arange(1, C.N_TURBINES + 1), name="TurbID")

        def to_matrix(col: str) -> pd.DataFrame:
            arr = df[col].to_numpy().reshape(C.N_TURBINES, C.N_STEPS).T
            return pd.DataFrame(arr, index=index, columns=cols)
    else:
        to_matrix = lambda col: wide_from_long(df, col)  # noqa: E731

    out = {col: to_matrix(col) for col in columns[:-1]}
    out["valid"] = to_matrix("target_valid").fillna(False).astype(bool)

    # angle encodings
    for src, name in [("Wdir", "Wdir"), ("Ndir", "Ndir")]:
        rad = np.deg2rad(out[src])
        out[f"{name}_sin"] = np.sin(rad)
        out[f"{name}_cos"] = np.cos(rad)
    return out


def _time_of_day(index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Sine/cosine encoding of the 10-minute slot within the day."""
    step = np.asarray(index, dtype=np.int64) % C.STEPS_PER_DAY
    ang = 2.0 * np.pi * step / C.STEPS_PER_DAY
    return np.sin(ang), np.cos(ang)


def build_design(
    mats: dict[str, pd.DataFrame],
    target: str = "Patv",
    masked: pd.DataFrame | None = None,
    mask_covariates: bool = False,
    lag_fill: str = "ffill",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Assemble the rung-1 design matrix.

    Parameters
    ----------
    mats:
        Output of :func:`wide_matrices`.
    target:
        Column to recover (``Patv`` or ``Wspd``).
    masked:
        Optional boolean ``(time x turbine)`` matrix marking entries whose *values are
        unavailable*. Masked entries are removed from the target's own history before the
        lags are built, so the regression cannot peek at them.
    mask_covariates:
        Also hide the turbine's covariates (wind speed, temperatures, pitch, angles) wherever
        ``masked`` is True. This is the honest setting for a **fully silent turbine**: in
        reality a failed turbine does not report its own wind speed either. Leave it False for
        the ``mcar`` / ``block`` modes, where only the value being recovered is unknown.
    lag_fill:
        How masked history enters the lag features. ``"ffill"`` (default) carries the last
        observed value forward, so the lag features are the same LOCF reconstruction a
        practitioner would use - it keeps the regression strong and fair. ``"none"`` leaves
        them missing, in which case :func:`fit_linear_regression` falls back to the training
        median.

    Returns
    -------
    (X, y, names, valid)
        ``X`` is ``(n_records, n_features)`` float32, ``y`` the target, ``names`` the feature
        names and ``valid`` the per-record target-validity flag.
    """
    src = mats[target]
    if masked is not None:
        src = src.mask(masked)
    lag_src = src.ffill() if (masked is not None and lag_fill == "ffill") else src

    def _maybe_mask(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.mask(masked) if (mask_covariates and masked is not None) else frame

    features: dict[str, np.ndarray] = {}
    for lag in TARGET_LAGS:
        features[f"{target}_lag{lag}"] = lag_src.shift(lag).to_numpy(dtype=np.float32)
    # covariates must never contain the target itself
    covariates = [c for c in COVARIATES if c != target]
    for col in covariates:
        features[f"{col}_t"] = _maybe_mask(mats[col]).to_numpy(dtype=np.float32)
    if target != "Wspd":
        wspd = _maybe_mask(mats["Wspd"])
        if masked is not None and lag_fill == "ffill":
            wspd = wspd.ffill()
        for lag in COVARIATE_LAGS:
            features[f"Wspd_lag{lag}"] = wspd.shift(lag).to_numpy(dtype=np.float32)
    for col in ["Wdir_sin", "Wdir_cos", "Ndir_sin", "Ndir_cos"]:
        features[col] = _maybe_mask(mats[col]).to_numpy(dtype=np.float32)

    tod_sin, tod_cos = _time_of_day(mats[target].index)
    n_t = len(mats[target].index)
    features["tod_sin"] = np.repeat(tod_sin[:, None], C.N_TURBINES, axis=1).astype(np.float32)
    features["tod_cos"] = np.repeat(tod_cos[:, None], C.N_TURBINES, axis=1).astype(np.float32)

    names = list(features)
    X = np.stack([features[k].ravel() for k in names], axis=1)
    y = mats[target].to_numpy(dtype=np.float32).ravel()
    valid = mats["valid"].to_numpy().ravel()
    return X, y, names, valid


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #

def mask_mcar(mats: dict[str, pd.DataFrame], rate: float, seed: int = 0,
              splits: tuple[str, ...] = ("train", "val", "test")) -> pd.DataFrame:
    """Hide a random ``rate`` fraction of the valid entries, independently per turbine."""
    valid = mats["valid"].to_numpy()
    t = mats["valid"].index.to_numpy()
    in_split = np.zeros(len(t), dtype=bool)
    for name in splits:
        lo, hi = D.split_bounds(name)
        in_split |= (t >= lo) & (t <= hi)
    eligible = valid & in_split[:, None]
    rng = np.random.default_rng(seed)
    draw = rng.random(eligible.shape)
    return pd.DataFrame(eligible & (draw < rate), index=mats["valid"].index,
                        columns=mats["valid"].columns)


def mask_blocks(
    mats: dict[str, pd.DataFrame],
    gap_lengths: np.ndarray,
    n_blocks_per_turbine: int = 4,
    seed: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    max_attempts: int = 50,
) -> pd.DataFrame:
    """Hide contiguous runs whose lengths are drawn from the empirical gap distribution.

    ``gap_lengths`` should come from Stage A's contiguous-invalid-run analysis, so the masked
    gaps look like the real sensor dropouts rather than like synthetic rectangles.
    """
    valid = mats["valid"].to_numpy()
    t = mats["valid"].index.to_numpy()
    in_split = np.zeros(len(t), dtype=bool)
    for name in splits:
        lo, hi = D.split_bounds(name)
        in_split |= (t >= lo) & (t <= hi)

    lengths = np.asarray(gap_lengths, dtype=int)
    lengths = lengths[(lengths > 0) & (lengths < len(t) // 4)]
    mask = np.zeros_like(valid)
    rng = np.random.default_rng(seed)

    for j in range(valid.shape[1]):
        col = valid[:, j] & in_split
        starts = np.flatnonzero(col)
        if starts.size == 0:
            continue
        placed = 0
        for _ in range(max_attempts):
            if placed >= n_blocks_per_turbine:
                break
            length = int(rng.choice(lengths))
            s = int(rng.choice(starts))
            e = min(s + length, valid.shape[0])
            window = col[s:e]
            if window.mean() < 0.9:          # keep blocks inside genuinely valid stretches
                continue
            mask[s:e, j] = True
            placed += 1
    return pd.DataFrame(mask, index=mats["valid"].index, columns=mats["valid"].columns)


def mask_turbines(mats: dict[str, pd.DataFrame], turbine_ids, splits: tuple[str, ...] = ("train", "val", "test")) -> pd.DataFrame:
    """Hide entire records of the given turbines."""
    ids = np.atleast_1d(turbine_ids)
    cols = mats["valid"].columns
    mask = pd.DataFrame(False, index=mats["valid"].index, columns=cols)
    keep = [c for c in cols if c in set(ids.tolist())]
    mask.loc[:, keep] = True
    return mask


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE and R² on the entries where both are finite."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    a, b = y_true[ok], y_pred[ok]
    err = b - a
    ss_res = float((err ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return {
        "n": int(ok.sum()),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Rung 0: floors
# --------------------------------------------------------------------------- #

def floor_predictions(mats: dict[str, pd.DataFrame], target: str, mask: pd.DataFrame,
                      train_only: bool = True) -> dict[str, pd.DataFrame]:
    """Naive predictions for every entry (evaluated later on the masked ones).

    * ``mean``  - per-turbine mean of the observed training values;
    * ``ffill`` - last observation carried forward (per turbine);
    * ``farm``  - mean over the turbines observed at the same timestamp.
    """
    values = mats[target]
    observed = ~mask

    train_lo, train_hi = D.split_bounds("train") if train_only else (0, values.shape[0] - 1)
    fit_rows = (values.index >= train_lo) & (values.index <= train_hi)
    per_turb_mean = values.where(observed).iloc[fit_rows].mean()
    mean_pred = pd.DataFrame(np.tile(per_turb_mean.reindex(values.columns).to_numpy(),
                                     (values.shape[0], 1)),
                             index=values.index, columns=values.columns)

    ffill_pred = values.where(observed).ffill()
    farm_pred = values.where(observed).mean(axis=1)
    farm_df = pd.DataFrame(np.repeat(farm_pred.to_numpy()[:, None], values.shape[1], axis=1),
                           index=values.index, columns=values.columns)
    return {"mean": mean_pred, "ffill": ffill_pred, "farm_mean": farm_df}


# --------------------------------------------------------------------------- #
# Rung 1: linear regression on the dataset itself
# --------------------------------------------------------------------------- #

def fit_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    per_turbine: bool = False,
    turbine_codes: np.ndarray | None = None,
    ridge: float = 1e-6,
    n_turbines: int = C.N_TURBINES,
) -> np.ndarray:
    """Ordinary least squares, either pooled (with turbine fixed effects) or per turbine.

    The pooled variant uses the within (fixed-effects) transformation - each feature and the
    target are demeaned per turbine - which is algebraically identical to including turbine
    intercepts but without materialising 133 dummy columns. That matters at this data size
    (millions of rows) and is also what makes the pooled model usable when a whole turbine is
    hidden.

    ``train_rows`` marks the entries used for fitting (train split, valid target, not masked);
    predictions are returned for the full record axis with NaN outside ``eval_rows``.

    Numerics are float32 for the large arrays (memory) and float64 for the small normal
    equations (stability); a tiny ridge term keeps the solve well conditioned.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n_feat = X.shape[1]
    out = np.full(X.shape[0], np.nan, dtype=np.float64)

    def _solve(Xa: np.ndarray, ya: np.ndarray) -> np.ndarray:
        Xa = Xa.astype(np.float64)
        ya = ya.astype(np.float64)
        XtX = Xa.T @ Xa
        Xty = Xa.T @ ya
        XtX[np.diag_indices_from(XtX)] += ridge * max(np.trace(XtX) / n_feat, 1e-12)
        return np.linalg.solve(XtX, Xty)

    median = np.nanmedian(X[train_rows], axis=0).astype(np.float32)
    Xf = np.where(np.isfinite(X), X, median)

    if per_turbine:
        if turbine_codes is None:
            raise ValueError("per_turbine fitting needs turbine_codes")
        for k in range(int(turbine_codes.max()) + 1):
            fit = train_rows & (turbine_codes == k)
            ev = eval_rows & (turbine_codes == k)
            if fit.sum() < n_feat * 5 or ev.sum() == 0:
                continue
            mu_x = Xf[fit].mean(axis=0)
            mu_y = float(y[fit].mean())
            coef = _solve(Xf[fit] - mu_x, y[fit] - mu_y)
            out[ev] = (Xf[ev].astype(np.float64) - mu_x) @ coef + mu_y
        return out

    if turbine_codes is None:
        raise ValueError("pooled fitting needs turbine_codes")

    # per-turbine means of the fit rows (loop keeps memory flat at 134 x n_feat)
    Xbar = np.zeros((n_turbines, n_feat), dtype=np.float32)
    ybar = np.zeros(n_turbines, dtype=np.float32)
    for k in range(n_turbines):
        fit = train_rows & (turbine_codes == k)
        if fit.any():
            Xbar[k] = Xf[fit].mean(axis=0)
            ybar[k] = y[fit].mean()

    fit = train_rows
    coef = _solve(Xf[fit] - Xbar[turbine_codes[fit]], y[fit] - ybar[turbine_codes[fit]])
    ev = eval_rows
    out[ev] = (Xf[ev].astype(np.float64) - Xbar[turbine_codes[ev]]) @ coef + ybar[turbine_codes[ev]]
    return out


# --------------------------------------------------------------------------- #
# Rungs 2 and 3: neighbour aggregation (the cheap, training-free graph models)
# --------------------------------------------------------------------------- #

def graph_operator(edges: pd.DataFrame, n_turbines: int = C.N_TURBINES,
                   weight: str | None = None) -> np.ndarray:
    """Row-normalised aggregation matrix for the edges ``i -> j``.

    Entry ``A[j, i] = w_ij / sum_k w_kj``, so ``A @ v`` gives, for every node ``j``, the
    weighted mean of the values of its upwind neighbours ``i``. ``weight='inverse_distance'``
    down-weights distant neighbours; the default is uniform.
    """
    A = np.zeros((n_turbines, n_turbines), dtype=np.float64)
    i = edges["i_idx"].to_numpy()
    j = edges["j_idx"].to_numpy()
    w = np.ones(len(edges), dtype=np.float64)
    if weight == "inverse_distance":
        w = 1.0 / np.maximum(edges["dist"].to_numpy(), 1.0)
    np.add.at(A, (j, i), w)
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return A / deg


def neighbour_mean_predictions(
    values: pd.DataFrame,
    observed: pd.DataFrame,
    operator: np.ndarray,
) -> pd.DataFrame:
    """Aggregate each turbine's neighbours from the *observed* values only.

    Uses the zero-fill + count trick so missing neighbours are excluded rather than treated
    as zeros: ``mean = (V0 @ A^T) / (O @ A^T)`` with ``V0`` the zero-filled values and ``O``
    the observed indicator.
    """
    V = values.to_numpy(dtype=np.float64)
    O = observed.to_numpy(dtype=np.float64)
    V0 = np.where(np.isfinite(V), V, 0.0)
    O = O * np.isfinite(V)
    num = V0 @ operator.T
    den = O @ operator.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def edges_from_adjacency(adj: np.ndarray, dist: np.ndarray | None = None) -> pd.DataFrame:
    """Long-form directed edge list (both directions) from a symmetric adjacency matrix."""
    i_idx, j_idx = np.nonzero(adj)
    d = np.sqrt(np.array([0.0])) if dist is None else dist[i_idx, j_idx]
    return pd.DataFrame({"i_idx": i_idx, "j_idx": j_idx, "dist": d})


def wind_operators(
    loc: pd.DataFrame,
    alpha_deg: float,
    radius_m: float,
    cone_deg: float,
    pairs: pd.DataFrame | None = None,
    n_sectors: int = 12,
    n_turbines: int = C.N_TURBINES,
    weight: str | None = None,
) -> list[np.ndarray]:
    """One aggregation matrix per wind sector, so the graph follows the wind.

    ``operator_s[j, i]`` is the weight with which upwind turbine ``i`` contributes to ``j``
    when the wind is in sector ``s``; the caller picks the sector of each timestamp with
    :func:`sector_assignments`.
    """
    if pairs is None:
        pairs = G.candidate_pairs(loc, radius_m)
    pairs = pairs[pairs.dist <= radius_m]
    ops = []
    for s in range(n_sectors):
        edges = G.edges_for_direction(pairs, G.sector_centers(n_sectors)[s], alpha_deg,
                                      cone_deg, radius_m)
        ops.append(graph_operator(edges, n_turbines=n_turbines, weight=weight))
    return ops


def sector_assignments(farm_dir: pd.Series, t_index: np.ndarray, n_sectors: int = 12) -> np.ndarray:
    """Sector index for every timestamp in ``t_index`` (-1 where the direction is unknown)."""
    dirs = farm_dir.reindex(t_index).to_numpy()
    return G.sector_index(dirs, n_sectors)


def neighbour_mean_by_sector(
    values: pd.DataFrame,
    observed: pd.DataFrame,
    operators: list[np.ndarray],
    sectors: np.ndarray,
    fallback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Neighbour means using the graph of the *current* wind sector.

    Where a turbine has no upwind neighbour in its sector, the value falls back to
    ``fallback`` (typically the farm mean of visible turbines), so the comparison is not
    decided by coverage alone.
    """
    out = np.full(values.shape, np.nan, dtype=np.float64)
    V = values.to_numpy(dtype=np.float64)
    O = observed.to_numpy(dtype=np.float64) * np.isfinite(V)
    V0 = np.where(np.isfinite(V), V, 0.0)
    for s, A in enumerate(operators):
        rows = sectors == s
        if not rows.any():
            continue
        num = V0[rows] @ A.T
        den = O[rows] @ A.T
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows] = np.where(den > 0, num / den, np.nan)
    pred = pd.DataFrame(out, index=values.index, columns=values.columns)
    if fallback is not None:
        pred = pred.where(np.isfinite(pred), fallback)
    return pred


# --------------------------------------------------------------------------- #
# Evaluation harness
# --------------------------------------------------------------------------- #

def evaluate_predictions(
    mats: dict[str, pd.DataFrame],
    target: str,
    mask: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    split: str = "test",
) -> pd.DataFrame:
    """Score every prediction on the masked entries of one split."""
    lo, hi = D.split_bounds(split)
    t = mats[target].index.to_numpy()
    rows = (t >= lo) & (t <= hi)
    m = mask.to_numpy()[rows]
    truth = mats[target].to_numpy(dtype=np.float64)[rows]
    valid = mats["valid"].to_numpy()[rows]
    evaluate = m & valid
    out = []
    for name, pred in predictions.items():
        p = pred.to_numpy(dtype=np.float64)[rows]
        met = regression_metrics(np.where(evaluate, truth, np.nan), np.where(evaluate, p, np.nan))
        out.append({"model": name, "split": split, **met})
    return pd.DataFrame(out)
