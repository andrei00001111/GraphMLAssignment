"""Non-graph baselines for single-turbine gap filling.

The task: turbine ``i`` stops reporting, its last reading was ``s`` steps ago, and we
reconstruct ``Patv[i, t]``. All of turbine ``i``'s own channels are hidden from ``t-s+1``
onwards; every other turbine reports normally. ``s`` is the only knob -- ``s = 1`` is
imputation from a 10-minute-old reading, ``s = 36`` is a 6-hour gap -- so one sweep over
``s`` gives the whole range between imputation and forecasting.

Each model implements ``fit(ctx, s)`` and ``predict(ctx, ep)``, where ``ctx`` holds the
power matrices and ``ep`` is a set of sampled episodes. Every model is scored on the
identical episode set, so the MAE column is directly comparable row to row.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

RNG_SEED = 16
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


# --------------------------------------------------------------------------- #
# Context: the matrices every model reads
# --------------------------------------------------------------------------- #

@dataclass
class Context:
    """``time x turbine`` views of the farm, plus the time-of-day climatology."""

    power: np.ndarray          # (T, N) observed Patv, NaN where the sensor was dark
    valid: np.ndarray          # (T, N) bool, usable as a target
    clim: np.ndarray           # (N, 144) per-turbine mean power by time of day
    obs_sum: np.ndarray        # (T,) sum over reporting turbines
    obs_count: np.ndarray      # (T,) number of reporting turbines

    @property
    def n_steps(self) -> int:
        return self.power.shape[0]

    @property
    def n_turbines(self) -> int:
        return self.power.shape[1]


def build_context(df: pd.DataFrame) -> Context:
    """Pivot the analysis table into the matrices the baselines consume."""
    turbines = np.arange(1, C.N_TURBINES + 1)
    steps = np.arange(C.N_STEPS)
    # Reindex on BOTH axes: pivot_table drops the 264 steps where no turbine reported at
    # all, which would otherwise shift every row index away from the global step t and
    # make a lag of s steps silently jump across the 37.8 h farm-wide blackout.
    power = (
        df.pivot_table(index="t", columns="TurbID", values="Patv", aggfunc="first")
        .reindex(index=steps, columns=turbines)
        .to_numpy(dtype=np.float32)
        .copy()                       # pandas 3 hands back a read-only view
    )
    valid = (
        df.pivot_table(index="t", columns="TurbID", values="target_valid", aggfunc="first")
        .reindex(index=steps, columns=turbines)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    power[~np.isfinite(power)] = np.nan

    lo, hi = D.split_bounds("train")
    tod = np.arange(power.shape[0]) % C.STEPS_PER_DAY
    clim = np.zeros((power.shape[1], C.STEPS_PER_DAY), dtype=np.float32)
    train = slice(lo, hi + 1)
    for k in range(C.STEPS_PER_DAY):
        rows = np.nonzero(tod[train] == k)[0] + lo
        block = np.where(valid[rows], power[rows], np.nan)
        clim[:, k] = np.nan_to_num(np.nanmean(block, axis=0))

    reporting = np.isfinite(power)
    obs_sum = np.where(reporting, power, 0.0).sum(axis=1, dtype=np.float64)
    obs_count = reporting.sum(axis=1).astype(np.float64)
    return Context(power=power, valid=valid, clim=clim, obs_sum=obs_sum, obs_count=obs_count)


# --------------------------------------------------------------------------- #
# Episodes
# --------------------------------------------------------------------------- #

@dataclass
class Episodes:
    """Sampled gap-filling cases: turbine ``turb`` is dark, we predict step ``t``."""

    t: np.ndarray
    turb: np.ndarray           # 0-based column index
    s: int
    split: str

    def __len__(self) -> int:
        return self.t.size

    def truth(self, ctx: Context) -> np.ndarray:
        return ctx.power[self.t, self.turb]


def sample_episodes(
    ctx: Context,
    split: str,
    s: int,
    n: int | None = None,
    window: int = 12,
    rng: np.random.Generator | None = None,
) -> Episodes:
    """Draw cases where the target is valid and the last pre-gap reading exists.

    Requiring ``power[t-s]`` to be observed is what makes persistence and the
    history-based models defined at all; every model is then scored on this one set.
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    lo, hi = D.split_bounds(split)
    first = lo + s + window                       # room for the history window
    rows = np.arange(first, hi + 1)

    ok = ctx.valid[rows] & np.isfinite(ctx.power[rows - s])
    ti, tj = np.nonzero(ok)
    t, turb = rows[ti], tj
    if n is not None and t.size > n:
        pick = rng.choice(t.size, size=n, replace=False)
        pick.sort()
        t, turb = t[pick], turb[pick]
    return Episodes(t=t, turb=turb, s=s, split=split)


# --------------------------------------------------------------------------- #
# Ridge helper
# --------------------------------------------------------------------------- #

def _ridge_path(X: np.ndarray, y: np.ndarray, alphas=ALPHAS) -> dict:
    """Standardised ridge for several alphas, sharing one Gram matrix."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = float(y.mean())
    gram = Z.T @ Z
    rhs = Z.T @ (y - ym)
    eye = np.eye(gram.shape[0])
    return {a: (np.linalg.solve(gram + a * eye, rhs), mu, sd, ym) for a in alphas}


def _apply(fit, X: np.ndarray) -> np.ndarray:
    w, mu, sd, ym = fit
    return ((np.asarray(X, dtype=np.float64) - mu) / sd) @ w + ym


def _pick_alpha(paths, X_val, y_val):
    """Lowest validation MAE wins."""
    best, best_mae = None, np.inf
    for a, fit in paths.items():
        mae = float(np.mean(np.abs(_apply(fit, X_val) - y_val)))
        if mae < best_mae:
            best, best_mae = a, mae
    return best, paths[best]


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #

def _own_lags(ctx: Context, t: np.ndarray, turb: np.ndarray, s: int, window: int) -> np.ndarray:
    """``window`` readings of the turbine's own power, ending at its last live step."""
    offsets = s + np.arange(window)
    rows = t[:, None] - offsets[None, :]
    vals = ctx.power[rows, turb[:, None]]
    gap = ~np.isfinite(vals)
    if gap.any():
        tod = rows[gap] % C.STEPS_PER_DAY
        vals[gap] = ctx.clim[np.broadcast_to(turb[:, None], rows.shape)[gap], tod]
    return vals.astype(np.float64)


def _peer_power(ctx: Context, t: np.ndarray, turb: np.ndarray) -> np.ndarray:
    """Every other turbine's power at the step being predicted, own column zeroed.

    A peer that is itself dark is filled with the mean over reporting turbines
    *excluding the masked one*. Using the plain farm mean would feed the answer back in
    at roughly 1/134 weight, which is leakage, not imputation.
    """
    block = ctx.power[t].astype(np.float64)
    own = ctx.power[t, turb]
    has_own = np.isfinite(own)
    tot = np.where(has_own, ctx.obs_sum[t] - np.nan_to_num(own), ctx.obs_sum[t])
    cnt = np.where(has_own, ctx.obs_count[t] - 1.0, ctx.obs_count[t])
    loo = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    gap = ~np.isfinite(block)
    if gap.any():
        block[gap] = np.broadcast_to(loo[:, None], block.shape)[gap]
    block[np.arange(t.size), turb] = 0.0
    return block


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class Baseline:
    name = "baseline"

    def fit(self, ctx: Context, s: int, window: int) -> "Baseline":
        return self

    def predict(self, ctx: Context, ep: Episodes) -> np.ndarray:
        raise NotImplementedError


class Climatology(Baseline):
    """Per-turbine mean power for that time of day. The do-nothing floor."""

    name = "climatology"

    def predict(self, ctx, ep):
        return ctx.clim[ep.turb, ep.t % C.STEPS_PER_DAY].astype(np.float64)


class Persistence(Baseline):
    """The turbine's last reading before it went dark, held flat."""

    name = "persistence"

    def predict(self, ctx, ep):
        return ctx.power[ep.t - ep.s, ep.turb].astype(np.float64)


class LinearOwn(Baseline):
    """Ridge on the turbine's own recent history. Temporal signal only.

    Pooled over turbines: every turbine is the same 1.5 MW machine, and the features
    are relative to its own last live step, so one coefficient vector serves all 134.
    """

    name = "linear-own"

    def __init__(self, n_train: int = 200_000):
        self.n_train = n_train

    def fit(self, ctx, s, window):
        rng = np.random.default_rng(RNG_SEED)
        tr = sample_episodes(ctx, "train", s, self.n_train, window, rng)
        va = sample_episodes(ctx, "val", s, 50_000, window, rng)
        Xtr, Xva = _own_lags(ctx, tr.t, tr.turb, s, window), _own_lags(ctx, va.t, va.turb, s, window)
        self.alpha, self.fit_ = _pick_alpha(
            _ridge_path(Xtr, tr.truth(ctx).astype(np.float64)), Xva, va.truth(ctx).astype(np.float64)
        )
        self.window = window
        return self

    def predict(self, ctx, ep):
        return _apply(self.fit_, _own_lags(ctx, ep.t, ep.turb, ep.s, self.window))


class LinearPeers(Baseline):
    """Ridge on all 133 other turbines at the step being predicted. Spatial signal only.

    This is the model the GNN has to beat: it already sees every neighbour's value, so
    whatever a graph adds on top of it is structure, not access to information.
    One coefficient vector per turbine, fitted on every valid training step.
    """

    name = "linear-peers"

    def __init__(self, n_train: int = 12_000):
        self.n_train = n_train

    def _rows(self, ctx, split, j, cap, rng):
        lo, hi = D.split_bounds(split)
        rows = np.nonzero(ctx.valid[lo : hi + 1, j])[0] + lo
        if rows.size > cap:
            rows = np.sort(rng.choice(rows, size=cap, replace=False))
        return rows

    def fit(self, ctx, s, window):
        rng = np.random.default_rng(RNG_SEED)
        self.fits_, self.alphas_ = {}, {}
        for j in range(ctx.n_turbines):
            tr = self._rows(ctx, "train", j, self.n_train, rng)
            va = self._rows(ctx, "val", j, 4_000, rng)
            if tr.size < 500 or va.size < 50:
                self.fits_[j] = None
                continue
            col = np.full(tr.size, j)
            Xtr = _peer_power(ctx, tr, col)
            Xva = _peer_power(ctx, va, np.full(va.size, j))
            a, fit = _pick_alpha(
                _ridge_path(Xtr, ctx.power[tr, j].astype(np.float64)),
                Xva,
                ctx.power[va, j].astype(np.float64),
            )
            self.fits_[j], self.alphas_[j] = fit, a
        return self

    def predict(self, ctx, ep):
        out = np.empty(len(ep), dtype=np.float64)
        X = _peer_power(ctx, ep.t, ep.turb)
        for j in np.unique(ep.turb):
            m = ep.turb == j
            fit = self.fits_.get(j)
            out[m] = (
                ctx.clim[j, ep.t[m] % C.STEPS_PER_DAY] if fit is None else _apply(fit, X[m])
            )
        return out


class LinearBoth(Baseline):
    """Ridge on own history and all other turbines together. Linear ceiling."""

    name = "linear-both"

    def __init__(self, n_train: int = 8_000):
        self.n_train = n_train

    def fit(self, ctx, s, window):
        rng = np.random.default_rng(RNG_SEED)
        tr = sample_episodes(ctx, "train", s, None, window, rng)
        va = sample_episodes(ctx, "val", s, None, window, rng)
        self.fits_ = {}
        for j in range(ctx.n_turbines):
            mt = np.nonzero(tr.turb == j)[0]
            mv = np.nonzero(va.turb == j)[0]
            if mt.size > self.n_train:
                mt = np.sort(rng.choice(mt, size=self.n_train, replace=False))
            if mv.size > 3_000:
                mv = np.sort(rng.choice(mv, size=3_000, replace=False))
            if mt.size < 500 or mv.size < 50:
                self.fits_[j] = None
                continue
            Xtr = np.hstack(
                [
                    _peer_power(ctx, tr.t[mt], tr.turb[mt]),
                    _own_lags(ctx, tr.t[mt], tr.turb[mt], s, window),
                ]
            )
            Xva = np.hstack(
                [
                    _peer_power(ctx, va.t[mv], va.turb[mv]),
                    _own_lags(ctx, va.t[mv], va.turb[mv], s, window),
                ]
            )
            _, fit = _pick_alpha(
                _ridge_path(Xtr, tr.truth(ctx)[mt].astype(np.float64)),
                Xva,
                va.truth(ctx)[mv].astype(np.float64),
            )
            self.fits_[j] = fit
        self.window = window
        return self

    def predict(self, ctx, ep):
        out = np.empty(len(ep), dtype=np.float64)
        for j in np.unique(ep.turb):
            m = np.nonzero(ep.turb == j)[0]
            fit = self.fits_.get(j)
            if fit is None:
                out[m] = ctx.clim[j, ep.t[m] % C.STEPS_PER_DAY]
                continue
            X = np.hstack(
                [
                    _peer_power(ctx, ep.t[m], ep.turb[m]),
                    _own_lags(ctx, ep.t[m], ep.turb[m], ep.s, self.window),
                ]
            )
            out[m] = _apply(fit, X)
        return out


DEFAULT_MODELS = (Climatology, Persistence, LinearOwn, LinearPeers, LinearBoth)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score(truth: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - truth
    return {
        "MAE [kW]": float(np.mean(np.abs(err))),
        "RMSE [kW]": float(np.sqrt(np.mean(err ** 2))),
    }


def run(
    ctx: Context,
    silences=(1, 6, 36),
    window: int = 12,
    n_test: int | None = 40_000,
    models=DEFAULT_MODELS,
    verbose: bool = True,
    purge: int | None = None,
) -> pd.DataFrame:
    """Fit and score every model at every silence length. One row per (model, s).

    ``purge`` forces the history headroom the episode sampler leaves, instead of deriving
    it from this call's model list. Pass the value an earlier run used when you want rows
    that are comparable to it: dropping a long-history model from the list would
    otherwise shorten the purge and silently draw a different episode set.
    """
    import time

    # Purging is only honoured if the episode set leaves room for the LONGEST history any
    # model reads; a model declaring `required_window` would otherwise reach back across
    # the split boundary for episodes near the start of the test block.
    need = max([window] + [getattr(m, "required_window", 0) for m in models])
    if purge is not None:
        if purge < need:
            raise ValueError(f"purge={purge} is shorter than the {need} steps these models read")
        need = purge

    rows = []
    for s in silences:
        ep = sample_episodes(ctx, "test", s, n_test, need, np.random.default_rng(RNG_SEED))
        truth = ep.truth(ctx).astype(np.float64)
        if verbose:
            print(f"s = {s:3d} ({s * C.SAMPLE_MINUTES:4d} min)   test episodes: {len(ep):,}")
        for cls in models:
            t0 = time.time()
            # accept either a class or a ready-made instance (models needing extra state)
            model = (cls() if isinstance(cls, type) else cls).fit(ctx, s, window)
            pred = model.predict(ctx, ep)
            row = {"model": model.name, "silence": s, "minutes": s * C.SAMPLE_MINUTES}
            row.update(score(truth, pred))
            row["n"] = len(ep)
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            if verbose:
                print(f"    {model.name:<14s} MAE {row['MAE [kW]']:7.2f}   "
                      f"RMSE {row['RMSE [kW]']:7.2f}   {row['seconds']:5.1f}s")
    return pd.DataFrame(rows)
