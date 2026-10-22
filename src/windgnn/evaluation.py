"""Stage D/E: metrics, degradation analysis and the paired tests behind H1-H3.

All metrics are computed on **scoreable ground truth only** (``Prediction.valid``), in kW, and
every comparison between two scenarios uses the *same* entries - the "complete data" reference
is the same windows with no silent turbine, so a degradation number is a paired quantity, not
a difference of two unrelated averages.

Definitions used throughout (and stated in the report):

``MAE``/``RMSE``
    mean absolute / root mean squared error, kW.
``R2``
    coefficient of determination against the **variance of the evaluated entries**, so it is
    meaningful for a subset (a subset with little variance cannot produce a large R2).
``degradation``
    ``(MAE_scenario - MAE_reference) / MAE_reference`` on the shared entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import graphs as G
from . import tensors as T
from .training import Predictions

# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #

def error_arrays(preds: Predictions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(signed error, abs error, squared error)`` per entry, NaN where not scoreable."""
    err = preds.pred.astype(np.float64) - preds.true.astype(np.float64)
    ok = preds.valid & np.isfinite(err)
    err = np.where(ok, err, np.nan)
    return err, np.abs(err), err ** 2


def error_matrix(preds: Predictions) -> np.ndarray:
    """Absolute error per entry, NaN where the ground truth is not scoreable."""
    _, abs_err, _ = error_arrays(preds)
    return abs_err


def metric_row(err: np.ndarray, true: np.ndarray | None = None) -> dict:
    """MAE/RMSE/R2/bias/n from a matrix of signed errors (NaN = skip).

    ``R2`` needs the targets: it is ``1 - SS_res / SS_tot`` around the mean of the *evaluated*
    targets, so it stays meaningful for subsets (unlike an R2 computed over all records).
    """
    ok = np.isfinite(err)
    if not ok.any():
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "bias": np.nan}
    signed = err[ok]
    r2 = np.nan
    if true is not None:
        target = np.asarray(true, dtype=np.float64)[ok]
        ss_res = float((signed ** 2).sum())
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    return {
        "n": int(signed.size),
        "MAE": float(np.abs(signed).mean()),
        "RMSE": float(np.sqrt((signed ** 2).mean())),
        "R2": r2,
        "bias": float(signed.mean()),
    }


def score(preds: Predictions, name: str = "model", subset: str = "all") -> dict:
    """One metrics row for a whole prediction set.

    ``subset`` selects which entries are scored: ``"all"`` (every scoreable entry),
    ``"hidden"`` (only silent turbines) or ``"visible"`` (only reporting turbines).
    """
    err, _, _ = error_arrays(preds)
    true = preds.true
    if subset == "hidden":
        keep = preds.hidden
        err = np.where(keep, err, np.nan)
        true = np.where(keep, true, np.nan)
    elif subset == "visible":
        err = np.where(preds.hidden, np.nan, err)
        true = np.where(preds.hidden, np.nan, true)
    return {"model": name, "subset": subset, **metric_row(err, true)}


# --------------------------------------------------------------------------- #
# Degradation against the complete-data reference
# --------------------------------------------------------------------------- #

def degradation_table(preds: dict[str, Predictions], reference: str,
                      subset: str = "hidden") -> pd.DataFrame:
    """MAE of every scenario and its degradation w.r.t. ``reference`` on shared entries.

    The reference must be the *same windows* with nobody hidden (build the datasets with the
    same ``t_end``/``h`` arrays), otherwise the comparison is not paired and the function
    refuses to compute it.
    """
    ref = preds[reference]
    ref_err, _, _ = error_arrays(ref)
    ref_true = ref.true
    rows = []
    for name, p in preds.items():
        if (p.t_end.shape != ref.t_end.shape) or not np.array_equal(p.t_end, ref.t_end):
            raise ValueError(f"{name!r} is not aligned with the reference windows")
        err, _, _ = error_arrays(p)
        sel = ref_err
        true = ref_true
        if subset == "hidden":
            sel = np.where(p.hidden, sel, np.nan)
            true = np.where(p.hidden, true, np.nan)
        elif subset == "visible":
            sel = np.where(p.hidden, np.nan, sel)
            true = np.where(p.hidden, np.nan, true)
        ref_row = metric_row(sel, true)
        row = {"model": name, **metric_row(np.where(np.isfinite(sel), err, np.nan), true)}
        row["MAE_reference"] = ref_row["MAE"]
        row["delta_MAE"] = (row["MAE"] - ref_row["MAE"]) / ref_row["MAE"] if ref_row["MAE"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def matched_improvement(preds_a: Predictions, preds_b: Predictions,
                        subset: str = "hidden") -> dict:
    """Paired comparison of two models on identical windows and masks.

    Returns MAE of each, the relative improvement of ``b`` over ``a``, and a bootstrap-free
    paired statistic (mean of per-entry absolute-error differences with its standard error).
    """
    ea, _, _ = error_arrays(preds_a)
    eb, _, _ = error_arrays(preds_b)
    if subset == "hidden":
        ea = np.where(preds_a.hidden, ea, np.nan)
        eb = np.where(preds_b.hidden, eb, np.nan)
    both = np.isfinite(ea) & np.isfinite(eb)
    if both.sum() == 0:
        return {"n": 0, "MAE_a": np.nan, "MAE_b": np.nan, "improvement": np.nan}
    aa, bb = np.abs(ea[both]), np.abs(eb[both])
    diff = aa - bb                              # >0 means b is better
    return {
        "n": int(both.sum()),
        "MAE_a": float(aa.mean()),
        "MAE_b": float(bb.mean()),
        "improvement": float((aa.mean() - bb.mean()) / aa.mean()) if aa.mean() else np.nan,
        "mean_entry_improvement_kW": float(diff.mean()),
        "se_entry_improvement_kW": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Grouping (H2): error by number of upwind neighbours
# --------------------------------------------------------------------------- #

def groups_from_in_degree(preds: Predictions, graph: T.GraphProvider) -> np.ndarray:
    """``(n_samples, N)`` in-degree of each turbine at each sample's own wind direction.

    This is H2's grouping variable. It is computed from the graph *actually used* by the
    model, so "front-row" is defined consistently with the model's own neighbourhood.
    """
    return graph.in_degree_batch(preds.t_end)


def group_metrics(preds: Predictions, groups: np.ndarray,
                  bins: tuple[int, ...] = (0, 1, 2, 3),
                  labels: tuple[str, ...] | None = None) -> pd.DataFrame:
    """MAE/RMSE/R2/n per group of the in-degree (``len(bins)-1`` buckets).

    ``bins=(0, 1, 2, 3)`` yields ``0``, ``1``, ``2``, ``>=3`` upwind neighbours.
    """
    err, _, _ = error_arrays(preds)
    labels = labels or tuple([f"{bins[i]}-{bins[i + 1] - 1}" if bins[i + 1] - bins[i] > 1
                              else f"{bins[i]}" for i in range(len(bins) - 1)]
                             + [f">={bins[-1]}"])
    rows = []
    edges = list(bins) + [np.inf]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (groups >= lo) & (groups < hi)
        rows.append({"group": labels[i], "lo": lo, "hi": hi,
                     **metric_row(np.where(sel, err, np.nan),
                                  np.where(sel, preds.true, np.nan))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Paired directional test (H3)
# --------------------------------------------------------------------------- #

@dataclass
class PairedResult:
    """Outcome of a paired comparison, with the test that supports it."""

    n_pairs: int
    mean_delta: float          # error(partner | upwind hidden) - error(partner | downwind hidden)
    median_delta: float
    stat: float
    p_value: float
    effect_size: float         # Cohen's d on the paired differences
    table: pd.DataFrame

    def summary(self) -> dict:
        return {
            "n_pairs": self.n_pairs, "mean_delta_kW": self.mean_delta,
            "median_delta_kW": self.median_delta, "wilcoxon_stat": self.stat,
            "p_value": self.p_value, "cohens_d": self.effect_size,
        }


def paired_direction_test(deltas: np.ndarray, alpha: float = 0.05) -> PairedResult:
    """Wilcoxon signed-rank test on paired error differences.

    ``deltas`` holds, per pair, the **degradation asymmetry**

    ``[err_j(hide i) - err_j(complete)] - [err_i(hide j) - err_i(complete)]``

    for a directed pair ``i -> j``. Using *degradations* rather than raw errors is what makes
    the test valid: the two conditions necessarily score different turbines, so comparing error
    levels would just measure which turbine is harder to forecast. A positive mean means losing
    the **upwind** turbine hurts its downwind neighbour more - H3's prediction.
    """
    from scipy import stats

    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 5:
        return PairedResult(len(d), float(d.mean()) if d.size else np.nan, np.nan,
                            np.nan, np.nan, np.nan, pd.DataFrame())
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            import warnings
            with warnings.catch_warnings():
                # scipy warns when the signed-rank statistic is degenerate (e.g. all deltas
                # equal); the p-value is meaningless there and reported as NaN below
                warnings.simplefilter("ignore", RuntimeWarning)
                res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        stat, p = float(res.statistic), float(res.pvalue)
        if not np.isfinite(p):
            stat, p = np.nan, np.nan
    except ValueError:            # all differences are zero
        stat, p = np.nan, np.nan
    sd = d.std(ddof=1)
    return PairedResult(
        n_pairs=int(d.size),
        mean_delta=float(d.mean()),
        median_delta=float(np.median(d)),
        stat=stat, p_value=p,
        effect_size=float(d.mean() / sd) if sd > 0 else np.nan,
        table=pd.DataFrame({"delta_kW": d}),
    )


def partner_error(preds: Predictions, partner: np.ndarray) -> np.ndarray:
    """**Absolute** error at one node per sample (``partner`` holds 0-based node indices).

    Absolute, because a paired asymmetry test compares magnitudes: a signed error would let
    over- and under-prediction cancel and could produce a negative "MAE".
    """
    _, abs_err, _ = error_arrays(preds)
    return abs_err[np.arange(len(partner)), np.asarray(partner, dtype=int)]


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def safe_mean(values) -> float:
    """``nanmean`` that returns NaN instead of warning when everything is NaN."""
    a = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def fmt(value, digits: int = 3) -> str:
    """Format a float for printing, with a readable placeholder for NaN."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:.{digits}f}" if np.isfinite(v) else "n/a"


def nan_to_none(obj):
    """Recursively replace NaN/inf by ``None`` so the result is valid JSON."""
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# Multi-seed summaries
# --------------------------------------------------------------------------- #

def seed_summary(rows: pd.DataFrame, by: tuple[str, ...] = ("model",),
                 metric: str = "MAE") -> pd.DataFrame:
    """Mean, standard deviation and range of a metric across seeds."""
    by = tuple(c for c in by if c in rows.columns) or ("model",)
    g = rows.groupby(list(by), observed=True)[metric]
    out = g.agg(mean="mean", std="std", min="min", max="max", n_seeds="count").reset_index()
    out["mean_pm_std"] = out.apply(
        lambda r: f"{r['mean']:.2f} ± {0.0 if pd.isna(r['std']) else r['std']:.2f}", axis=1)
    return out.sort_values("mean").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Persistence floor for the same windows (a reference every model must beat)
# --------------------------------------------------------------------------- #

def persistence_predictions(preds: Predictions, fd: T.FeatureData,
                            input_len: int = 12) -> Predictions:
    """Persistence floor on the same windows, respecting the silent-turbine mask.

    For a reporting turbine this is its own last observed power (the standard persistence
    forecast). For a **silent** turbine that value is exactly what the model is denied, so the
    floor falls back to the mean of the last observed values of the *visible* turbines at that
    timestamp. Using its own history here would hand the baseline information the models never
    see and make the comparison meaningless.
    """
    lookback = int(input_len)
    t_end = preds.t_end
    patv = fd.patv
    observed = fd.observed
    n = fd.n_turbines
    pred = np.full((len(t_end), n), np.nan, dtype=np.float32)
    for k, t in enumerate(t_end):
        t = int(t)
        start = max(t - lookback, 0)
        window = patv[start:t + 1]                 # (<=lookback+1, N)
        obs = observed[start:t + 1]
        vals = np.where(obs, window, np.nan)
        # last observed value in the window (forward fill over time)
        idx = np.where(np.isfinite(vals), np.arange(len(vals))[:, None], -1)
        last = idx.max(axis=0)
        cols = np.arange(n)
        got = vals[last, cols]
        hid = preds.hidden[k]
        if hid.any():
            visible = np.where(~hid & np.isfinite(got), got, np.nan)
            fill = (np.nanmean(visible) if np.isfinite(visible).any()
                    else np.nanmean(got) if np.isfinite(got).any() else np.nan)
            got = np.where(hid, fill, got)
        pred[k] = got
    return preds.copy_with(pred=pred)
