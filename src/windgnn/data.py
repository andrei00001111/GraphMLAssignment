"""Loading, time-indexing, cleaning and record flagging for the SDWPF dataset.

The 245-day / 134-turbine KDD Cup variant is the modelling dataset for this project.
It stores ``Day`` (1..245) and ``Tmstamp`` (``HH:MM``, 144 steps/day) but **no calendar
date**; see :data:`windgnn.config.RELATIVE_ANCHOR` for why a relative index is used.

Record validity follows the dataset's own rules (Zhou et al., *Scientific Data* 11:649,
"Usage Notes"):

===============  =====================================================================
category         rule
===============  =====================================================================
zero             ``Patv < 0`` (auxiliary loads) -> clipped to 0
missing          sensor value not collected (NaN)
unknown          ``Patv <= 0 and Wspd > 2.5``, or any ``Pab_i > 89`` (turbine at rest)
abnormal         ``Ndir`` outside [-720, 720] or ``Wdir`` outside [-180, 180]
===============  =====================================================================

The three validity categories are kept as separate boolean columns because later stages
use them differently: ``target_valid`` masks the metrics, while ``is_missing`` also
describes input availability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# Location / geometry
# --------------------------------------------------------------------------- #

def load_location(with_elevation: bool = True, path: Path | None = None) -> pd.DataFrame:
    """Turbine relative coordinates (metres) and, if available, elevation.

    Returns a frame indexed by ``TurbID`` (1..134) with columns ``x``, ``y`` and
    optionally ``Ele``.
    """
    if path is None:
        path = C.LOCATION_FULL if with_elevation else C.LOCATION_245
    loc = pd.read_csv(path, encoding="utf-8-sig")
    loc.columns = [c.strip() for c in loc.columns]
    loc = loc.sort_values("TurbID").set_index("TurbID")
    if not with_elevation and "Ele" in loc.columns:
        loc = loc.drop(columns=["Ele"])
    return loc


# --------------------------------------------------------------------------- #
# SCADA table
# --------------------------------------------------------------------------- #

def load_scada(
    turbine_ids=None,
    days=None,
    columns=None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load (a slice of) the 245-day SCADA table.

    Parameters
    ----------
    turbine_ids, days:
        Optional filters applied after parsing. Rows are stored turbine-major
        (all 35 280 timestamps of turbine 1, then turbine 2, ...), so filtering on
        ``days`` still reads the whole file -- use it for convenience, not for speed.
    columns:
        Optional subset of columns to keep (``TurbID``/``Day``/``Tmstamp`` are always kept).
    """
    path = path or C.SCADA_245
    parse = None
    if columns is not None:
        parse = [c for c in columns if c not in ("TurbID", "Day", "Tmstamp")]
    df = pd.read_csv(path, usecols=columns)
    if turbine_ids is not None:
        df = df[df["TurbID"].isin(np.atleast_1d(turbine_ids))]
    if days is not None:
        lo, hi = (np.atleast_1d(days).min(), np.atleast_1d(days).max())
        df = df[(df["Day"] >= lo) & (df["Day"] <= hi)]
    return df.reset_index(drop=True)


def step_index(tmstamp: pd.Series) -> np.ndarray:
    """Map ``HH:MM`` strings to a 0..143 step index within the day."""
    minutes = tmstamp.str.slice(0, 2).astype(np.int16) * 60 + tmstamp.str.slice(3, 5).astype(np.int16)
    return (minutes // C.SAMPLE_MINUTES).to_numpy()


def add_time_index(df: pd.DataFrame, anchor: str | None = None) -> pd.DataFrame:
    """Add ``t`` (global 0-based 10-minute step) and a *relative* ``timestamp``.

    ``timestamp`` is anchored at :data:`windgnn.config.RELATIVE_ANCHOR`; the KDD Cup
    release does not document the real calendar period, so treat it as a label only.
    """
    out = df.copy()
    out["step"] = step_index(out["Tmstamp"])
    out["t"] = (out["Day"].to_numpy() - 1) * C.STEPS_PER_DAY + out["step"].to_numpy()
    out["timestamp"] = pd.Timestamp(anchor or C.RELATIVE_ANCHOR) + pd.to_timedelta(out["t"] * C.SAMPLE_MINUTES, unit="m")
    return out


# --------------------------------------------------------------------------- #
# Wind direction conventions
# --------------------------------------------------------------------------- #

def wind_direction(df: pd.DataFrame, convention: str | None = None) -> np.ndarray:
    """Absolute wind direction (degrees from true north, the direction wind comes FROM).

    ``Wdir`` in SDWPF is *relative*: the angle between the wind direction and the
    nacelle direction. The absolute direction therefore has to combine ``Ndir`` and
    ``Wdir``; :func:`test_wind_direction_conventions` picks between the candidates.
    """
    convention = convention or C.WIND_DIRECTION_CONVENTION
    ndir = df["Ndir"].to_numpy(dtype=np.float64)
    wdir = df["Wdir"].to_numpy(dtype=np.float64)
    if convention == "rel_sum":
        abs_dir = ndir + wdir
    elif convention == "rel_diff":
        abs_dir = ndir - wdir
    elif convention == "absolute":
        abs_dir = wdir
    else:  # pragma: no cover - guarded by the candidate dict
        raise ValueError(f"unknown wind-direction convention: {convention!r}")
    return np.mod(abs_dir, 360.0)


def mean_resultant_length(angles_deg: np.ndarray) -> float:
    """Circular concentration in [0, 1]: 1 = all angles identical."""
    good = np.isfinite(angles_deg)
    if good.sum() < 2:
        return np.nan
    rad = np.deg2rad(angles_deg[good])
    return float(np.hypot(np.cos(rad).mean(), np.sin(rad).mean()))


def circular_mean(angles_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees, returned in [0, 360)."""
    a = np.asarray(angles_deg, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    rad = np.deg2rad(a)
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def circular_std(angles_deg: np.ndarray) -> float:
    """Circular standard deviation in degrees (0 = perfectly concentrated)."""
    r = mean_resultant_length(np.asarray(angles_deg, dtype=np.float64))
    if not np.isfinite(r) or r <= 0:
        return np.nan
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(min(max(r, 1e-12), 1.0)))))


def circular_median(angles_deg: np.ndarray) -> float:
    """Circular median in degrees, returned in [0, 360)."""
    a = np.asarray(angles_deg, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    c = circular_mean(a)
    return float(np.mod(c + np.median(wrap180(a - c)), 360.0))


def wrap180(angles_deg):
    """Wrap angles to (-180, 180]."""
    return (np.asarray(angles_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def wind_direction_diagnostics(
    df: pd.DataFrame,
    wspd_min: float = 4.0,
    max_timestamps: int | None = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare the candidate ways of reconstructing the absolute wind direction.

    ``Wdir`` is documented as the angle *between* the wind direction and the nacelle
    direction, so the absolute direction should come from ``Ndir`` and ``Wdir`` together.
    The test must avoid a trap: a column that is nearly constant (such as ``Wdir``, which
    sits near 0 whenever turbines are yawed into the wind) scores a perfect cross-turbine
    concentration *trivially*. Three metrics are therefore reported:

    ``cross_turbine_R``
        Mean per-snapshot circular concentration across turbines. A genuine wind
        direction is nearly identical farm-wide, so higher is better.
    ``temporal_R``
        Circular concentration of the per-snapshot farm mean over the whole period. A
        real wind direction follows the weather and varies a lot over 245 days, so
        *lower* is better. A near-constant (degenerate) column gives ~1 and is rejected.
    ``cross_turbine_R_calibrated``
        Same as ``cross_turbine_R`` after removing each turbine's own stable bias. In
        SDWPF a minority of nacelle bearings carry a constant offset, which depresses
        the raw figure; this column shows how much structure remains.

    Decision rule: reject candidates with ``temporal_R`` above ~0.5 (not weather-like),
    then prefer the highest calibrated coherence.
    """
    use = df if "t" in df.columns else add_time_index(df)
    if "target_valid" in use.columns:
        use = use[use["target_valid"]]
    use = use[use["Wspd"] >= wspd_min]

    ts = np.sort(use["t"].unique())
    if max_timestamps is not None and len(ts) > max_timestamps:
        keep = np.random.default_rng(seed).choice(ts, size=max_timestamps, replace=False)
        use = use[use["t"].isin(keep)]

    t = use["t"].to_numpy()
    turb = use["TurbID"].to_numpy()
    rows = []
    for name, description in C.WIND_DIRECTION_CANDIDATES.items():
        cand = wind_direction(use, name)
        tmp = pd.DataFrame({"t": t, "TurbID": turb, "ang": cand})
        per_snapshot = tmp.groupby("t", observed=True)["ang"].apply(
            lambda s: mean_resultant_length(s.to_numpy())
        )
        consensus = tmp.groupby("t", observed=True)["ang"].apply(
            lambda s: circular_median(s.to_numpy())
        )

        dev = pd.DataFrame({"TurbID": turb, "dev": wrap180(cand - consensus.reindex(t).to_numpy())})
        offsets = dev.groupby("TurbID", observed=True)["dev"].median()
        corrected = np.mod(cand - offsets.reindex(turb).to_numpy(), 360.0)
        cal = pd.DataFrame({"t": t, "ang": corrected}).groupby("t", observed=True)["ang"].apply(
            lambda s: mean_resultant_length(s.to_numpy())
        )
        rows.append({
            "convention": name,
            "description": description,
            "cross_turbine_R": float(per_snapshot.mean()),
            "temporal_R": mean_resultant_length(consensus.to_numpy()),
            "cross_turbine_R_calibrated": float(cal.mean()),
        })
    return pd.DataFrame(rows).sort_values("cross_turbine_R_calibrated", ascending=False).reset_index(drop=True)



def turbine_yaw_offsets(
    df: pd.DataFrame,
    farm_dir: pd.Series,
    convention: str | None = None,
    wspd_min: float = 4.0,
) -> pd.Series:
    """Per-turbine stable deviation between the reconstructed direction and the farm consensus.

    Returns a Series indexed by ``TurbID`` holding the circular median deviation (deg).
    A small, stable deviation means the turbine is yawed with the farm; a large one means
    its nacelle bearing carries a constant bias (or the turbine is parked at an angle).
    """
    conv = convention or C.WIND_DIRECTION_CONVENTION
    if "t" not in df.columns:
        df = add_time_index(df)
    use = df[(df["Wspd"] >= wspd_min)]
    if "target_valid" in use.columns:
        use = use[use["target_valid"]]
    cand = wind_direction(use, conv)
    ref = farm_dir.reindex(use["t"].to_numpy()).to_numpy()
    dev = pd.DataFrame({"TurbID": use["TurbID"].to_numpy(), "dev": wrap180(cand - ref)})
    return dev.groupby("TurbID", observed=True)["dev"].apply(
        lambda s: float(np.median(s.to_numpy()))
    )


def estimate_farm_wind_direction(
    df: pd.DataFrame,
    convention: str | None = None,
    wspd_min: float = 4.0,
    calibrate: bool = True,
    offsets: pd.Series | None = None,
    n_iter: int = 3,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate one absolute wind direction per 10-minute snapshot.

    Turbines yaw into the wind, but a sizeable minority of nacelle bearings in SDWPF carry
    a *stable* per-turbine bias (see :func:`turbine_yaw_offsets`). The estimator therefore
    alternates between a robust per-snapshot consensus and a per-turbine offset estimate.

    The global rotation of the result is unidentifiable from ``Ndir``/``Wdir`` alone: only
    the direction *relative* to the farm layout matters for the wind-following graph, and
    the farm's own axis orientation is estimated separately in Stage B. Pass ``offsets``
    fitted on the training split to avoid leakage when evaluating on val/test.

    Returns
    -------
    (farm_dir, offsets)
        ``farm_dir`` is a DataFrame indexed by ``t`` with columns ``farm_dir``
        (degrees, arbitrary global rotation), ``n_turbines`` and ``coherence``
        (cross-turbine concentration after calibration). ``offsets`` is the per-turbine
        bias table actually used (empty if ``calibrate`` is False).
    """
    conv = convention or C.WIND_DIRECTION_CONVENTION
    if "t" not in df.columns:
        df = add_time_index(df)
    use = df[df["Wspd"] >= wspd_min]
    if "target_valid" in use.columns:
        use = use[use["target_valid"]]
    use = use[use["Ndir"].notna() & use["Wdir"].notna()]

    cand = wind_direction(use, conv)
    t = use["t"].to_numpy()
    turb = use["TurbID"].to_numpy()

    # Robust starting point: circular median over turbines, per snapshot.
    tmp = pd.DataFrame({"t": t, "ang": cand})
    farm_dir = tmp.groupby("t", observed=True)["ang"].apply(
        lambda s: circular_median(s.to_numpy())
    )

    used_offsets = pd.Series(dtype=float)
    if calibrate:
        for _ in range(n_iter):
            if offsets is None:
                dev = pd.DataFrame({"TurbID": turb, "dev": wrap180(cand - farm_dir.reindex(t).to_numpy())})
                used_offsets = dev.groupby("TurbID", observed=True)["dev"].apply(
                    lambda s: float(np.median(s.to_numpy()))
                )
            else:
                used_offsets = offsets
            corrected = np.mod(cand - used_offsets.reindex(turb).to_numpy(), 360.0)
            farm_dir = pd.DataFrame({"t": t, "ang": corrected}).groupby("t", observed=True)["ang"].apply(
                lambda s: circular_mean(s.to_numpy())
            )
            if offsets is not None:
                break

    # Quality per snapshot: concentration of the (corrected) candidate across turbines.
    final = np.mod(cand - (used_offsets.reindex(turb).to_numpy() if len(used_offsets) else 0.0), 360.0)
    quality = pd.DataFrame({"t": t, "ang": final}).groupby("t", observed=True)["ang"].agg(
        n_turbines="count",
        coherence=lambda s: mean_resultant_length(s.to_numpy()),
    )
    out = pd.concat([farm_dir.rename("farm_dir"), quality], axis=1)
    out.index.name = "t"
    return out, used_offsets



def add_derived(df: pd.DataFrame, convention: str | None = None) -> pd.DataFrame:
    """Add absolute wind direction, its sin/cos encodings and the flow direction."""
    out = df.copy()
    abs_from = wind_direction(out, convention)
    out["wind_dir_abs"] = abs_from                       # direction the wind comes FROM
    out["wind_dir_flow"] = np.mod(abs_from + 180.0, 360.0)  # direction the wind blows TOWARDS
    rad_from = np.deg2rad(abs_from)
    rad_flow = np.deg2rad(out["wind_dir_flow"].to_numpy())
    out["wind_from_sin"] = np.sin(rad_from)
    out["wind_from_cos"] = np.cos(rad_from)
    out["wind_flow_sin"] = np.sin(rad_flow)
    out["wind_flow_cos"] = np.cos(rad_flow)
    # Wind-speed vector components (paper 1 uses u/v decomposition as node features).
    wspd = out["Wspd"].to_numpy(dtype=np.float64)
    out["wspd_u"] = wspd * out["wind_from_sin"].to_numpy()
    out["wspd_v"] = wspd * out["wind_from_cos"].to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Record flagging
# --------------------------------------------------------------------------- #

FLAG_COLUMNS = ["is_missing", "is_zero", "is_unknown", "is_abnormal", "target_valid"]


def flag_records(df: pd.DataFrame, clip_negative_power: bool = True) -> pd.DataFrame:
    """Apply the dataset's validity rules and add one boolean column per category.

    ``is_missing`` describes the whole record (any measurement absent), whereas
    ``target_valid`` is specifically about whether ``Patv`` may be scored.
    """
    out = df.copy()

    out["is_missing"] = out[C.MEASUREMENT_COLS].isna().any(axis=1)
    out["is_zero"] = out[C.TARGET_COL].notna() & (out[C.TARGET_COL] < 0)

    patv = out[C.TARGET_COL]
    wspd = out["Wspd"]
    stopped_low = (patv <= 0) & (wspd > C.UNKNOWN_WSPD_THRESHOLD)
    feathered = (out[["Pab1", "Pab2", "Pab3"]] > C.PITCH_UNKNOWN_THRESHOLD).any(axis=1)
    out["is_unknown"] = (stopped_low | feathered).fillna(False)

    ndir_bad = out["Ndir"].notna() & (
        (out["Ndir"] > C.NDIR_ABNORMAL_RANGE[1]) | (out["Ndir"] < C.NDIR_ABNORMAL_RANGE[0])
    )
    wdir_bad = out["Wdir"].notna() & (
        (out["Wdir"] > C.WDIR_ABNORMAL_RANGE[1]) | (out["Wdir"] < C.WDIR_ABNORMAL_RANGE[0])
    )
    out["is_abnormal"] = ndir_bad | wdir_bad

    out["target_valid"] = ~(
        out[C.TARGET_COL].isna() | out["is_unknown"] | out["is_abnormal"]
    )

    if clip_negative_power:
        out[C.TARGET_COL] = out[C.TARGET_COL].clip(lower=0.0)

    return out


def clean(df: pd.DataFrame, convention: str | None = None) -> pd.DataFrame:
    """Full Stage-A preparation: time index -> derived wind features -> validity flags."""
    out = df if "t" in df.columns else add_time_index(df)
    out = flag_records(out)
    out = add_derived(out, convention=convention)
    return out


# --------------------------------------------------------------------------- #
# Cached analysis table
# --------------------------------------------------------------------------- #

def cache_path(convention: str | None = None) -> Path:
    convention = convention or C.WIND_DIRECTION_CONVENTION
    return C.INTERIM_DIR / f"scada_245d_{convention}.parquet"


def build_analysis_table(
    force: bool = False,
    convention: str | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the SCADA table, clean it and cache the result as parquet."""
    convention = convention or C.WIND_DIRECTION_CONVENTION
    out_path = cache_path(convention) if path is None else path
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    df = load_scada()
    df = clean(df, convention=convention)
    df["TurbID"] = df["TurbID"].astype(np.int16)
    df["Day"] = df["Day"].astype(np.int16)
    df["t"] = df["t"].astype(np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_analysis_table(force: bool = False, convention: str | None = None) -> pd.DataFrame:
    """Return the cleaned 4.7M-row analysis table (built once, then cached)."""
    return build_analysis_table(force=force, convention=convention)


# --------------------------------------------------------------------------- #
# Convenience views
# --------------------------------------------------------------------------- #

def power_wide(df: pd.DataFrame, column: str = "Patv", valid_only: bool = False) -> pd.DataFrame:
    """Pivot a per-turbine column into a ``time x turbine`` matrix."""
    use = df
    if valid_only:
        use = df[df["target_valid"]]
    wide = use.pivot_table(index="t", columns="TurbID", values=column, aggfunc="first")
    return wide.reindex(columns=np.arange(1, C.N_TURBINES + 1))


def split_bounds(name: str) -> tuple[int, int]:
    """First and last global step index of a named split (``train``/``val``/``test``)."""
    lo_day, hi_day = C.SPLIT_DAYS[name]
    return (lo_day - 1) * C.STEPS_PER_DAY, hi_day * C.STEPS_PER_DAY - 1


def split_labels(t: np.ndarray) -> np.ndarray:
    """Map global step indices to split labels."""
    labels = np.full(len(t), "none", dtype=object)
    for name in C.SPLIT_DAYS:
        lo, hi = split_bounds(name)
        labels[(t >= lo) & (t <= hi)] = name
    return labels
