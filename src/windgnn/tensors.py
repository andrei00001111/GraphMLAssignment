"""Stage D: feature tensors, sliding windows, graph providers and input masking.

Everything downstream of Stage C needs the same three objects:

* a **feature tensor** ``(T, N, F)`` built once from the cleaned analysis table and
  standardised with training-split statistics only;
* **window samples** ``(t_end, h)`` per split, with a purge gap so no window straddles a
  split boundary;
* a **graph for every timestamp** ``G_t``, rebuilt from the wind direction (or taken from a
  static baseline graph), delivered as ``(src, dst, edge_attr, mask)``.

The graph representation is deliberately "one topology, per-sample attributes/mask": all
turbine pairs inside ``R`` form a fixed candidate set (Stage B precomputed it), and a
timestamp only changes the *edge attributes* (``dist/R``, ``along``, ``cross``,
``|align|/90``) plus, for the hard-cone ablations, an edge **mask**. That keeps batching
trivial (one edge list for the whole batch) while still expressing a graph that changes
every 10 minutes - and it makes the hard cone an ablation rather than a separate code path.

Masking
-------
Two masks must never be conflated:

``hidden`` (per sample, per turbine)
    the turbine reports nothing for the whole input window: all its features are zeroed and
    the ``observed`` flag is set to 0. This is what the proposal means by a silent turbine.
``valid`` (per timestamp, per turbine)
    the ground truth may be scored. Applied only by the metric code.

See PLAN.md sections 5.2 and 11.4 for why the soft cone (all pairs within ``R``, direction as
an edge attribute) is the primary configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C
from . import data as D
from . import graphs as G
from . import imputation as I

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

#: ``(name, source column, kind)``. ``kind`` drives standardisation:
#: ``"value"`` -> z-score, ``"angle"`` -> already sin/cos, left as is, ``"flag"`` -> 0/1.
FEATURES: tuple[tuple[str, str, str], ...] = (
    ("patv", "Patv", "value"),          # the turbine's own recent power (0 when hidden)
    ("wspd", "Wspd", "value"),
    ("wdir_sin", "Wdir_sin", "angle"),  # relative wind direction, sin/cos encoded
    ("wdir_cos", "Wdir_cos", "angle"),
    ("etmp", "Etmp", "value"),
    ("itmp", "Itmp", "value"),
    ("ndir_sin", "Ndir_sin", "angle"),
    ("ndir_cos", "Ndir_cos", "angle"),
    ("pab1", "Pab1", "value"),
    ("prtv", "Prtv", "value"),
    ("farm_sin", "farm_sin", "angle"),  # farm-level absolute wind direction (shared)
    ("farm_cos", "farm_cos", "angle"),
    ("tod_sin", "tod_sin", "angle"),
    ("tod_cos", "tod_cos", "angle"),
    ("observed", "observed", "flag"),   # 1 when the turbine reported at this timestamp
)

FEATURE_NAMES: tuple[str, ...] = tuple(f[0] for f in FEATURES)

#: Plausible physical ranges for the SCADA measurements, in the dataset's units.
#: The dataset's own "abnormal" rule only covers ``Ndir``/``Wdir`` (paper 3, Usage Notes); in
#: practice the temperature channels also carry sentinel values (``Etmp`` reaches -273 C and
#: +394 C). Such records are treated as *unobserved inputs* here. This is a modelling
#: decision about feature sanitation - it does not change the dataset's validity flags, which
#: remain the scoring mask.
PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "Wspd": (0.0, 40.0),
    "Etmp": (-50.0, 60.0),
    "Itmp": (-50.0, 80.0),
    "Pab1": (-15.0, 100.0),
    "Prtv": (-1000.0, 1000.0),
    "Patv": (-100.0, 1700.0),
}


@dataclass
class FeatureData:
    """Standardised features plus everything needed to build targets and graphs."""

    x: np.ndarray                 # (T, N, F) float32, standardised
    names: list[str]
    patv: np.ndarray              # (T, N) float32, kW, NaN where missing
    valid: np.ndarray             # (T, N) bool, ground truth may be scored
    observed: np.ndarray          # (T, N) bool, a record exists
    t_index: np.ndarray           # (T,) global 10-minute step
    farm_dir: np.ndarray          # (T,) absolute wind direction the wind comes FROM
    turbine_ids: np.ndarray       # (N,) 1..134
    loc: pd.DataFrame             # geometry, indexed by TurbID
    stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_turbines(self) -> int:
        return self.x.shape[1]

    @property
    def n_steps(self) -> int:
        return self.x.shape[0]

    def feature_index(self, name: str) -> int:
        return self.names.index(name)


def _to_matrix(values: np.ndarray, n_turbines: int) -> np.ndarray:
    """``(N*T,)`` turbine-major -> ``(T, N)``."""
    return values.reshape(n_turbines, -1).T


def build_feature_data(
    df: pd.DataFrame | None = None,
    farm_dir: pd.DataFrame | None = None,
    loc: pd.DataFrame | None = None,
) -> FeatureData:
    """Build the standardised feature tensor from the Stage-A analysis table.

    Parameters
    ----------
    df:
        Cleaned analysis table (``data.load_analysis_table()``). Loaded if omitted.
    farm_dir:
        Farm wind direction, indexed by ``t`` (``data/interim/farm_wind_direction.csv``).
        Loaded if omitted. Used for the graph and for the two farm-level features.
    loc:
        Turbine coordinates; loaded if omitted.

    Notes
    -----
    Standardisation uses the mean/std of the **observed** entries of the training split only.
    ``patv`` is stored in kW in ``FeatureData.patv``; its standardised version in ``x`` is what
    the model sees, and predictions are converted back with ``stats``.
    """
    df = D.load_analysis_table() if df is None else df
    if loc is None:
        loc = D.load_location()
    if farm_dir is None:
        farm_dir = pd.read_csv(C.INTERIM_DIR / "farm_wind_direction.csv", index_col=0)
    if not I.is_turbine_major_grid(df):
        raise ValueError("build_feature_data expects the full turbine-major 134 x 35 280 grid")

    n_t = C.N_STEPS
    n_n = C.N_TURBINES
    t_index = np.arange(n_t, dtype=np.int32)

    def mat(col: str) -> np.ndarray:
        return _to_matrix(df[col].to_numpy(), n_n).astype(np.float32)

    raw = {col: mat(col) for col in
           ["Patv", "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Prtv", "target_valid"]}
    patv_raw = raw["Patv"]

    # ---- input availability -------------------------------------------------------
    # A record is "observed" (usable as a model input) when it was collected, is not flagged
    # abnormal by the dataset's own rule, and every measurement lies inside its physical
    # range. This is deliberately *not* the same mask as ``valid`` (which is about whether
    # Patv may be scored): a stopped turbine still reports wind speed.
    present = (~mat("is_missing").astype(bool)) & (~mat("is_abnormal").astype(bool))
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        arr = raw[col]
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(arr) & (arr >= lo) & (arr <= hi)
        present &= ok
        raw[col] = np.where(ok, arr, np.nan).astype(np.float32)
    observed = present

    # Targets: the dataset's validity rule *and* a plausible physical value.
    patv = raw["Patv"]
    valid = mat("target_valid").astype(bool) & np.isfinite(patv)

    # farm-level direction, broadcast over turbines
    dirs = farm_dir.reindex(t_index)["farm_dir"].to_numpy(dtype=np.float64)
    finite = np.isfinite(dirs)
    if not finite.all():             # fill gaps with the circular mean of the rest
        fill = D.circular_mean(dirs[finite])
        dirs = np.where(finite, dirs, fill)

    tod = 2.0 * np.pi * (t_index % C.STEPS_PER_DAY) / C.STEPS_PER_DAY
    columns: dict[str, np.ndarray] = {
        "Patv": patv,
        "Wspd": raw["Wspd"],
        "Wdir_sin": np.sin(np.deg2rad(raw["Wdir"])),
        "Wdir_cos": np.cos(np.deg2rad(raw["Wdir"])),
        "Etmp": raw["Etmp"],
        "Itmp": raw["Itmp"],
        "Ndir_sin": np.sin(np.deg2rad(raw["Ndir"])),
        "Ndir_cos": np.cos(np.deg2rad(raw["Ndir"])),
        "Pab1": raw["Pab1"],
        "Prtv": raw["Prtv"],
        "farm_sin": np.sin(np.deg2rad(dirs))[:, None].repeat(n_n, axis=1).astype(np.float32),
        "farm_cos": np.cos(np.deg2rad(dirs))[:, None].repeat(n_n, axis=1).astype(np.float32),
        "tod_sin": np.sin(tod)[:, None].repeat(n_n, axis=1).astype(np.float32),
        "tod_cos": np.cos(tod)[:, None].repeat(n_n, axis=1).astype(np.float32),
        "observed": observed.astype(np.float32),
    }

    # ---- standardisation statistics from the training split, observed entries only ----
    tr_lo, tr_hi = D.split_bounds("train")
    train_rows = (t_index >= tr_lo) & (t_index <= tr_hi)
    stats_rows = []
    x = np.zeros((n_t, n_n, len(FEATURES)), dtype=np.float32)
    for k, (name, source, kind) in enumerate(FEATURES):
        arr = columns[source]
        if kind == "value":
            # the target's scale is defined on scoreable records; every other feature on the
            # records the model will actually see
            usable = valid if name == "patv" else observed
            obs = arr[train_rows][usable[train_rows]]
            mean = float(np.nanmean(obs)) if obs.size else 0.0
            std = float(np.nanstd(obs)) if obs.size else 1.0
        else:
            mean, std = 0.0, 1.0
        std = std if std > 1e-8 else 1.0
        if kind == "flag":
            mean, std = 0.0, 1.0
        vals = (arr - mean) / std
        vals[~np.isfinite(vals)] = 0.0          # NaN -> 0 == the feature mean
        x[:, :, k] = vals.astype(np.float32)
        stats_rows.append({"feature": name, "mean": mean, "std": std, "kind": kind})

    return FeatureData(
        x=x,
        names=list(FEATURE_NAMES),
        patv=patv,
        valid=valid,
        observed=observed,
        t_index=t_index,
        farm_dir=dirs.astype(np.float32),
        turbine_ids=np.arange(1, n_n + 1, dtype=np.int16),
        loc=loc,
        stats=pd.DataFrame(stats_rows),
    )


def standardize_target(fd: FeatureData, values: np.ndarray) -> np.ndarray:
    """Map raw kW values to the model's standardised target scale."""
    row = fd.stats.set_index("feature").loc["patv"]
    return (values - row["mean"]) / row["std"]


def inverse_target(fd: FeatureData, values: np.ndarray) -> np.ndarray:
    """Map standardised predictions back to kW."""
    row = fd.stats.set_index("feature").loc["patv"]
    return values * row["std"] + row["mean"]


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WindowSpec:
    """Input window length and forecast horizons, in 10-minute steps."""

    input_len: int = 12                       # 2 h of history
    horizons: tuple[int, ...] = (1, 6, 36)    # 10 min, 1 h, 6 h

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def purge(self) -> int:
        """Steps of separation kept between splits (a window plus the longest horizon)."""
        return self.input_len + self.max_horizon


def sample_indices(
    fd: FeatureData,
    spec: WindowSpec,
    split: str,
    stride: int = 1,
    horizons: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(t_end, h)`` pairs whose window and target both live inside ``split``.

    The window is ``[t_end - input_len + 1, t_end]`` and the target is ``t_end + h``, so a
    sample never sees across a split boundary and never predicts into another split.
    """
    horizons = tuple(spec.horizons if horizons is None else horizons)
    lo, hi = D.split_bounds(split)
    t_end_all, h_all = [], []
    for h in horizons:
        first = lo + spec.input_len - 1
        last = hi - h
        if last < first:
            continue
        t_end = np.arange(first, last + 1, stride, dtype=np.int32)
        t_end_all.append(t_end)
        h_all.append(np.full(t_end.shape, h, dtype=np.int16))
    if not t_end_all:
        return np.empty(0, np.int32), np.empty(0, np.int16)
    return np.concatenate(t_end_all), np.concatenate(h_all)


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #

def hidden_none(n_turbines: int = C.N_TURBINES) -> np.ndarray:
    """Nobody is hidden (the complete-data reference)."""
    return np.zeros(n_turbines, dtype=bool)


def hidden_turbines(ids, n_turbines: int = C.N_TURBINES) -> np.ndarray:
    """Hide the given turbine ids (1-based)."""
    mask = np.zeros(n_turbines, dtype=bool)
    mask[np.atleast_1d(ids).astype(int) - 1] = True
    return mask


class RandomTurbineMasker:
    """Training-time augmentation: hide each turbine independently with probability ``p``.

    ``p`` is drawn per sample from ``U(0, p_max)`` so the model sees the whole range from no
    outage to a large outage (PLAN.md 5.2). The draw is a deterministic function of
    ``(seed, sample index, epoch)``, so runs are reproducible without relying on DataLoader
    worker state.
    """

    def __init__(self, p_max: float = 0.5, seed: int = 0, n_turbines: int = C.N_TURBINES,
                 min_p: float = 0.0):
        self.p_max = float(p_max)
        self.min_p = float(min_p)
        self.seed = int(seed)
        self.n = int(n_turbines)

    def __call__(self, index: int, epoch: int = 0) -> np.ndarray:
        rng = np.random.default_rng((self.seed, int(epoch), int(index)))
        p = rng.uniform(self.min_p, self.p_max) if self.p_max > 0 else 0.0
        return rng.random(self.n) < p


# --------------------------------------------------------------------------- #
# Outage scenarios
# --------------------------------------------------------------------------- #

def outage_masks(
    n_samples: int,
    n_turbines: int = C.N_TURBINES,
    share: float = 0.0,
    mode: str = "random",
    seed: int = 0,
    ids=None,
    columns: np.ndarray | None = None,
) -> np.ndarray:
    """``(n_samples, N)`` silent-turbine masks for the proposal's outage scenarios.

    ==========  ==========================================================================
    mode        what is hidden
    ==========  ==========================================================================
    ``random``  ``round(share * N)`` turbines drawn independently for every sample
    ``one``     exactly one turbine per sample, cycling through ``ids`` (the proposal's
                "one turbine at a time")
    ``block``   a contiguous block of ``round(share * N)`` turbine ids (spatially contiguous,
                because turbine ids are assigned along the columns)
    ``column``  whole geometric columns (the worst case: everything upwind is gone)
    ==========  ==========================================================================

    A ``share`` of 0 returns an all-False mask (the complete-data reference).
    """
    rng = np.random.default_rng(seed)
    masks = np.zeros((n_samples, n_turbines), dtype=bool)
    if mode == "one":
        ids = np.arange(1, n_turbines + 1) if ids is None else np.atleast_1d(ids)
        masks[np.arange(n_samples), (np.asarray(ids)[np.arange(n_samples) % len(ids)] - 1)] = True
        return masks
    k = int(round(float(share) * n_turbines))
    if k <= 0:
        return masks
    k = min(k, n_turbines)
    if mode == "random":
        for i in range(n_samples):
            masks[i, rng.choice(n_turbines, size=k, replace=False)] = True
    elif mode == "block":
        start = rng.integers(0, n_turbines - k + 1, size=n_samples)
        for i in range(n_samples):
            masks[i, start[i]:start[i] + k] = True
    elif mode == "column":
        if columns is None:
            raise ValueError("mode='column' needs the column labels from graphs.detect_columns")
        order = np.argsort(columns)
        groups = np.split(order, np.flatnonzero(np.diff(np.sort(columns))) + 1)
        pick, total = [], 0
        for g in groups:                     # whole columns until the budget is met
            if total >= k:
                break
            pick.append(g)
            total += len(g)
        masks[:, np.concatenate(pick)] = True
    else:
        raise ValueError(f"unknown outage mode: {mode!r}")
    return masks


# --------------------------------------------------------------------------- #
# Graph providers
# --------------------------------------------------------------------------- #

#: Edge attribute layout produced by :class:`GraphProvider`.
EDGE_FEATURES: tuple[str, ...] = ("dist_over_r", "along", "cross", "angle_over_90")


class GraphProvider:
    """Delivers ``(src, dst, edge_attr, edge_mask)`` for a batch of timestamps.

    Parameters
    ----------
    kind:
        ``"wind"`` - candidate pairs within ``radius_m``, direction encoded in the edge
        attributes (the *soft cone*). ``"knn"`` - static k-nearest neighbours by distance.
        ``"full"`` - every ordered pair. ``"none"`` - no edges (the graph-free ablation).
    cone_deg:
        Half-angle of the wind cone used to build the per-sample edge **mask**. ``180``
        admits every candidate edge (the soft cone); a small value reproduces the hard cone
        of Stage B. Only used when ``kind="wind"``.
    reverse:
        Flip the direction convention (the ``alpha + 180`` control). Equivalent to using
        ``alpha + 180``; kept explicit because it is the decisive falsification test.
    self_loops:
        Append one self-loop per node; its edge attribute is the per-sample mean of the
        real edges (the same convention as PyG's ``fill_value="mean"``). Self-loops are
        never masked out.
    """

    def __init__(
        self,
        loc: pd.DataFrame,
        farm_dir: np.ndarray,
        t_index: np.ndarray,
        alpha_deg: float,
        radius_m: float = 1500.0,
        cone_deg: float = 180.0,
        kind: str = "wind",
        k: int = 5,
        reverse: bool = False,
        self_loops: bool = True,
        n_turbines: int = C.N_TURBINES,
    ):
        self.loc = loc
        self.farm_dir = np.asarray(farm_dir, dtype=np.float64)
        self.t_index = np.asarray(t_index)
        self.alpha_deg = float(alpha_deg) + (180.0 if reverse else 0.0)
        self.radius_m = float(radius_m)
        self.cone_deg = float(cone_deg)
        self.kind = kind
        self.k = int(k)
        self.self_loops = bool(self_loops)
        self.n = int(n_turbines)
        self._t_lookup = {int(t): i for i, t in enumerate(self.t_index)}

        self._build_topology()

    # -- topology (constant across samples) ------------------------------- #
    def _build_topology(self) -> None:
        n = self.n
        if self.kind == "none":
            self.src = np.empty(0, np.int64)
            self.dst = np.empty(0, np.int64)
            self.dist = np.empty(0, np.float64)
            self.dx = np.empty(0, np.float64)
            self.dy = np.empty(0, np.float64)
        elif self.kind == "knn":
            adj = G.static_distance_knn(self.loc, self.k)
            dist = G.pairwise_distances(self.loc)
            i_idx, j_idx = np.nonzero(adj)
            self.src, self.dst = i_idx, j_idx
            self.dist = dist[i_idx, j_idx]
            xy = G.coordinates(self.loc)
            self.dx = xy[j_idx, 0] - xy[i_idx, 0]
            self.dy = xy[j_idx, 1] - xy[i_idx, 1]
        elif self.kind == "full":
            pairs = G.candidate_pairs(self.loc, np.inf)
            self.src = pairs["i_idx"].to_numpy()
            self.dst = pairs["j_idx"].to_numpy()
            self.dist = pairs["dist"].to_numpy()
            self.dx = pairs["dx"].to_numpy()
            self.dy = pairs["dy"].to_numpy()
        elif self.kind == "wind":
            pairs = G.candidate_pairs(self.loc, self.radius_m)
            self.src = pairs["i_idx"].to_numpy()
            self.dst = pairs["j_idx"].to_numpy()
            self.dist = pairs["dist"].to_numpy()
            self.dx = pairs["dx"].to_numpy()
            self.dy = pairs["dy"].to_numpy()
        else:
            raise ValueError(f"unknown graph kind: {self.kind!r}")

        if self.self_loops and self.kind != "none":
            nodes = np.arange(n, dtype=np.int64)
            self.src = np.concatenate([self.src, nodes])
            self.dst = np.concatenate([self.dst, nodes])
            zeros = np.zeros(n, dtype=np.float64)
            self.dist = np.concatenate([self.dist, zeros])
            self.dx = np.concatenate([self.dx, zeros])
            self.dy = np.concatenate([self.dy, zeros])
            self._is_self = np.concatenate([np.zeros(len(self.src) - n, bool), np.ones(n, bool)])
        else:
            self._is_self = np.zeros(len(self.src), bool)

        self.n_edges = len(self.src)
        # normalised distance, computed once
        d = np.maximum(self.dist, 1.0)
        self._dist_unit = np.where(self._is_self, 0.0, self.dist / max(self.radius_m, 1.0))
        self._ux = np.where(self._is_self, 0.0, self.dx / d)
        self._uy = np.where(self._is_self, 0.0, self.dy / d)

    # -- per-timestamp attributes ----------------------------------------- #
    def edge_attr_for(self, t_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Edge attributes ``(B, E, 4)`` and cone mask ``(B, E)`` for the given timestamps.

        Edge ``i -> j`` exists when ``j`` is downwind of ``i``; ``along = +1`` means the pair
        axis points exactly in the flow direction. For ``kind != "wind"`` the direction
        columns are zero and the mask admits every edge.
        """
        t_ends = np.asarray(t_ends, dtype=np.int64)
        b = len(t_ends)
        rows = np.array([self._t_lookup.get(int(t), -1) for t in t_ends])
        fallback = D.circular_mean(self.farm_dir)
        dirs = np.where(rows >= 0, self.farm_dir[np.clip(rows, 0, None)], fallback)

        attrs = np.zeros((b, self.n_edges, len(EDGE_FEATURES)), dtype=np.float32)
        attrs[:, :, 0] = self._dist_unit[None, :]
        mask = np.ones((b, self.n_edges), dtype=bool)
        if self.kind != "wind" or self.n_edges == 0:
            return attrs, mask

        u = G.flow_unit_vectors(dirs, self.alpha_deg)          # (B, 2) in the farm frame
        along = self._ux[None, :] * u[:, 0:1] + self._uy[None, :] * u[:, 1:2]
        cross = self._ux[None, :] * u[:, 1:2] - self._uy[None, :] * u[:, 0:1]
        attrs[:, :, 1] = along
        attrs[:, :, 2] = cross
        attrs[:, :, 3] = np.arccos(np.clip(along, -1.0, 1.0)) / (np.pi / 2.0)
        attrs[:, self._is_self, 1] = 1.0        # a self-loop is trivially "aligned"
        attrs[:, self._is_self, 3] = 0.0

        if self.cone_deg < 180.0:
            mask = along >= np.cos(np.deg2rad(self.cone_deg))
            mask[:, self._is_self] = True
        return attrs, mask

    def batch(self, t_ends: np.ndarray, device=None) -> dict:
        """Torch tensors for a batch: ``src``/``dst`` ``(E,)``, ``edge_attr`` ``(B,E,4)``,
        ``mask`` ``(B,E)``."""
        attrs, mask = self.edge_attr_for(t_ends)
        out = {
            "src": torch.as_tensor(self.src, dtype=torch.long),
            "dst": torch.as_tensor(self.dst, dtype=torch.long),
            "edge_attr": torch.as_tensor(attrs, dtype=torch.float32),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
            "n_nodes": self.n,
        }
        if device is not None:
            out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
        return out

    def in_degree_batch(self, t_ends: np.ndarray) -> np.ndarray:
        """``(B, N)`` number of *real* neighbours each turbine receives from (self-loops excluded).

        This is the grouping variable for H2: a front-row turbine for the current wind has
        in-degree 0, a turbine deep inside the array has several upwind neighbours.
        """
        _, mask = self.edge_attr_for(t_ends)
        if self.n_edges == 0:
            return np.zeros((len(t_ends), self.n), dtype=np.int32)
        mask = mask & (~self._is_self)[None, :]
        out = np.zeros((len(t_ends), self.n), dtype=np.int32)
        for b in range(len(t_ends)):
            out[b] = np.bincount(self.dst[mask[b]], minlength=self.n)
        return out

    # -- diagnostics ------------------------------------------------------- #
    def degree(self, t_end: int | None = None) -> np.ndarray:
        """In-degree per node (self-loops excluded) at one representative timestamp."""
        t = int(self.t_index[len(self.t_index) // 2]) if t_end is None else int(t_end)
        return self.in_degree_batch(np.array([t]))[0]

    def meta(self) -> dict:
        return {
            "kind": self.kind, "alpha_deg": self.alpha_deg, "radius_m": self.radius_m,
            "cone_deg": self.cone_deg, "k": self.k, "self_loops": self.self_loops,
            "n_edges": int(self.n_edges),
            "n_real_edges": int((~self._is_self).sum()),
        }


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #

class ForecastWindows(Dataset):
    """Sliding-window samples for one split.

    Each item is a dict with

    ``x``        ``(input_len, N, F)`` standardised features, with hidden turbines zeroed
    ``hidden``   ``(N,)`` bool, which turbines report nothing
    ``y``        ``(N,)`` standardised target
    ``y_valid``  ``(N,)`` bool, ground truth may be scored
    ``t_end``, ``h``  scalars

    When ``masker`` is given, the hidden set is drawn per item (training augmentation).
    Otherwise ``hidden`` is all ``False``, or a mask supplied as ``fixed_hidden`` - either a
    single ``(N,)`` mask applied to every sample, or a ``(n_samples, N)`` matrix giving each
    sample its own silent turbines (used by the one-turbine-at-a-time and paired H3 designs).
    """

    def __init__(
        self,
        fd: FeatureData,
        t_end: np.ndarray,
        h: np.ndarray,
        spec: WindowSpec,
        masker: RandomTurbineMasker | None = None,
        fixed_hidden: np.ndarray | None = None,
        epoch: int = 0,
    ):
        self.fd = fd
        self.t_end = np.asarray(t_end, dtype=np.int64)
        self.h = np.asarray(h, dtype=np.int64)
        self.spec = spec
        #: column of the model's output layer for each sample (the model always predicts all
        #: horizons of ``spec``; a dataset may hold only a subset of them)
        self.h_index = np.array([spec.horizons.index(int(x)) for x in self.h], dtype=np.int64)
        self.masker = masker
        self.fixed_hidden = fixed_hidden
        self.epoch = int(epoch)
        self.n = fd.n_turbines
        self.f_observed = fd.feature_index("observed")
        self.f_patv = fd.feature_index("patv")

        # target in standardised units; built once per split (memory: n_samples x N)
        self.y = np.empty((len(self.t_end), self.n), dtype=np.float32)
        self.y_valid = np.empty((len(self.t_end), self.n), dtype=bool)
        for k, (t, hh) in enumerate(zip(self.t_end, self.h)):
            self.y[k] = standardize_target(fd, fd.patv[t + hh])
            self.y_valid[k] = fd.valid[t + hh]

    def __len__(self) -> int:
        return len(self.t_end)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def hidden_for(self, index: int) -> np.ndarray:
        if self.masker is not None:
            return self.masker(index, self.epoch)
        if self.fixed_hidden is not None:
            fixed = np.asarray(self.fixed_hidden, dtype=bool)
            return fixed if fixed.ndim == 1 else fixed[index]
        return hidden_none(self.n)

    def __getitem__(self, index: int) -> dict:
        t = int(self.t_end[index])
        h = int(self.h[index])
        lo = t - self.spec.input_len + 1
        x = self.fd.x[lo:t + 1].copy()                     # (L, N, F)
        hidden = self.hidden_for(index)
        if hidden.any():
            x[:, hidden, :] = 0.0
            x[:, hidden, self.f_observed] = 0.0
        return {
            "x": torch.from_numpy(x),
            "hidden": torch.from_numpy(hidden),
            "y": torch.from_numpy(self.y[index]),
            "y_valid": torch.from_numpy(self.y_valid[index]),
            "t_end": t,
            "h": h,
            "h_index": int(self.h_index[index]),
        }


def collate(batch: list[dict]) -> dict:
    """Stack dataset items into batch tensors."""
    out = {
        "x": torch.stack([b["x"] for b in batch]),            # (B, L, N, F)
        "hidden": torch.stack([b["hidden"] for b in batch]),  # (B, N)
        "y": torch.stack([b["y"] for b in batch]),            # (B, N)
        "y_valid": torch.stack([b["y_valid"] for b in batch]),
        "t_end": np.array([b["t_end"] for b in batch], dtype=np.int64),
        "h": np.array([b["h"] for b in batch], dtype=np.int64),
        "h_index": np.array([b["h_index"] for b in batch], dtype=np.int64),
    }
    return out


def collate_device(batch: list[dict], device) -> dict:
    """``collate`` and move tensors to ``device`` (``t_end``/``h`` stay numpy)."""
    out = collate(batch)
    for key in ("x", "hidden", "y", "y_valid"):
        out[key] = out[key].to(device)
    return out


# --------------------------------------------------------------------------- #
# Wind-up / provenance
# --------------------------------------------------------------------------- #

def load_graph_meta(path=None) -> dict:
    """Stage B's chosen graph configuration (``alpha``, ``R``, cone)."""
    path = path or (C.INTERIM_DIR / "graph_meta.json")
    return json.loads(path.read_text())


def build_all(
    spec: WindowSpec | None = None,
    stride: int = 1,
) -> tuple[FeatureData, WindowSpec]:
    """Convenience: feature tensor plus the window specification."""
    spec = spec or WindowSpec()
    return build_feature_data(), spec
