"""Neighbour means, MLP, Transformer and GATv2 as gap fillers on the silence axis.

The three neural models exist to answer one question: with the *same* information the
linear ceiling gets, does a nonlinear model, attention over turbines, or a graph buy
anything? So every one of them reads exactly what ``LinearBoth`` reads --

* the farm snapshot at the step being predicted, with the masked turbine's column zeroed
  and flagged, and
* the masked turbine's own last ``window`` readings, ending at its final live step,

-- and nothing else. Any difference in the table is therefore architecture, not access.
``LinearPeers`` remains the spatial-only reference and ``NeighbourMean`` the unweighted
one, which together separate "the graph carries signal" from "averaging destroys it".

Dark peers are filled with the training mean (zero after standardisation) and marked
``observed = 0``, rather than with the leave-one-out farm mean ``LinearPeers`` needs:
the flag tells the model the value is absent, so there is no imputation path through
which the answer could leak back in.

Deviations from the published designs this mirrors are deliberate: one shared model with
a learned turbine embedding instead of one model per turbine, and a budget of a few
epochs on subsampled episodes.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from . import baselines as B
from . import config as C
from . import data as D

SEED = B.RNG_SEED
THREADS = 6


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def knn_edges(k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Directed ``src -> dst`` edge list over the ``k`` nearest turbines, self excluded.

    Symmetrised, so information can flow both ways. This is Edward's "GNN undirected"
    topology, which beat every wind-following cone he tried.
    """
    loc = D.load_location()
    xy = loc[["x", "y"]].to_numpy(dtype=np.float64)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argsort(d2, axis=1)[:, :k]
    src = nn_idx.ravel()
    dst = np.repeat(np.arange(xy.shape[0]), k)
    pairs = set(map(tuple, np.stack([src, dst], 1))) | set(map(tuple, np.stack([dst, src], 1)))
    arr = np.array(sorted(pairs), dtype=np.int64)
    return arr[:, 0], arr[:, 1]


# --------------------------------------------------------------------------- #
# Unweighted neighbour means (the stage C rungs, on the silence axis)
# --------------------------------------------------------------------------- #

class NeighbourMean(B.Baseline):
    """Plain mean of the reporting turbines, over the whole farm or the ``k`` nearest.

    Parameter-free, so there is nothing to fit. The masked turbine is always excluded
    from its own mean. Where no neighbour reports, the per-turbine time-of-day
    climatology stands in.
    """

    required_window = 12

    def __init__(self, k: int | None = None):
        self.k = k
        self.name = "farm-mean" if k is None else f"knn-mean k={k}"

    def fit(self, ctx, s, window):
        if self.k is not None:
            loc = D.load_location()
            xy = loc[["x", "y"]].to_numpy(dtype=np.float64)
            d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            self.nbr_ = np.argsort(d2, axis=1)[:, : self.k]
        return self

    def predict(self, ctx, ep):
        block = ctx.power[ep.t]                       # (B, N), NaN where dark
        obs = np.isfinite(block)
        vals = np.where(obs, block, 0.0).astype(np.float64)
        rows = np.arange(len(ep))
        obs = obs.copy()
        obs[rows, ep.turb] = False                    # never average yourself in
        vals[rows, ep.turb] = 0.0
        if self.k is None:
            num, den = vals.sum(1), obs.sum(1).astype(np.float64)
        else:
            idx = self.nbr_[ep.turb]                  # (B, k)
            num = np.take_along_axis(vals, idx, 1).sum(1)
            den = np.take_along_axis(obs, idx, 1).sum(1).astype(np.float64)
        out = np.divide(num, den, out=np.full(len(ep), np.nan), where=den > 0)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = ctx.clim[ep.turb[bad], ep.t[bad] % C.STEPS_PER_DAY]
        return out


# --------------------------------------------------------------------------- #
# Shared feature construction
# --------------------------------------------------------------------------- #

class Standardiser:
    """Train-split mean/std for power, shared by every neural model."""

    def fit(self, ctx):
        lo, hi = D.split_bounds("train")
        block = ctx.power[lo : hi + 1]
        vals = block[np.isfinite(block)].astype(np.float64)
        self.mean = float(vals.mean())
        self.std = float(vals.std()) or 1.0
        return self

    def z(self, x):
        return (x - self.mean) / self.std

    def inv(self, z):
        return z * self.std + self.mean


def snapshot(ctx, t, turb, s, window, sc: Standardiser):
    """``(nodes, own)``: per-turbine channels ``(B, N, 3)`` and own lags ``(B, window)``.

    Node channels are ``[power, observed, is_target]``. The target's power channel is
    zeroed and its ``observed`` flag cleared, so the snapshot never contains the answer.
    """
    block = ctx.power[t].astype(np.float64)
    obs = np.isfinite(block)
    rows = np.arange(t.size)
    p = np.where(obs, sc.z(np.nan_to_num(block)), 0.0)
    o = obs.astype(np.float64)
    p[rows, turb] = 0.0
    o[rows, turb] = 0.0
    tgt = np.zeros_like(o)
    tgt[rows, turb] = 1.0
    nodes = np.stack([p, o, tgt], axis=-1).astype(np.float32)
    own = sc.z(B._own_lags(ctx, t, turb, s, window)).astype(np.float32)
    return nodes, own


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #

class _Stem(nn.Module):
    """Node channels plus a learned turbine embedding; own lags added at the target."""

    def __init__(self, n_turbines: int, window: int, dim: int, in_ch: int = 3):
        super().__init__()
        self.embed = nn.Embedding(n_turbines, dim)
        self.proj = nn.Linear(in_ch, dim)
        self.own = nn.Sequential(nn.Linear(window, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, nodes, own, turb):
        h = self.proj(nodes) + self.embed.weight[None, :, :]
        rows = torch.arange(h.shape[0], device=h.device)
        return h.index_put((rows, turb), h[rows, turb] + self.own(own))


class MlpGapFill(nn.Module):
    """Flatten the snapshot and the own lags into one vector; two hidden layers."""

    def __init__(self, n_turbines: int, window: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(n_turbines, 32)
        d_in = n_turbines * 2 + window + 32
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, nodes, own, turb):
        flat = torch.cat([nodes[:, :, 0], nodes[:, :, 1], own, self.embed(turb)], dim=1)
        return self.net(flat).squeeze(-1)


class TransformerGapFill(nn.Module):
    """One token per turbine, full self-attention, read out at the masked turbine.

    A learned fully-connected graph: attention may pick any turbine, which is the
    control the GATv2 model needs -- it separates "a fixed topology helps" from
    "looking at other turbines helps".
    """

    def __init__(self, n_turbines: int, window: int, dim: int = 48, heads: int = 4,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = _Stem(n_turbines, window, dim)
        enc = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=2 * dim, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.body = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dim, 1)

    def forward(self, nodes, own, turb):
        h = self.body(self.stem(nodes, own, turb))
        rows = torch.arange(h.shape[0], device=h.device)
        return self.out(h[rows, turb]).squeeze(-1)


class _GATv2Layer(nn.Module):
    """GATv2 (Brody et al., 2022) over a fixed edge list, batched over snapshots."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.c = heads, dim // heads
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.att = nn.Parameter(torch.empty(1, heads, self.c))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, src, dst):
        bsz, n, _ = x.shape
        zs = self.lin_src(x).view(bsz, n, self.h, self.c)
        zd = self.lin_dst(x).view(bsz, n, self.h, self.c)
        e = (nn.functional.leaky_relu(zs[:, src] + zd[:, dst], 0.2) * self.att).sum(-1)
        # softmax over the edges entering each destination node, shifted for stability
        m = torch.full((bsz, n, self.h), -1e30, device=x.device, dtype=e.dtype)
        m = m.index_reduce(1, dst, e, "amax", include_self=True)
        w = torch.exp(e - m[:, dst])
        den = torch.zeros((bsz, n, self.h), device=x.device, dtype=e.dtype)
        den.index_add_(1, dst, w)
        a = self.drop(w / (den[:, dst] + 1e-16))
        out = torch.zeros_like(zs)
        out.index_add_(1, dst, zs[:, src] * a.unsqueeze(-1))
        return out.reshape(bsz, n, self.h * self.c)


class GnnGapFill(nn.Module):
    """Two GATv2 hops over the kNN graph, read out at the masked turbine."""

    def __init__(self, n_turbines: int, window: int, src, dst, dim: int = 48,
                 heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = _Stem(n_turbines, window, dim)
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        self.layers = nn.ModuleList(_GATv2Layer(dim, heads, dropout) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.act = nn.GELU()
        self.out = nn.Linear(dim, 1)

    def forward(self, nodes, own, turb):
        h = self.stem(nodes, own, turb)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + self.act(layer(h, self.src, self.dst)))
        rows = torch.arange(h.shape[0], device=h.device)
        return self.out(h[rows, turb]).squeeze(-1)


# --------------------------------------------------------------------------- #
# The Baseline wrapper that trains any of them
# --------------------------------------------------------------------------- #

class NeuralGapFill(B.Baseline):
    """Trains one shared network per silence length and scores it on the episode set."""

    required_window = 12

    LOSSES = {"mse": nn.MSELoss, "l1": nn.L1Loss, "huber": nn.HuberLoss}

    def __init__(self, kind: str = "mlp", n_train: int = 40_000, n_val: int = 6_000,
                 epochs: int = 6, batch: int = 256, lr: float = 1e-3, k: int = 8,
                 patience: int = 2, verbose: bool = False, loss: str = "mse",
                 seed: int = SEED):
        self.kind = kind
        self.name = {"mlp": "mlp", "transformer": "transformer", "gnn": "gnn-gatv2"}[kind]
        self.n_train, self.n_val = n_train, n_val
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.k, self.patience, self.verbose = k, patience, verbose
        # `mse` is the natural loss for RMSE, `l1` for MAE. The ridge baselines pick their
        # alpha on validation MAE, so comparing an MSE-trained network against them on MAE
        # scores each family on the metric only one of them was tuned for.
        self.loss, self.seed = loss, seed

    def _build(self, window):
        n = C.N_TURBINES
        if self.kind == "mlp":
            return MlpGapFill(n, window)
        if self.kind == "transformer":
            return TransformerGapFill(n, window)
        src, dst = knn_edges(self.k)
        return GnnGapFill(n, window, src, dst)

    def _batches(self, ctx, ep, s, window, idx, shuffle, rng):
        def gen():
            order = rng.permutation(idx) if shuffle else idx
            for a in range(0, order.size, self.batch):
                sel = order[a : a + self.batch]
                nodes, own = snapshot(ctx, ep.t[sel], ep.turb[sel], s, window, self.sc_)
                y = self.sc_.z(ep.truth(ctx)[sel].astype(np.float64)).astype(np.float32)
                yield (torch.from_numpy(nodes), torch.from_numpy(own),
                       torch.from_numpy(ep.turb[sel].astype(np.int64)), torch.from_numpy(y))
        return gen

    def fit(self, ctx, s, window):
        torch.manual_seed(self.seed)
        torch.set_num_threads(THREADS)
        rng = np.random.default_rng(self.seed)
        self.sc_ = Standardiser().fit(ctx)
        self.window = window

        tr = B.sample_episodes(ctx, "train", s, None, window, rng)
        va = B.sample_episodes(ctx, "val", s, None, window, rng)
        itr = rng.choice(len(tr), size=min(self.n_train, len(tr)), replace=False)
        iva = rng.choice(len(va), size=min(self.n_val, len(va)), replace=False)

        dev = _device()
        model = self._build(window).to(dev)
        self.n_params_ = n_parameters(model)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = self.LOSSES[self.loss]()
        train_b = self._batches(ctx, tr, s, window, itr, True, rng)
        val_b = self._batches(ctx, va, s, window, iva, False, rng)

        best, best_state, bad = np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            t0, tot, cnt = time.time(), 0.0, 0
            for nodes, own, turb, y in train_b():
                nodes, own, turb, y = (z.to(dev) for z in (nodes, own, turb, y))
                opt.zero_grad()
                loss = crit(model(nodes, own, turb), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()) * y.shape[0]
                cnt += y.shape[0]
            sched.step()
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for nodes, own, turb, y in val_b():
                    nodes, own, turb, y = (z.to(dev) for z in (nodes, own, turb, y))
                    vt += float(crit(model(nodes, own, turb), y)) * y.shape[0]
                    vn += y.shape[0]
            vloss = vt / max(vn, 1)
            if self.verbose:
                print(f"      epoch {epoch}/{self.epochs}  train {tot/max(cnt,1):.5f}  "
                      f"val {vloss:.5f}  {time.time()-t0:.0f}s", flush=True)
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"      early stop at epoch {epoch}", flush=True)
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        # named for the loss actually optimised, which is not always MSE
        self.model_, self.dev_, self.val_loss_ = model, dev, best
        return self

    def predict(self, ctx, ep):
        self.model_.eval()
        out = np.empty(len(ep), dtype=np.float64)
        with torch.no_grad():
            for a in range(0, len(ep), 1024):
                sl = slice(a, min(a + 1024, len(ep)))
                nodes, own = snapshot(ctx, ep.t[sl], ep.turb[sl], ep.s, self.window, self.sc_)
                z = self.model_(
                    torch.from_numpy(nodes).to(self.dev_),
                    torch.from_numpy(own).to(self.dev_),
                    torch.from_numpy(ep.turb[sl].astype(np.int64)).to(self.dev_),
                )
                out[sl] = self.sc_.inv(z.cpu().numpy().astype(np.float64))
        return out


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
