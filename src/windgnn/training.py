"""Stage D: training loop, seeding and inference helpers.

The loop is deliberately small and boring: Adam, masked MSE, early stopping on the validation
MAE (in kW, on the scoreable entries), and a plateau scheduler. Everything is deterministic
given ``TrainConfig.seed``; the validation/test data never influence the loss.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import tensors as T


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Seed torch, numpy and python's RNG (call once per run)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    """``cuda`` when a usable GPU is visible, else ``cpu`` (works for ROCm and CUDA builds)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> dict:
    """What this environment can actually use - printed by the notebooks."""
    return {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "device": str(resolve_device()),
    }


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    """Everything that controls one training run."""

    epochs: int = 40
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 1e-3
    samples_per_epoch: int = 2000
    grad_clip: float = 1.0
    seed: int = 42
    device: str | None = None
    loss: str = "mse"                 # "mse" or "huber"
    verbose: bool = False
    log_every: int = 10

    def resolved_device(self) -> torch.device:
        if self.device is None:
            return resolve_device()
        return torch.device(self.device)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def masked_loss(pred, target, valid, kind: str = "mse"):
    """Loss on the entries whose ground truth may be scored.

    The mask is ``valid`` only: a silent *input* turbine is still a training target, which is
    exactly how the model learns to rely on neighbours (PLAN.md 5.2).
    """
    valid = valid.to(pred.device).bool()
    if valid.sum() == 0:
        return pred.sum() * 0.0
    diff = pred[valid] - target[valid]
    if kind == "huber":
        return nn.functional.huber_loss(diff, torch.zeros_like(diff), delta=1.0)
    return (diff ** 2).mean()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

class Trainer:
    """Train one model; keeps the best validation state."""

    def __init__(self, model: nn.Module, cfg: TrainConfig, graph: T.GraphProvider,
                 feature_names: list[str] | None = None,
                 target_stats: dict | None = None):
        self.model = model
        self.cfg = cfg
        self.graph = graph
        self.device = cfg.resolved_device()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(cfg.patience // 3, 2))
        # standardised target -> kW
        self.target_stats = target_stats or {"mean": 0.0, "std": 1.0}
        self.history: list[dict] = []
        self.best_state: dict | None = None
        self.best_val: float = float("inf")
        self.best_epoch: int = -1

    # -- helpers ---------------------------------------------------------- #
    def _to_device(self, batch: dict) -> dict:
        out = dict(batch)
        for key in ("x", "hidden", "y", "y_valid"):
            out[key] = out[key].to(self.device)
        return out

    def _forward(self, batch: dict):
        g = self.graph.batch(batch["t_end"], device=self.device)
        pred = self.model(batch["x"], g, batch["hidden"])
        # gather the output column matching each sample's horizon
        cols = torch.as_tensor(batch["h_index"], device=self.device, dtype=torch.long)
        return pred.gather(2, cols.view(-1, 1, 1).expand(-1, pred.shape[1], 1)).squeeze(-1)

    def _mae_kw(self, pred_std, target_std, valid) -> float:
        sel = valid.bool()
        if sel.sum() == 0:
            return float("nan")
        p = pred_std[sel].detach() * self.target_stats["std"] + self.target_stats["mean"]
        t = target_std[sel] * self.target_stats["std"] + self.target_stats["mean"]
        return float((p - t).abs().mean())

    # -- main loop -------------------------------------------------------- #
    def fit(self, train_ds: T.ForecastWindows, val_ds: T.ForecastWindows) -> list[dict]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n_train = len(train_ds)
        bs = cfg.batch_size
        t_start = time.perf_counter()

        for epoch in range(cfg.epochs):
            train_ds.set_epoch(epoch)
            self.model.train()
            if cfg.samples_per_epoch and cfg.samples_per_epoch < n_train:
                order = rng.choice(n_train, size=cfg.samples_per_epoch, replace=False)
            else:
                order = rng.permutation(n_train)

            total, seen = 0.0, 0
            for start in range(0, len(order), bs):
                idx = order[start:start + bs]
                batch = self._to_device(T.collate([train_ds[int(i)] for i in idx]))
                pred = self._forward(batch)
                loss = masked_loss(pred, batch["y"], batch["y_valid"], kind=cfg.loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                total += float(loss.detach()) * len(idx)
                seen += len(idx)

            val_mae = self.evaluate_mae(val_ds)
            self.scheduler.step(val_mae)
            rec = {"epoch": epoch, "train_loss": total / max(seen, 1), "val_mae_kW": val_mae,
                   "lr": self.optimizer.param_groups[0]["lr"],
                   "sec": time.perf_counter() - t_start}
            self.history.append(rec)

            improved = val_mae < self.best_val - cfg.min_delta
            if improved:
                self.best_val = val_mae
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
            if cfg.verbose and (epoch % cfg.log_every == 0 or improved):
                print(f"  epoch {epoch:3d}  loss {rec['train_loss']:.4f}  "
                      f"val MAE {val_mae:7.2f} kW  lr {rec['lr']:.2e}"
                      + ("  *" if improved else ""))
            if epoch - self.best_epoch >= cfg.patience:
                if cfg.verbose:
                    print(f"  early stop at epoch {epoch} (best {self.best_epoch})")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()
        return self.history

    # -- evaluation ------------------------------------------------------- #
    @torch.no_grad()
    def evaluate_mae(self, ds: T.ForecastWindows) -> float:
        """MAE in kW on the scoreable entries of a dataset."""
        self.model.eval()
        preds, target, valid = [], [], []
        for start in range(0, len(ds), self.cfg.batch_size):
            idx = range(start, min(start + self.cfg.batch_size, len(ds)))
            batch = self._to_device(T.collate([ds[i] for i in idx]))
            preds.append(self._forward(batch).cpu())
            target.append(batch["y"].cpu())
            valid.append(batch["y_valid"].cpu())
        if not preds:
            return float("nan")
        p = torch.cat(preds)
        t = torch.cat(target)
        v = torch.cat(valid)
        return self._mae_kw(p, t, v)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@dataclass
class Predictions:
    """Model output on a set of windows, in kW, aligned with ground truth."""

    pred: np.ndarray        # (n_samples, N) kW
    true: np.ndarray        # (n_samples, N) kW
    valid: np.ndarray       # (n_samples, N) bool
    hidden: np.ndarray      # (n_samples, N) bool
    t_end: np.ndarray       # (n_samples,)
    h: np.ndarray           # (n_samples,)

    def __len__(self) -> int:
        return len(self.t_end)

    def select(self, index) -> "Predictions":
        """Subset the samples (e.g. keep one horizon): ``preds.select(preds.h == 6)``."""
        index = np.asarray(index)
        return Predictions(pred=self.pred[index], true=self.true[index], valid=self.valid[index],
                           hidden=self.hidden[index], t_end=self.t_end[index], h=self.h[index])

    def copy_with(self, **kwargs) -> "Predictions":
        data = dict(pred=self.pred, true=self.true, valid=self.valid, hidden=self.hidden,
                    t_end=self.t_end, h=self.h)
        data.update(kwargs)
        return Predictions(**data)


@torch.no_grad()
def predict(
    model: nn.Module,
    ds: T.ForecastWindows,
    graph: T.GraphProvider,
    batch_size: int = 64,
    device=None,
    return_attention: bool = False,
) -> Predictions | tuple[Predictions, list[np.ndarray]]:
    """Run a trained model over every window of a dataset.

    Predictions are converted back to kW. Attention (when requested) is returned as a list
    over layers of ``(n_samples, n_edges, heads)`` arrays.
    """
    device = device or resolve_device()
    model.to(device)
    model.eval()
    stats = ds.fd.stats.set_index("feature").loc["patv"]
    mean, std = float(stats["mean"]), float(stats["std"])

    preds, targets, valid, hidden, t_ends, hs = [], [], [], [], [], []
    attn_layers: list[list[np.ndarray]] = []
    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        batch = T.collate([ds[i] for i in idx])
        g = graph.batch(batch["t_end"], device=device)
        x = batch["x"].to(device)
        hidden_t = batch["hidden"].to(device)
        cols = torch.as_tensor(batch["h_index"], device=device, dtype=torch.long)

        if return_attention:
            out, attn = model(x, g, hidden_t, return_attn=True)
            for k, a in enumerate(attn):
                if a is None:
                    continue
                while len(attn_layers) <= k:
                    attn_layers.append([])
                attn_layers[k].append(a.detach().cpu().numpy())
        else:
            out = model(x, g, hidden_t)

        pred_col = out.gather(2, cols.view(-1, 1, 1).expand(-1, out.shape[1], 1)).squeeze(-1)
        preds.append(pred_col.detach().cpu().numpy() * std + mean)
        targets.append(batch["y"].numpy() * std + mean)
        valid.append(batch["y_valid"].numpy())
        hidden.append(batch["hidden"].numpy())
        t_ends.append(batch["t_end"])
        hs.append(batch["h"])

    result = Predictions(
        pred=np.concatenate(preds).astype(np.float32),
        true=np.concatenate(targets).astype(np.float32),
        valid=np.concatenate(valid),
        hidden=np.concatenate(hidden),
        t_end=np.concatenate(t_ends),
        h=np.concatenate(hs),
    )
    if return_attention:
        return result, [np.concatenate(a) for a in attn_layers]
    return result


def checkpoint_path(name: str, seed: int, directory=None):
    """Where a trained run is stored (``results/checkpoints``)."""
    from . import config as C

    directory = directory or (C.RESULTS_DIR / "checkpoints")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}_seed{seed}.pt"


def save_checkpoint(trainer: Trainer, name: str, seed: int, meta: dict | None = None):
    """Persist the trained weights together with the configuration that produced them."""
    path = checkpoint_path(name, seed)
    torch.save({"state_dict": trainer.model.state_dict(), "meta": meta or {},
                "config": trainer.cfg.__dict__, "best_val": trainer.best_val,
                "best_epoch": trainer.best_epoch}, path)
    return path


def load_checkpoint(model: nn.Module, name: str, seed: int, device=None) -> dict:
    """Load weights into ``model``; returns the stored metadata."""
    path = checkpoint_path(name, seed)
    blob = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.to(device or resolve_device()).eval()
    return blob.get("meta", {})


def fit_model(
    model: nn.Module,
    train_ds: T.ForecastWindows,
    val_ds: T.ForecastWindows,
    graph: T.GraphProvider,
    cfg: TrainConfig,
    fd: T.FeatureData,
) -> Trainer:
    """Seed, train and return the trainer (with the best state loaded)."""
    set_seed(cfg.seed)
    stats = fd.stats.set_index("feature").loc["patv"]
    trainer = Trainer(model, cfg, graph,
                      target_stats={"mean": float(stats["mean"]), "std": float(stats["std"])})
    trainer.fit(train_ds, val_ds)
    return trainer
