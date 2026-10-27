"""The Zhou et al. (2022) KDD Cup GRU baseline, ported from PaddlePaddle to PyTorch.

The original lives in ``paper_code/`` and cannot run here: PaddlePaddle publishes no
wheel for Python 3.13 on Windows. The architecture is small enough to port exactly --
a 2-layer GRU with hidden size 48 and a linear projection -- so the port is faithful
and the hyperparameters below are the published defaults from ``paper_code/prepare.py``.

Two protocols are provided:

``run_paper_protocol``
    The published setup, as a sanity check on the port: 10 input channels, 144 steps
    (24 h) in, 288 steps (48 h) out, trained on the paper's day split and scored with
    the official metric on the official ``final_phase_test`` files.

``GruGapFill``
    The same architecture re-targeted to our gap-filling task, so it can sit in the
    same table as the linear baselines: the masked turbine's own 10 channels for the
    144 steps ending at its last live reading, predicting its power ``s`` steps later.

Both report in the units of the protocol they belong to -- kW pooled for gap filling,
MW summed over turbines for the published metric. They are not interchangeable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import baselines as B
from . import config as C
from . import data as D

# Published defaults, paper_code/prepare.py
IN_VAR = 10
OUT_VAR = 1
HIDDEN = 48
LAYERS = 2
DROPOUT = 0.05
INPUT_LEN = 144            # 24 h
OUTPUT_LEN = 288           # 48 h
LR = 1e-4
BATCH = 32
EPOCHS = 10
PATIENCE = 3
GRAD_CLIP = 50.0
TRAIN_DAYS = 153
VAL_DAYS = 16

SEED = 16


class BaselineGru(nn.Module):
    """Port of ``paper_code/model.py::BaselineGruModel``.

    The forward pass appends ``output_len`` zero steps to the input, runs the GRU over
    the whole concatenated sequence, projects every step and keeps the tail -- exactly
    what the Paddle implementation does.
    """

    def __init__(self, in_var=IN_VAR, out_var=OUT_VAR, hidden=HIDDEN,
                 layers=LAYERS, dropout=DROPOUT, output_len=OUTPUT_LEN):
        super().__init__()
        self.output_len = output_len
        self.out_var = out_var
        self.gru = nn.GRU(in_var, hidden, num_layers=layers)   # sequence-first, as in Paddle
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden, out_var)

    def forward(self, x):                      # x: (B, L, C)
        pad = torch.zeros(x.shape[0], self.output_len, x.shape[2], dtype=x.dtype, device=x.device)
        seq = torch.cat((x, pad), dim=1).transpose(0, 1)
        dec, _ = self.gru(seq)
        dec = dec.transpose(0, 1)
        out = self.projection(self.dropout(dec))
        return out[:, -self.output_len:, -self.out_var:]


class LastStep(nn.Module):
    """Expose only the final predicted step of a :class:`BaselineGru`.

    The published model emits ``output_len`` steps. Gap filling scores exactly one --
    the step at the end of the silence -- so the loss must be taken against that step
    alone. Without this wrapper an ``(B, 1, 1)`` target broadcasts against an
    ``(B, s, 1)`` output and every step in the gap is trained toward the final value.
    """

    def __init__(self, inner: BaselineGru):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)[:, -1:, :]


@dataclass
class Scaler:
    """Per-channel standardisation, fitted on training rows only."""

    mean: np.ndarray = field(default=None)
    std: np.ndarray = field(default=None)

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_target(self, y):
        """Undo scaling for the target channel only (Patv is the last column)."""
        return y * self.std[-1] + self.mean[-1]


def channel_tensor(df: pd.DataFrame) -> np.ndarray:
    """``(T, N, 10)`` array of the measurement channels on the full step grid.

    NaN is replaced by 0, as the published pipeline does
    (``paper_code/wind_turbine_data.py:149``). The replacement is applied to raw units
    before scaling, so a dark sensor enters the model as "zero", not as "average".
    """
    cols = C.MEASUREMENT_COLS
    out = np.zeros((C.N_STEPS, C.N_TURBINES, len(cols)), dtype=np.float32)
    wide = df.pivot_table(index="t", columns="TurbID", values=cols, aggfunc="first")
    steps = np.arange(C.N_STEPS)
    turbines = np.arange(1, C.N_TURBINES + 1)
    for k, col in enumerate(cols):
        out[:, :, k] = (
            wide[col].reindex(index=steps, columns=turbines).to_numpy(dtype=np.float32)
        )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train(model, batches, val_batches, epochs, lr=None, patience=PATIENCE, verbose=True):
    """Adam + MSE on standardised targets, halving LR each epoch, with early stopping."""
    lr = LR if lr is None else lr
    dev = _device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        for group in opt.param_groups:                 # paper_code lr_adjust 'type1'
            group["lr"] = lr * (0.5 ** (epoch - 1))
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in batches():
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss.detach()) * xb.shape[0]
            n += xb.shape[0]
        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_batches():
                xb, yb = xb.to(dev), yb.to(dev)
                vt += float(crit(model(xb), yb)) * xb.shape[0]
                vn += xb.shape[0]
        vloss = vt / max(vn, 1)
        if verbose:
            print(f"    epoch {epoch:2d}/{epochs}  train {total/max(n,1):.5f}  "
                  f"val {vloss:.5f}  {time.time()-t0:.0f}s")
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------- #
# (a) the published protocol -- sanity check on the port
# --------------------------------------------------------------------------- #

def official_test_cases(limit: int | None = None):
    """Load the ``final_phase_test`` infile/outfile pairs shipped with the release."""
    base = C.KDDCUP_DIR / "final_phase_test"
    infiles = sorted((base / "infile").glob("*.csv"))
    outfiles = sorted((base / "outfile").glob("*.csv"))
    if not infiles:
        raise FileNotFoundError(f"no official test files under {base}")
    pairs = list(zip(infiles, outfiles))[: limit or None]
    cols = C.MEASUREMENT_COLS
    for fin, fout in pairs:
        din = pd.read_csv(fin).sort_values(["TurbID", "Day", "Tmstamp"])
        dout = pd.read_csv(fout).sort_values(["TurbID", "Day", "Tmstamp"])
        n_in = len(din) // C.N_TURBINES
        n_out = len(dout) // C.N_TURBINES
        x = din[cols].to_numpy(dtype=np.float32).reshape(C.N_TURBINES, n_in, len(cols))
        y = dout[cols].to_numpy(dtype=np.float32).reshape(C.N_TURBINES, n_out, len(cols))
        yield np.nan_to_num(x, nan=0.0), y, dout


def official_score(pred_mw, truth_mw, exclude):
    """``paper_code/metrics.py``: per-turbine MAE/RMSE in MW, summed over turbines.

    ``pred_mw``/``truth_mw`` are ``(cases, turbines, steps)``; ``exclude`` marks the
    steps the published rule drops (stopped, feathered, or out-of-range directions).
    """
    maes, rmses = [], []
    for j in range(pred_mw.shape[1]):
        per_case_mae, per_case_rmse = [], []
        for i in range(pred_mw.shape[0]):
            keep = ~exclude[i, j]
            if not keep.any():
                continue
            err = pred_mw[i, j][keep] - truth_mw[i, j][keep]
            per_case_mae.append(np.mean(np.abs(err)))
            per_case_rmse.append(np.sqrt(np.mean(err ** 2)))
        if per_case_mae:
            maes.append(np.mean(per_case_mae))
            rmses.append(np.mean(per_case_rmse))
    total_mae, total_rmse = float(np.sum(maes)), float(np.sum(rmses))
    return {
        "MAE [MW, summed]": total_mae,
        "RMSE [MW, summed]": total_rmse,
        "score": (total_mae + total_rmse) / 2,
        "turbines": len(maes),
    }


def exclusion_mask(dout: pd.DataFrame) -> np.ndarray:
    """The published ``cond`` from ``turbine_scores``, as ``(turbines, steps)``."""
    cond = (
        ((dout["Patv"] <= 0) & (dout["Wspd"] > C.UNKNOWN_WSPD_THRESHOLD))
        | (dout["Pab1"] > C.PITCH_UNKNOWN_THRESHOLD)
        | (dout["Pab2"] > C.PITCH_UNKNOWN_THRESHOLD)
        | (dout["Pab3"] > C.PITCH_UNKNOWN_THRESHOLD)
        | (dout["Wdir"] < C.WDIR_ABNORMAL_RANGE[0]) | (dout["Wdir"] > C.WDIR_ABNORMAL_RANGE[1])
        | (dout["Ndir"] < C.NDIR_ABNORMAL_RANGE[0]) | (dout["Ndir"] > C.NDIR_ABNORMAL_RANGE[1])
    )
    cond = cond | dout["Patv"].isna()
    n = len(dout) // C.N_TURBINES
    return cond.to_numpy().reshape(C.N_TURBINES, n)


def run_paper_protocol(df, n_windows=120_000, epochs=4, batch=128, limit_cases=None,
                       verbose=True):
    """Train on the paper's day split, score with the official metric on its test files.

    Deviation from the published run, forced by the hardware available: the paper fits
    one model per turbine (134 models); this fits a single shared model on windows
    pooled across turbines, and subsamples windows per epoch. The architecture,
    sequence lengths, optimiser, learning-rate schedule and metric are unchanged.
    """
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    chan = channel_tensor(df)

    train_end = TRAIN_DAYS * C.STEPS_PER_DAY
    val_end = train_end + VAL_DAYS * C.STEPS_PER_DAY
    scaler = Scaler().fit(chan[:train_end].reshape(-1, chan.shape[2]).astype(np.float64))
    scaled = scaler.transform(chan.astype(np.float64)).astype(np.float32)

    def window_starts(lo, hi):
        last = hi - INPUT_LEN - OUTPUT_LEN
        return np.arange(lo, last + 1)

    tr_starts, va_starts = window_starts(0, train_end), window_starts(train_end, val_end)
    if verbose:
        print(f"paper protocol: train days 1-{TRAIN_DAYS}, val {TRAIN_DAYS+1}-{TRAIN_DAYS+VAL_DAYS}")
        print(f"  window starts  train {len(tr_starts):,}  val {len(va_starts):,}  "
              f"x {C.N_TURBINES} turbines")

    def make_batches(starts, cap, shuffle=True):
        def gen():
            s = rng.choice(starts, size=min(cap, len(starts) * C.N_TURBINES), replace=True) \
                if shuffle else np.repeat(starts, 1)
            j = rng.integers(0, C.N_TURBINES, size=len(s))
            for k in range(0, len(s), batch):
                ss, jj = s[k:k + batch], j[k:k + batch]
                idx = ss[:, None] + np.arange(INPUT_LEN)[None, :]
                xb = scaled[idx, jj[:, None], :]
                tgt = ss[:, None] + INPUT_LEN + np.arange(OUTPUT_LEN)[None, :]
                yb = scaled[tgt, jj[:, None], -1:]
                yield torch.from_numpy(xb), torch.from_numpy(yb)
        return gen

    model = BaselineGru(output_len=OUTPUT_LEN)
    _train(model, make_batches(tr_starts, n_windows), make_batches(va_starts, n_windows // 8),
           epochs=epochs, verbose=verbose)

    dev = _device()
    model.eval()
    preds, truths, excl = [], [], []
    t0 = time.time()
    for x, y, dout in official_test_cases(limit_cases):
        hist = scaler.transform(x[:, -INPUT_LEN:, :].astype(np.float64)).astype(np.float32)
        with torch.no_grad():
            p = model(torch.from_numpy(hist).to(dev)).cpu().numpy()[:, :, 0]
        preds.append(scaler.inverse_target(p.astype(np.float64)))
        truths.append(np.nan_to_num(y[:, :, -1].astype(np.float64)))
        excl.append(exclusion_mask(dout))
    if verbose:
        print(f"  scored {len(preds)} official test cases in {time.time()-t0:.0f}s")
    pred = np.stack(preds) / 1000.0            # kW -> MW, as the official metric does
    truth = np.stack(truths) / 1000.0
    return official_score(pred, truth, np.stack(excl))


# --------------------------------------------------------------------------- #
# (b) the same architecture, re-targeted to gap filling
# --------------------------------------------------------------------------- #

class GruGapFill(B.Baseline):
    """Zhou et al. GRU architecture predicting the masked turbine ``s`` steps ahead.

    Inputs are the masked turbine's own ten channels over the ``INPUT_LEN`` steps
    ending at its last live reading -- strictly more information than ``linear-own``
    gets, and the same information the published model uses, so the comparison does not
    handicap it. It never sees another turbine, which is the point: it marks how far a
    purely temporal model can go.
    """

    name = "paper-gru"

    def __init__(self, n_train=60_000, n_val=8_000, epochs=3, batch=128, input_len=INPUT_LEN,
                 lr=LR):
        self.n_train, self.n_val = n_train, n_val
        self.epochs, self.batch, self.input_len = epochs, batch, input_len
        self.lr = lr
        self.required_window = input_len      # keeps baselines.run() purging honest

    def attach(self, chan: np.ndarray):
        """Supply the ``(T, N, 10)`` channel tensor; built once and shared across models."""
        self.chan = chan
        return self

    def _windows(self, ep, s):
        idx = ep.t[:, None] - s - self.input_len + 1 + np.arange(self.input_len)[None, :]
        return self.chan[idx, ep.turb[:, None], :]

    def fit(self, ctx, s, window):
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        tr = B.sample_episodes(ctx, "train", s, self.n_train, self.input_len + s, rng)
        va = B.sample_episodes(ctx, "val", s, self.n_val, self.input_len + s, rng)

        raw = self._windows(tr, s).reshape(-1, self.chan.shape[2]).astype(np.float64)
        self.scaler = Scaler().fit(raw)
        self.s = s

        def batches(ep):
            def gen():
                order = rng.permutation(len(ep))
                for k in range(0, len(order), self.batch):
                    sel = order[k:k + self.batch]
                    sub = B.Episodes(t=ep.t[sel], turb=ep.turb[sel], s=s, split=ep.split)
                    xb = self.scaler.transform(self._windows(sub, s).astype(np.float64))
                    y = (ctx.power[sub.t, sub.turb].astype(np.float64) - self.scaler.mean[-1]) \
                        / self.scaler.std[-1]
                    yield (torch.from_numpy(xb.astype(np.float32)),
                           torch.from_numpy(y.astype(np.float32)).view(-1, 1, 1))
            return gen

        self.model = LastStep(BaselineGru(output_len=s))
        _train(self.model, batches(tr), batches(va), epochs=self.epochs, lr=self.lr,
               verbose=False)
        return self

    def predict(self, ctx, ep):
        dev = _device()
        self.model.eval()
        out = np.empty(len(ep), dtype=np.float64)
        with torch.no_grad():
            for k in range(0, len(ep), 512):
                sel = slice(k, k + 512)
                sub = B.Episodes(t=ep.t[sel], turb=ep.turb[sel], s=ep.s, split=ep.split)
                xb = self.scaler.transform(self._windows(sub, ep.s).astype(np.float64))
                p = self.model(torch.from_numpy(xb.astype(np.float32)).to(dev))
                out[sel] = self.scaler.inverse_target(p[:, -1, 0].cpu().numpy().astype(np.float64))
        return out
