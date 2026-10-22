"""Stage D: graph neural forecasters.

The design follows PLAN.md section 6, revised by the Stage C findings (PLAN.md 11.4):

* the **primary graph is the soft cone** - every turbine inside ``R`` is a candidate neighbour
  and the wind direction/angle enters through four **edge features**
  (``dist/R``, ``along``, ``cross``, ``|angle|/90``), so attention decides which neighbours
  matter. Hard cones (``cone < 180``) and the reversed orientation are masks over the same
  edge list, which keeps them exact ablations rather than separate models;
* a **GRU encoder** turns the ``input_len`` window into one embedding per turbine;
* a **GATv2 processor** (Brody et al., 2022) aggregates neighbours; GAT (Velickovic et al.,
  2018) is available as an ablation because its attention ranking is query-independent -
  exactly the property this project is about;
* a **silent turbine** is handled by replacing its node embedding with a learnable
  placeholder (``hidden_placeholder="learnable"``) or by leaving the zeroed embedding in
  place (``"zeros"``), with the ``observed`` flag already inside the node features.

Implementation note
-------------------
``torch_geometric.nn.GATv2Conv`` is the reference for the attention maths, but it runs one
graph at a time and, on CPU, costs ~350 ms/step for a batch of 32 farm graphs. Because all
samples share one topology here (only the edge attributes and the cone mask differ), the
layer below computes the *same* function batched over graphs with vectorised operations.
``tests`` in notebook 04 verify that transferring PyG weights into it reproduces PyG's outputs
exactly (max |diff| = 0), and it runs ~16x faster.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Attention / convolution layers (batched over graphs sharing one topology)
# --------------------------------------------------------------------------- #


class BatchedGATv2Layer(nn.Module):
    """GATv2 with edge features; batched over ``B`` graphs sharing ``(src, dst)``.

    Implements (Brody et al., ICLR 2022; edge-feature variant of PyG's ``GATv2Conv``):

    .. math::
        e_{ij} = a^{\\top} \\mathrm{LeakyReLU}\\big(W_l h_i + W_r h_j + W_e e_{ij}\\big), \\qquad
        \\alpha_{ij} = \\mathrm{softmax}_{i \\in \\mathcal{N}(j)}(e_{ij})

    with ``i`` the source (upwind) node and ``j`` the target. Weights are named like PyG's so
    a reference implementation can be loaded for verification.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        concat: bool = True,
        edge_dim: int | None = None,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.out_channels = out_dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        h, c = heads, out_dim
        self.lin_l = nn.Linear(in_dim, h * c, bias=bias)
        self.lin_r = nn.Linear(in_dim, h * c, bias=bias)
        self.lin_edge = nn.Linear(edge_dim, h * c, bias=False) if edge_dim else None
        self.att = nn.Parameter(torch.empty(1, h, c))
        self.bias = nn.Parameter(torch.zeros(h * c if concat else c)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, src, dst, edge_attr=None, mask=None, return_attn=False):
        """``x`` ``(B,N,din)`` -> ``(B,N,H*C)``; returns attention ``(B,E,H)`` if asked."""
        b, n, _ = x.shape
        h, c = self.heads, self.out_channels
        xl = self.lin_l(x).view(b, n, h, c)
        xr = self.lin_r(x).view(b, n, h, c)
        logits = xl[:, src] + xr[:, dst]
        if self.lin_edge is not None and edge_attr is not None:
            logits = logits + self.lin_edge(edge_attr).view(b, -1, h, c)
        logits = (F.leaky_relu(logits, self.negative_slope) * self.att).sum(-1)   # (B,E,H)

        alpha = masked_softmax(logits, dst, n, mask)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = xl[:, src] * alpha.unsqueeze(-1)
        out = torch.zeros(b, n, h, c, device=x.device, dtype=x.dtype)
        out = out.index_add(1, dst, msg)
        out = out.reshape(b, n, h * c) if self.concat else out.mean(2)
        if self.bias is not None:
            out = out + self.bias
        return (out, alpha) if return_attn else out


class BatchedGATLayer(nn.Module):
    """GAT (Velickovic et al., 2018) with the *static* attention of the original paper.

    .. math::
        e_{ij} = \\mathrm{LeakyReLU}\\big(a_l^{\\top} W h_i + a_r^{\\top} W h_j + b^{\\top} e_{ij}\\big)

    The logit is additive in the two node terms, so for a fixed target ``j`` the ranking of its
    neighbours ``i`` does not depend on the query - the failure mode the project is arguing
    against. Kept for that ablation only.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True,
                 edge_dim: int | None = None, negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.heads, self.out_channels, self.concat = heads, out_dim, concat
        self.negative_slope, self.dropout = negative_slope, dropout
        h, c = heads, out_dim
        self.lin_l = nn.Linear(in_dim, h * c, bias=bias)
        self.lin_r = nn.Linear(in_dim, h * c, bias=bias)
        self.att_l = nn.Parameter(torch.empty(1, h, c))
        self.att_r = nn.Parameter(torch.empty(1, h, c))
        self.lin_edge = nn.Linear(edge_dim, h, bias=False) if edge_dim else None
        self.bias = nn.Parameter(torch.zeros(h * c if concat else c)) if bias else None
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, src, dst, edge_attr=None, mask=None, return_attn=False):
        b, n, _ = x.shape
        h, c = self.heads, self.out_channels
        xl = self.lin_l(x).view(b, n, h, c)
        xr = self.lin_r(x).view(b, n, h, c)
        logits = (xl[:, src] * self.att_l).sum(-1) + (xr[:, dst] * self.att_r).sum(-1)
        if self.lin_edge is not None and edge_attr is not None:
            logits = logits + self.lin_edge(edge_attr)
        logits = F.leaky_relu(logits, self.negative_slope)

        alpha = masked_softmax(logits, dst, n, mask)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = xl[:, src] * alpha.unsqueeze(-1)
        out = torch.zeros(b, n, h, c, device=x.device, dtype=x.dtype).index_add(1, dst, msg)
        out = out.reshape(b, n, h * c) if self.concat else out.mean(2)
        if self.bias is not None:
            out = out + self.bias
        return (out, alpha) if return_attn else out


class BatchedGCNLayer(nn.Module):
    """Graph convolution with a per-sample normalised adjacency (no edge features).

    ``out = D^{-1/2} (A + I) D^{-1/2} X W`` with ``A`` the masked edge set of each sample.
    Used for the "standard graph model on a fixed distance graph" baseline.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x, src, dst, edge_attr=None, mask=None, n_nodes=None, return_attn=False):
        b, n, _ = x.shape
        device, dtype = x.device, x.dtype
        if mask is None:
            mask = torch.ones(b, src.numel(), dtype=torch.bool, device=device)
        a = torch.zeros(b, n, n, device=device, dtype=dtype)
        idx_b = torch.arange(b, device=device).unsqueeze(1).expand(b, src.numel())
        a.index_put_((idx_b[mask], dst.expand(b, -1)[mask], src.expand(b, -1)[mask]),
                     torch.ones(1, device=device, dtype=dtype), accumulate=True)
        a = a + torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
        deg = a.sum(-1).clamp(min=1.0)
        dinv = deg.pow(-0.5)
        a_norm = a * dinv.unsqueeze(-1) * dinv.unsqueeze(1)
        out = a_norm @ self.lin(x)
        if self.bias is not None:
            out = out + self.bias
        return (out, None) if return_attn else out


def masked_softmax(logits, dst, n_nodes: int, mask=None):
    """Softmax over the incoming edges of each target node, batched over graphs.

    ``logits`` is ``(B, E, H)`` and ``dst`` indexes the target node of every edge. Edges with
    ``mask=False`` get weight 0. A node with no admissible edge gets an all-zero row, which
    (with the residual connection) leaves its embedding unchanged.
    """
    b, e, h = logits.shape
    if mask is not None:
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    idx = dst.view(1, -1, 1).expand(b, -1, h)
    mx = torch.full((b, n_nodes, h), float("-inf"), device=logits.device, dtype=logits.dtype)
    mx = mx.scatter_reduce(1, idx, logits, reduce="amax", include_self=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    w = torch.exp(logits - mx.gather(1, idx))
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    den = torch.zeros(b, n_nodes, h, device=logits.device, dtype=logits.dtype).index_add_(1, dst, w)
    return w / den.gather(1, idx).clamp(min=1e-16)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class GRUEncoder(nn.Module):
    """Shared GRU over the input window, applied independently to every turbine."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(in_dim, out_dim, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out_dim = out_dim

    def forward(self, x):
        """``x`` ``(B, L, N, F)`` -> ``(B, N, d)``."""
        b, l, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, l, f)
        _, h = self.rnn(seq)
        return h[-1].view(b, n, self.out_dim)


class MLP(nn.Module):
    """Two-layer MLP with GELU and dropout."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """``LayerNorm(x + dropout(act(layer(x))))`` - the processor block of the model."""

    def __init__(self, layer: nn.Module, dim: int, dropout: float = 0.1, act=nn.GELU):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.act = act()

    def forward(self, x, **kwargs):
        out = self.layer(x, **kwargs)
        if isinstance(out, tuple):
            out, attn = out
            return self.norm(x + self.drop(self.act(out))), attn
        return self.norm(x + self.drop(self.act(out)))


# --------------------------------------------------------------------------- #
# The forecaster
# --------------------------------------------------------------------------- #

GRAPH_KINDS = ("none", "gcn", "gat", "gatv2")


class WindForecaster(nn.Module):
    """Graph neural forecaster for all turbines at once.

    Parameters
    ----------
    n_features, n_horizons:
        Input feature count and number of forecast horizons (one output per horizon).
    d:
        Hidden width.
    graph:
        ``"gatv2"`` (default), ``"gat"``, ``"gcn"`` or ``"none"`` (per-turbine MLP; the
        non-graph baseline).
    n_layers:
        Number of processor layers (ignored when ``graph="none"``).
    heads:
        Attention heads (``gat``/``gatv2`` only).
    edge_dim:
        Number of edge features; ``None`` disables them (edge-feature ablation).
    hidden_placeholder:
        ``"learnable"`` - replace a silent turbine's embedding with a learned vector;
        ``"zeros"`` - keep the zeroed-input embedding.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 3,
        d: int = 64,
        graph: str = "gatv2",
        n_layers: int = 2,
        heads: int = 4,
        edge_dim: int | None = 4,
        dropout: float = 0.1,
        temporal_dropout: float = 0.1,
        hidden_placeholder: str = "learnable",
        residual_persistence: bool = False,
        patv_feature: int = 0,
    ):
        super().__init__()
        if graph not in GRAPH_KINDS:
            raise ValueError(f"graph must be one of {GRAPH_KINDS}, got {graph!r}")
        self.graph = graph
        self.n_features = n_features
        self.n_horizons = n_horizons
        self.edge_dim = edge_dim
        self.hidden_placeholder = hidden_placeholder
        self.residual_persistence = residual_persistence
        self.patv_feature = patv_feature

        self.inp = nn.Sequential(nn.Linear(n_features, d), nn.GELU(), nn.Dropout(temporal_dropout))
        self.encoder = GRUEncoder(d, d, dropout=temporal_dropout)

        self.placeholder = (nn.Parameter(torch.zeros(d)) if hidden_placeholder == "learnable"
                            else None)

        self.blocks = nn.ModuleList()
        if graph != "none":
            for _ in range(n_layers):
                if graph == "gcn":
                    layer = BatchedGCNLayer(d, d)
                elif graph == "gat":
                    layer = BatchedGATLayer(d, d // heads, heads=heads, concat=True,
                                            edge_dim=edge_dim, dropout=dropout)
                else:
                    layer = BatchedGATv2Layer(d, d // heads, heads=heads, concat=True,
                                              edge_dim=edge_dim, dropout=dropout)
                self.blocks.append(ResidualBlock(layer, d, dropout=dropout))

        self.decoder = MLP(d, d, n_horizons, dropout=dropout)

    # ------------------------------------------------------------------ #
    def encode(self, x, hidden=None):
        """Raw window ``(B, L, N, F)`` -> node embeddings ``(B, N, d)``."""
        b, _, n, _ = x.shape
        if hidden is None:
            hidden = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        h = self.encoder(self.inp(x))
        if self.placeholder is not None:
            h = torch.where(hidden.unsqueeze(-1), self.placeholder.expand_as(h), h)
        return h

    def process(self, h, graph: dict | None, return_attn: bool = False):
        """Aggregate neighbours: embeddings ``(B, N, d)`` -> embeddings ``(B, N, d)``."""
        attn: list[torch.Tensor] = []
        if self.graph == "none":
            return h, attn
        if graph is None:
            raise ValueError(f"graph={self.graph!r} needs a graph dict")
        src, dst = graph["src"], graph["dst"]
        edge_attr = graph.get("edge_attr") if self.edge_dim is not None else None
        mask = graph.get("mask")
        for block in self.blocks:
            if return_attn:
                h, a = block(h, src=src, dst=dst, edge_attr=edge_attr, mask=mask,
                             return_attn=True)
                attn.append(a)
            else:
                h = block(h, src=src, dst=dst, edge_attr=edge_attr, mask=mask)
        if return_attn and any(a is None for a in attn):
            attn = []
        return h, attn

    def decode(self, h, x=None):
        """Node embeddings -> ``(B, N, n_horizons)``."""
        out = self.decoder(h)
        if self.residual_persistence and x is not None:
            out = out + x[:, -1, :, self.patv_feature].unsqueeze(-1)
        return out

    # ------------------------------------------------------------------ #
    def forward(self, x, graph: dict | None = None, hidden=None, return_attn: bool = False):
        """``x`` ``(B, L, N, F)`` -> ``(B, N, n_horizons)``.

        ``graph`` is the dict produced by :class:`windgnn.tensors.GraphProvider`. ``hidden``
        is the ``(B, N)`` silent-turbine mask.
        """
        h = self.encode(x, hidden)
        h, attn = self.process(h, graph, return_attn=return_attn)
        out = self.decode(h, x)
        return (out, attn) if return_attn else out

    # ------------------------------------------------------------------ #
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (f"WindForecaster(graph={self.graph}, layers={len(self.blocks)}, "
                f"d={self.inp[0].out_features}, horizons={self.n_horizons}, "
                f"edge_dim={self.edge_dim}, params={self.n_parameters():,})")


def build_model(cfg: dict, n_features: int, n_horizons: int, patv_feature: int = 0) -> WindForecaster:
    """Construct a model from a plain dict (so notebooks can sweep configurations)."""
    return WindForecaster(
        n_features=n_features,
        n_horizons=n_horizons,
        d=cfg.get("d", 64),
        graph=cfg.get("graph", "gatv2"),
        n_layers=cfg.get("n_layers", 2),
        heads=cfg.get("heads", 4),
        edge_dim=cfg.get("edge_dim", 4),
        dropout=cfg.get("dropout", 0.1),
        temporal_dropout=cfg.get("temporal_dropout", 0.1),
        hidden_placeholder=cfg.get("hidden_placeholder", "learnable"),
        residual_persistence=cfg.get("residual_persistence", False),
        patv_feature=patv_feature,
    )
