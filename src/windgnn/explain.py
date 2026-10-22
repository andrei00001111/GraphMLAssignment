"""Stage F: explaining which neighbours a silent turbine's forecast relies on.

Three attribution families are implemented, deliberately, because they support different
claims (Lecture 11: attention is not automatically a faithful explanation):

``attention``
    GATv2 attention coefficients, averaged over heads and layers. Cheap, always available,
    but only an *architectural* signal - hence the faithfulness check below.
``occlusion``
    Model-agnostic: hide one neighbour at a time and measure how much the target turbine's
    prediction moves. Expensive but faithful by construction.
``GNNExplainer``
    PyG's mask-optimisation explainer (Ying et al., 2019). Runs on the *processor* only, with
    the GRU embeddings held fixed, so the masks are spatial (which neighbours / edges) and a
    regression target is well defined - PyG supports
    ``ModelConfig(mode="regression", task_level="node")``.

Evaluation follows the Lecture 11 protocol:

* **plausibility** - AUPRC / precision@k of the attribution ranking against the physical
  ground truth "this neighbour is inside the upwind cone";
* **faithfulness** - sufficiency and comprehensiveness of the top-k attributed edges;
* **sparsity** - normalised entropy of the attribution distribution (lower is sparser).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import tensors as T
from .models import WindForecaster


# --------------------------------------------------------------------------- #
# Attribution container
# --------------------------------------------------------------------------- #

@dataclass
class EdgeAttribution:
    """Attribution scores over the edges that point *into* one turbine.

    ``scores`` holds one entry per listed edge (``edge_ids`` indexes the global edge list), so
    it can be compared directly with the graph's own ``along`` values and the cone ground
    truth.
    """

    node: int                  # 1-based turbine id
    edge_ids: np.ndarray       # indices into the global edge list
    scores: np.ndarray         # attribution per listed edge
    src: np.ndarray            # upwind turbine id (1-based) per listed edge
    edge_attr: np.ndarray      # (n_edges, 4) edge features at this timestamp
    method: str = "attention"
    horizon: int = 0

    def __len__(self) -> int:
        return len(self.edge_ids)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _expand_batch(graph: dict, b: int) -> dict:
    """Broadcast a single-sample graph dict to a batch of ``b`` identical graphs."""
    out = dict(graph)
    for key in ("edge_attr", "mask"):
        v = out.get(key)
        if v is not None and torch.is_tensor(v) and v.shape[0] == 1 and b > 1:
            out[key] = v.expand(b, *v.shape[1:])
    return out


def _attribution_from_scores(provider, graph: dict, node: int, scores: np.ndarray,
                             method: str, horizon: int = 0) -> EdgeAttribution:
    dst = provider.dst
    src = provider.src
    # edges into the node, *excluding self-loops*: a self-loop is the node's own message, not
    # a neighbour, and it would trivially look "aligned with the wind" (along = 1).
    edge_ids = np.flatnonzero(dst == node - 1)
    edge_ids = edge_ids[~provider._is_self[edge_ids]]
    attrs = graph["edge_attr"][0].detach().cpu().numpy() if graph.get("edge_attr") is not None \
        else np.zeros((provider.n_edges, len(T.EDGE_FEATURES)))
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] == provider.n_edges:      # global per-edge scores -> this node's edges
        scores = scores[edge_ids]
    return EdgeAttribution(
        node=node, edge_ids=edge_ids, scores=scores,
        src=(src[edge_ids] + 1).astype(int), edge_attr=attrs[edge_ids],
        method=method, horizon=horizon,
    )


# --------------------------------------------------------------------------- #
# Attention attribution
# --------------------------------------------------------------------------- #

def aggregate_attention(attn_layers: list[torch.Tensor]) -> np.ndarray:
    """Average attention over heads and layers: list of ``(B,E,H)`` -> ``(B,E)``."""
    if not attn_layers:
        raise ValueError("no attention tensors were returned (graph='none' or a GCN)")
    stack = [a.mean(dim=-1) for a in attn_layers]
    return torch.stack(stack, dim=0).mean(dim=0).cpu().numpy()


@torch.no_grad()
def attention_attribution(
    model: WindForecaster,
    x: np.ndarray,
    provider: T.GraphProvider,
    graph: dict,
    hidden: np.ndarray,
    node: int,
    horizon_index: int,
    device=None,
) -> EdgeAttribution:
    """Attention mass that each upwind neighbour sends to ``node`` at one timestamp."""
    device = device or torch.device("cpu")
    model.to(device).eval()
    xb = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    hidden0 = torch.as_tensor(hidden, dtype=torch.bool, device=device).unsqueeze(0)
    h = model.encode(xb, hidden0)
    _, attn = model.process(h, _expand_batch(graph, 1), return_attn=True)
    scores = aggregate_attention(attn)[0]
    return _attribution_from_scores(provider, graph, node, scores, "attention", horizon_index)


# --------------------------------------------------------------------------- #
# Occlusion attribution
# --------------------------------------------------------------------------- #

@torch.no_grad()
def occlusion_attribution(
    model: WindForecaster,
    x: np.ndarray,
    provider: T.GraphProvider,
    graph: dict,
    hidden: np.ndarray,
    node: int,
    horizon_index: int,
    device=None,
    batch_size: int = 128,
    target_std: float = 1.0,
    target_mean: float = 0.0,
) -> EdgeAttribution:
    """Hide each neighbour of ``node`` in turn and measure the prediction change (kW).

    Faithful by construction: it is the model's own sensitivity to the removal of that
    neighbour's data, not an architectural proxy.
    """
    device = device or torch.device("cpu")
    model.to(device).eval()
    xb = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    hidden0 = torch.as_tensor(hidden, dtype=torch.bool, device=device).unsqueeze(0)
    idx = node - 1

    def run(masks):
        b = masks.shape[0]
        g = _expand_batch(graph, b)
        out = model(xb.expand(b, *xb.shape[1:]), g, masks)
        return (out[:, idx, horizon_index].detach().cpu().numpy() * target_std + target_mean)

    base = float(run(hidden0)[0])
    edge_ids = np.flatnonzero(provider.dst == idx)
    edge_ids = edge_ids[~provider._is_self[edge_ids]]
    if edge_ids.size == 0:
        return EdgeAttribution(node, edge_ids, np.zeros(0), np.zeros(0, int),
                               np.zeros((0, 4)), "occlusion", horizon_index)

    scores = np.zeros(len(edge_ids))
    for start in range(0, len(edge_ids), batch_size):
        chunk = edge_ids[start:start + batch_size]
        masks = hidden0.repeat(len(chunk), 1).clone()
        masks[torch.arange(len(chunk), device=device),
              torch.as_tensor(provider.src[chunk], device=device)] = True
        scores[start:start + len(chunk)] = np.abs(run(masks) - base)
    return _attribution_from_scores(provider, graph, node, scores, "occlusion", horizon_index)


# --------------------------------------------------------------------------- #
# GNNExplainer (processor only, embeddings fixed)
# --------------------------------------------------------------------------- #

class _PyGProcessorRegressor(torch.nn.Module):
    """PyG-native surrogate of the trained processor, for ``GNNExplainer``.

    ``GNNExplainer`` optimises masks through hooks on ``MessagePassing`` modules, which our
    batched layers are not. This wrapper rebuilds the *same* computation with PyG
    ``GATv2Conv`` layers and transfers the trained weights into them - our layer was verified
    to be bit-exact with ``GATv2Conv`` (notebook 04), so the explained model is the trained
    model, not an approximation. The input is the frozen GRU embedding of one timestamp; the
    output is one horizon's prediction for every node, which PyG accepts as
    ``ModelConfig(mode="regression", task_level="node")``.
    """

    def __init__(self, model: "WindForecaster", horizon_index: int):
        super().__init__()
        import copy as _copy

        from torch_geometric.nn import GATv2Conv

        if model.graph != "gatv2":
            raise NotImplementedError(
                "GNNExplainer is wired for graph='gatv2'; use attention/occlusion for the "
                "other processors")
        self.horizon_index = int(horizon_index)
        d = model.blocks[0].layer.lin_l.out_features // model.blocks[0].layer.heads
        heads = model.blocks[0].layer.heads
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for block in model.blocks:
            conv = GATv2Conv(block.layer.lin_l.in_features, d, heads=heads, concat=True,
                             edge_dim=model.edge_dim, add_self_loops=False)
            conv.load_state_dict(block.layer.state_dict(), strict=True)
            self.convs.append(conv)
            self.norms.append(_copy.deepcopy(block.norm))
        self.act = block.act
        self.drop = block.drop
        self.decoder = _copy.deepcopy(model.decoder)

    def forward(self, x, edge_index, edge_attr=None):
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + self.drop(self.act(out)))
        out = self.decoder(x)                       # (N, H)
        return out[:, self.horizon_index].unsqueeze(-1)


def gnnexplainer_attribution(
    model: "WindForecaster",
    x: np.ndarray,
    provider: T.GraphProvider,
    graph: dict,
    hidden: np.ndarray,
    node: int,
    horizon_index: int,
    epochs: int = 100,
    device=None,
) -> EdgeAttribution:
    """Edge importance from PyG's ``GNNExplainer`` for one (turbine, timestamp, horizon).

    The surrogate is built with the trained weights, so the mask optimisation explains the
    model that produced the forecast. Only ``graph="gatv2"`` is supported (see
    :class:`_PyGProcessorRegressor`).
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig

    device = device or torch.device("cpu")
    model.to(device).eval()
    xb = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    hidden0 = torch.as_tensor(hidden, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        h = model.encode(xb, hidden0)[0].detach()

    # only the edges that are actually active for this timestamp (cone mask)
    if graph.get("mask") is not None:
        active = torch.nonzero(graph["mask"][0], as_tuple=False).flatten()
    else:
        active = torch.arange(provider.n_edges, device=device)
    src = torch.as_tensor(provider.src, dtype=torch.long, device=device)[active]
    dst = torch.as_tensor(provider.dst, dtype=torch.long, device=device)[active]
    edge_index = torch.stack([src, dst])
    edge_attr = (graph["edge_attr"][0].to(device)[active]
                 if graph.get("edge_attr") is not None else None)

    wrapper = _PyGProcessorRegressor(model, horizon_index).to(device)
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=ModelConfig(task_level="node", mode="regression"),
    )
    explanation = explainer(x=h, edge_index=edge_index, edge_attr=edge_attr, index=node - 1)
    mask = explanation.edge_mask.detach().cpu().numpy()

    dst_all = provider.dst
    edge_ids = np.flatnonzero(dst_all == node - 1)
    edge_ids = edge_ids[~np.asarray(provider._is_self)[edge_ids]]
    global_of_active = np.asarray(active.detach().cpu())
    pos = {int(g): i for i, g in enumerate(global_of_active)}
    scores = np.array([mask[pos[int(e)]] if int(e) in pos else 0.0 for e in edge_ids])
    attrs = (graph["edge_attr"][0].detach().cpu().numpy()
             if graph.get("edge_attr") is not None else np.zeros((provider.n_edges, 4)))
    return EdgeAttribution(
        node=node, edge_ids=edge_ids, scores=scores.astype(np.float64),
        src=(np.asarray(provider.src)[edge_ids] + 1).astype(int),
        edge_attr=attrs[edge_ids], method="gnnexplainer", horizon=horizon_index,
    )


# --------------------------------------------------------------------------- #
# Ground truth and plausibility
# --------------------------------------------------------------------------- #

def upwind_ground_truth(attribution: EdgeAttribution, cone_deg: float = 45.0) -> np.ndarray:
    """Boolean "is this neighbour inside the upwind cone", from the edge's ``along``."""
    along = attribution.edge_attr[:, 1]
    return along >= np.cos(np.deg2rad(cone_deg))


def downwind_ground_truth(attribution: EdgeAttribution, cone_deg: float = 45.0) -> np.ndarray:
    """The mirror label ("this neighbour is *downwind* of the turbine"), i.e. the control."""
    along = attribution.edge_attr[:, 1]
    return along <= -np.cos(np.deg2rad(cone_deg))


def plausibility(attribution: EdgeAttribution, cone_deg: float = 45.0,
                 direction: str = "upwind") -> dict:
    """AUPRC, ROC-AUC, precision@k and F1 against the cone ground truth.

    ``direction="downwind"`` scores against the mirrored label and is the control: a model that
    genuinely prefers upwind neighbours should score clearly higher on ``upwind``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    label = upwind_ground_truth if direction == "upwind" else downwind_ground_truth
    truth = label(attribution, cone_deg).astype(int)
    scores = np.nan_to_num(np.asarray(attribution.scores, dtype=np.float64))
    out = {"method": attribution.method, "n_edges": int(len(scores)),
           "n_upwind": int(truth.sum()), "auprc": np.nan, "roc_auc": np.nan,
           "precision_at_k": np.nan, "f1": np.nan}
    # A ranking problem needs both classes to be populated; with one positive among five
    # candidates AUPRC is trivially 1.0 and carries no information.
    if truth.sum() < 2 or (len(truth) - truth.sum()) < 2:
        return out
    out["auprc"] = float(average_precision_score(truth, scores))
    out["roc_auc"] = float(roc_auc_score(truth, scores))
    k = int(truth.sum())
    top = np.argsort(-scores)[:k]
    out["precision_at_k"] = float(truth[top].mean())
    pred = np.zeros_like(truth)
    pred[top] = 1
    tp = int(((pred == 1) & (truth == 1)).sum())
    out["f1"] = float(2 * tp / (pred.sum() + truth.sum())) if (pred.sum() + truth.sum()) else np.nan
    return out


def sparsity(scores: np.ndarray) -> float:
    """Normalised entropy of the attribution distribution (0 = a single edge)."""
    s = np.abs(np.asarray(scores, dtype=np.float64))
    s = s[np.isfinite(s)]
    if s.size <= 1 or s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(s)))


# --------------------------------------------------------------------------- #
# Faithfulness: sufficiency and comprehensiveness
# --------------------------------------------------------------------------- #

@torch.no_grad()
def edge_subset_effect(
    model: WindForecaster,
    x: np.ndarray,
    graph: dict,
    hidden: np.ndarray,
    node: int,
    horizon_index: int,
    keep: np.ndarray,
    device=None,
    target_std: float = 1.0,
    target_mean: float = 0.0,
) -> dict:
    """Prediction change when an edge subset is kept vs. removed (both in kW).

    * ``kept_delta_kW`` - **sufficiency**: keeping only these edges still reproduces the
      prediction (small is good);
    * ``removed_delta_kW`` - **comprehensiveness**: deleting them changes it (large is good).
    """
    device = device or torch.device("cpu")
    model.to(device).eval()
    xb = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    hidden0 = torch.as_tensor(hidden, dtype=torch.bool, device=device).unsqueeze(0)
    idx = node - 1
    base_mask = graph.get("mask")

    def run(mask):
        g = dict(graph)
        g["mask"] = mask
        out = model(xb, g, hidden0)
        return float(out[0, idx, horizon_index]) * target_std + target_mean

    keep_t = torch.as_tensor(keep, dtype=torch.bool, device=device).unsqueeze(0)
    only = keep_t if base_mask is None else (base_mask & keep_t)
    without = ~keep_t if base_mask is None else (base_mask & ~keep_t)
    base = run(base_mask)
    kept = run(only)
    removed = run(without)
    return {"base_kW": base, "kept_kW": kept, "removed_kW": removed,
            "kept_delta_kW": kept - base, "removed_delta_kW": removed - base}


def top_k_mask(attribution: EdgeAttribution, n_edges: int, k: int,
               provider=None) -> np.ndarray:
    """Boolean mask over the **global** edge list selecting the ``k`` best attributed edges.

    Self-loops are always kept (dropping them removes a node's own message, which is a
    different question from "which neighbour matters").
    """
    keep = np.zeros(n_edges, dtype=bool)
    scores = np.asarray(attribution.scores, dtype=np.float64)
    if k > 0 and len(scores):
        top = np.argsort(-scores)[:k]
        keep[attribution.edge_ids[top]] = True
    if provider is not None:
        keep |= provider._is_self
    return keep


def explain_graph(provider: T.GraphProvider, t_end: int, device=None) -> dict:
    """The graph dict for a single timestamp (a batch of one)."""
    return provider.batch(np.array([t_end]), device=device)
