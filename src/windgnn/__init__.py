"""Wind-following GNN project package.

Stages A–F of the project live here:

``config``      paths, dataset constants, figure styling
``data``        loading, cleaning, record validity, farm wind direction   (Stage A)
``graphs``      geometry, static graphs, the wind-following graph         (Stage B)
``imputation``  masked-value recovery: floors, OLS, neighbour means       (Stage C)
``tensors``     feature tensor, windows, graph providers, outages         (Stage D)
``models``      batched GATv2 / GAT / GCN forecasters                     (Stage D)
``training``    training loop, checkpoints, inference                     (Stage D)
``evaluation``  metrics, degradation analysis, H1–H3 statistics          (Stage E)
``explain``     attention / occlusion / GNNExplainer attributions        (Stage F)
``viz``         plotting helpers
"""

from . import config, data, evaluation, explain, graphs, imputation, models, tensors, training, viz  # noqa: F401

__all__ = [
    "config", "data", "graphs", "imputation", "tensors", "models", "training",
    "evaluation", "explain", "viz",
]
