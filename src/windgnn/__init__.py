"""Wind-following GNN project package.

Stage A (data analysis) and Stage B (graph construction) live here; later stages
(models, training, explainability) extend the same package.
"""

from . import baselines, config, data, graphs, viz  # noqa: F401

__all__ = ["baselines", "config", "data", "graphs", "viz"]
