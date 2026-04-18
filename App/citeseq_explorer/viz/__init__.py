from .base import QueryResult, Viz, VizContext, cohort_mask_for_query, rank_cell_types
from .heatmap_viz import HeatmapViz
from .umap_viz import UMAPViz
from .violin_viz import ViolinViz

REGISTRY: list[Viz] = [
    UMAPViz(),
    HeatmapViz(),
    ViolinViz(),
]

__all__ = [
    "Viz", "VizContext", "QueryResult",
    "REGISTRY", "cohort_mask_for_query", "rank_cell_types",
]
