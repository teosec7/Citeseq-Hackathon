from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class QueryResult:
    text: str                # rendered caption
    embedding: np.ndarray    # (256,)
    similarity: np.ndarray   # (n_cells,) cosine similarity
    umap_point: np.ndarray | None = None  # (2,) projected query location, if computed


@dataclass
class VizContext:
    """Everything a viz needs to render, bundled once per session."""
    cell_umap: np.ndarray            # (n_cells, 2)
    cell_types: np.ndarray | None    # (n_cells,) or None
    n_cells: int
    has_h5mu: bool
    obs: pd.DataFrame | None = None  # full per-cell metadata (Status, Subject, Run, …)


class Viz(Protocol):
    name: str
    description: str
    requires_cell_types: bool
    requires_query: bool = True

    def render(
        self,
        ctx: VizContext,
        query_a: QueryResult | None,
        query_b: QueryResult | None,
    ) -> None:
        ...


def rank_cell_types(query: QueryResult, cell_types: np.ndarray) -> list[tuple[str, float]]:
    """Rank every cell type by mean cosine-similarity to the query. Descending."""
    unique = np.unique(cell_types)
    scored = [(str(ct), float(query.similarity[cell_types == ct].mean())) for ct in unique]
    scored.sort(key=lambda kv: -kv[1])
    return scored


def cohort_mask_for_query(
    query: QueryResult,
    cell_types: np.ndarray,
    n_top_types: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Cells belonging to the top-N cell types by mean similarity to the query.
    Returns (bool mask of length n_cells, list of chosen cell-type names)."""
    ranked = rank_cell_types(query, cell_types)
    chosen = [ct for ct, _ in ranked[:n_top_types]]
    mask = np.isin(cell_types, chosen)
    return mask, chosen
