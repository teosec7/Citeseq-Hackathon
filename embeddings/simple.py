"""Simple / linear dimensionality-reduction embeddings via Scanpy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


def _ensure_pca(adata: AnnData, n_comps: int = 50, force: bool = False) -> None:
    if "X_pca" not in adata.obsm or force:
        import scanpy as sc

        logger.info("Computing PCA with %d components …", n_comps)
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    else:
        logger.info("Reusing existing PCA in adata.obsm['X_pca'].")


def _ensure_neighbors(adata: AnnData, force: bool = False) -> None:
    has_neighbors = "neighbors" in adata.uns and "connectivities" in adata.obsp
    if not has_neighbors or force:
        import scanpy as sc

        _ensure_pca(adata, force=False)
        logger.info("Computing neighbor graph …")
        sc.pp.neighbors(adata)
    else:
        logger.info("Reusing existing neighbor graph.")


# ── Individual extractors ────────────────────────────────────────────


def extract_pca(
    adata: AnnData,
    *,
    n_comps: int = 50,
    force_recompute: bool = False,
    **_kwargs: Any,
) -> np.ndarray:
    """Return PCA embedding (n_cells, n_comps)."""
    _ensure_pca(adata, n_comps=n_comps, force=force_recompute)
    return np.asarray(adata.obsm["X_pca"])


def extract_umap(
    adata: AnnData,
    *,
    force_recompute: bool = False,
    **_kwargs: Any,
) -> np.ndarray:
    """Return UMAP embedding (n_cells, 2)."""
    import scanpy as sc

    if "X_umap" not in adata.obsm or force_recompute:
        _ensure_neighbors(adata, force=force_recompute)
        logger.info("Computing UMAP …")
        sc.tl.umap(adata)
    else:
        logger.info("Reusing existing UMAP in adata.obsm['X_umap'].")
    return np.asarray(adata.obsm["X_umap"])


def extract_tsne(
    adata: AnnData,
    *,
    force_recompute: bool = False,
    **_kwargs: Any,
) -> np.ndarray:
    """Return t-SNE embedding (n_cells, 2)."""
    import scanpy as sc

    if "X_tsne" not in adata.obsm or force_recompute:
        _ensure_pca(adata, force=False)
        logger.info("Computing t-SNE …")
        sc.tl.tsne(adata)
    else:
        logger.info("Reusing existing t-SNE in adata.obsm['X_tsne'].")
    return np.asarray(adata.obsm["X_tsne"])


def extract_diffmap(
    adata: AnnData,
    *,
    force_recompute: bool = False,
    **_kwargs: Any,
) -> np.ndarray:
    """Return diffusion-map embedding (n_cells, n_comps)."""
    import scanpy as sc

    if "X_diffmap" not in adata.obsm or force_recompute:
        _ensure_neighbors(adata, force=force_recompute)
        logger.info("Computing diffusion map …")
        sc.tl.diffmap(adata)
    else:
        logger.info("Reusing existing diffusion map in adata.obsm['X_diffmap'].")
    return np.asarray(adata.obsm["X_diffmap"])
