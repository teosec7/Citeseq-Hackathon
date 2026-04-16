"""Classical VAE-based embeddings via scvi-tools."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


def _train_scvi_model(
    model_cls: type,
    adata: AnnData,
    setup_kwargs: dict[str, Any],
    model_kwargs: dict[str, Any],
    max_epochs: int,
) -> np.ndarray:
    """Shared helper: setup -> init -> train -> latent representation."""
    model_cls.setup_anndata(adata, **setup_kwargs)
    model = model_cls(adata, **model_kwargs)
    model.train(max_epochs=max_epochs)
    return np.asarray(model.get_latent_representation())


# ── Individual extractors ────────────────────────────────────────────


def extract_scvi(
    adata: AnnData,
    *,
    batch_key: str | None = None,
    n_latent: int = 30,
    max_epochs: int = 100,
    **_kwargs: Any,
) -> np.ndarray:
    """Train scVI and return latent embedding (n_cells, n_latent)."""
    import scvi

    logger.info("Training scVI (n_latent=%d, max_epochs=%d) …", n_latent, max_epochs)
    return _train_scvi_model(
        scvi.model.SCVI,
        adata.copy(),
        setup_kwargs={"batch_key": batch_key},
        model_kwargs={"n_latent": n_latent},
        max_epochs=max_epochs,
    )


def extract_scanvi(
    adata: AnnData,
    *,
    batch_key: str | None = None,
    labels_key: str | None = None,
    unlabeled_category: str = "Unknown",
    n_latent: int = 30,
    max_epochs: int = 100,
    **_kwargs: Any,
) -> np.ndarray:
    """Train scANVI and return latent embedding (n_cells, n_latent).

    Requires *labels_key* pointing to a column in ``adata.obs``.
    """
    import scvi

    if labels_key is None or labels_key not in adata.obs.columns:
        raise ValueError(
            f"scANVI requires labels_key (got {labels_key!r}). "
            "Pass labels_key= pointing to an adata.obs column."
        )

    logger.info("Training scANVI (n_latent=%d, max_epochs=%d) …", n_latent, max_epochs)

    ad = adata.copy()
    scvi.model.SCVI.setup_anndata(ad, batch_key=batch_key)
    scvi_model = scvi.model.SCVI(ad, n_latent=n_latent)
    scvi_model.train(max_epochs=max_epochs)

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category=unlabeled_category,
        labels_key=labels_key,
    )
    scanvi_model.train(max_epochs=max_epochs)
    return np.asarray(scanvi_model.get_latent_representation())


def extract_totalvi(
    adata: AnnData,
    *,
    batch_key: str | None = None,
    protein_expression_obsm_key: str = "protein_expression",
    n_latent: int = 30,
    max_epochs: int = 100,
    **_kwargs: Any,
) -> np.ndarray:
    """Train TOTALVI (RNA + protein) and return latent embedding.

    Requires protein counts stored in ``adata.obsm[protein_expression_obsm_key]``.
    """
    import scvi

    if protein_expression_obsm_key not in adata.obsm:
        raise ValueError(
            f"TOTALVI requires protein counts in adata.obsm['{protein_expression_obsm_key}']. "
            "Set protein_expression_obsm_key= if stored under a different key."
        )

    logger.info("Training TOTALVI (n_latent=%d, max_epochs=%d) …", n_latent, max_epochs)

    ad = adata.copy()
    scvi.model.TOTALVI.setup_anndata(
        ad,
        batch_key=batch_key,
        protein_expression_obsm_key=protein_expression_obsm_key,
    )
    model = scvi.model.TOTALVI(ad, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    return np.asarray(model.get_latent_representation())


def extract_peakvi(
    adata: AnnData,
    *,
    batch_key: str | None = None,
    n_latent: int = 30,
    max_epochs: int = 100,
    **_kwargs: Any,
) -> np.ndarray:
    """Train PeakVI (ATAC-seq peaks) and return latent embedding.

    Expects *adata.X* to contain a peak-count matrix (cells x peaks).
    """
    import scvi

    logger.info("Training PeakVI (n_latent=%d, max_epochs=%d) …", n_latent, max_epochs)

    ad = adata.copy()
    scvi.model.PEAKVI.setup_anndata(ad, batch_key=batch_key)
    model = scvi.model.PEAKVI(ad, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    return np.asarray(model.get_latent_representation())
