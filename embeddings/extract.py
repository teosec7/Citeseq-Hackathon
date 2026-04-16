"""Orchestrator that dispatches to per-method embedding extractors."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

EmbeddingFn = Callable[..., np.ndarray]

SIMPLE_METHODS: dict[str, tuple[str, str]] = {
    "pca": ("embeddings.simple", "extract_pca"),
    "umap": ("embeddings.simple", "extract_umap"),
    "tsne": ("embeddings.simple", "extract_tsne"),
    "diffmap": ("embeddings.simple", "extract_diffmap"),
}

CLASSICAL_METHODS: dict[str, tuple[str, str]] = {
    "scvi": ("embeddings.classical", "extract_scvi"),
    "scanvi": ("embeddings.classical", "extract_scanvi"),
    "totalvi": ("embeddings.classical", "extract_totalvi"),
    "peakvi": ("embeddings.classical", "extract_peakvi"),
}

FOUNDATION_METHODS: dict[str, tuple[str, str]] = {
    "scgpt": ("embeddings.foundation", "extract_scgpt"),
    "geneformer": ("embeddings.foundation", "extract_geneformer"),
    "uce": ("embeddings.foundation", "extract_uce"),
}

METHOD_REGISTRY: dict[str, tuple[str, str]] = {
    **SIMPLE_METHODS,
    **CLASSICAL_METHODS,
    **FOUNDATION_METHODS,
}

ALL_METHODS = list(METHOD_REGISTRY)

# Category look-up for nicer log messages
_CATEGORY = {}
for _name in SIMPLE_METHODS:
    _CATEGORY[_name] = "simple"
for _name in CLASSICAL_METHODS:
    _CATEGORY[_name] = "classical"
for _name in FOUNDATION_METHODS:
    _CATEGORY[_name] = "foundation"


def _resolve_fn(module_path: str, fn_name: str) -> EmbeddingFn:
    """Lazily import *module_path* and return the callable *fn_name*."""
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_methods(available_only: bool = False) -> list[str]:
    """Return known method names.

    Parameters
    ----------
    available_only
        If ``True``, only return methods whose dependencies can be imported
        right now.
    """
    if not available_only:
        return list(METHOD_REGISTRY)

    out: list[str] = []
    for name, (mod_path, fn_name) in METHOD_REGISTRY.items():
        try:
            _resolve_fn(mod_path, fn_name)
            out.append(name)
        except (ImportError, ModuleNotFoundError):
            pass
    return out


def extract_all_embeddings(
    adata: AnnData,
    methods: list[str] | None = None,
    *,
    force_recompute: bool = False,
    batch_key: str | None = None,
    labels_key: str | None = None,
    unlabeled_category: str = "Unknown",
    protein_expression_obsm_key: str = "protein_expression",
    n_latent: int = 30,
    max_epochs: int = 100,
    n_comps: int = 50,
    model_paths: dict[str, str | None] | None = None,
    device: str = "auto",
    species: str = "human",
    gene_col: str = "index",
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Compute cell embeddings for *adata* across multiple methods.

    Parameters
    ----------
    adata
        Annotated data matrix (cells x genes).
    methods
        Which methods to run.  ``None`` means *all methods whose
        dependencies are installed*.
    force_recompute
        Re-run simple methods even if results already exist in ``adata.obsm``.
    batch_key
        Column in ``adata.obs`` encoding batch labels (used by scVI family).
    labels_key
        Column in ``adata.obs`` encoding cell-type labels (required by scANVI).
    unlabeled_category
        Value in *labels_key* that marks unlabeled cells (scANVI).
    protein_expression_obsm_key
        Key in ``adata.obsm`` holding protein counts (TOTALVI).
    n_latent
        Latent-space dimensionality for VAE models.
    max_epochs
        Training epochs for VAE models.
    n_comps
        Number of PCA components.
    model_paths
        Per-method paths to pretrained checkpoints, e.g.
        ``{"scgpt": "/data/scGPT_human", "geneformer": "ctheodoris/Geneformer"}``.
    device
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (pick GPU if available).
    species
        Species string for UCE (``"human"``, ``"mouse"``, …).
    gene_col
        Column in ``adata.var`` with gene names for scGPT.  ``"index"`` uses
        ``adata.var_names``.
    **kwargs
        Forwarded to individual extractor functions.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from method name to an ``(n_cells, n_dims)`` embedding array.
    """
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if methods is None:
        methods = list_methods(available_only=True)
        logger.info("Auto-detected available methods: %s", methods)
    else:
        unknown = set(methods) - set(METHOD_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown method(s): {unknown}. "
                f"Choose from {list(METHOD_REGISTRY)}."
            )

    model_paths = model_paths or {}

    shared_kwargs: dict[str, Any] = {
        "force_recompute": force_recompute,
        "batch_key": batch_key,
        "labels_key": labels_key,
        "unlabeled_category": unlabeled_category,
        "protein_expression_obsm_key": protein_expression_obsm_key,
        "n_latent": n_latent,
        "max_epochs": max_epochs,
        "n_comps": n_comps,
        "device": device,
        "species": species,
        "gene_col": gene_col,
        **kwargs,
    }

    results: dict[str, np.ndarray] = {}

    for name in methods:
        mod_path, fn_name = METHOD_REGISTRY[name]
        category = _CATEGORY.get(name, "unknown")

        method_kwargs = dict(shared_kwargs)
        if name in model_paths and model_paths[name] is not None:
            method_kwargs["model_dir"] = model_paths[name]

        try:
            fn = _resolve_fn(mod_path, fn_name)
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "Skipping %s (%s): missing dependency – %s", name, category, exc
            )
            continue

        logger.info("── Running %s (%s) ──", name, category)
        t0 = time.perf_counter()
        try:
            emb = fn(adata, **method_kwargs)
            elapsed = time.perf_counter() - t0
            results[name] = emb
            logger.info(
                "   %s done — shape %s (%.1fs)", name, emb.shape, elapsed
            )
        except Exception as exc:
            logger.warning("Skipping %s (%s): %s", name, category, exc)

    logger.info(
        "Finished: %d/%d methods succeeded: %s",
        len(results),
        len(methods),
        list(results),
    )
    return results
