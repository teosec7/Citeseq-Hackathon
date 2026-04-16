"""Foundation-model cell embeddings (scGPT, Geneformer, UCE)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# scGPT
# ---------------------------------------------------------------------------


def extract_scgpt(
    adata: AnnData,
    *,
    model_dir: str | Path = "scGPT_human",
    gene_col: str = "index",
    max_length: int = 1200,
    batch_size: int = 64,
    device: str = "cuda",
    **_kwargs: Any,
) -> np.ndarray:
    """Embed cells with a pretrained scGPT checkpoint.

    Parameters
    ----------
    model_dir
        Path to the scGPT checkpoint folder (must contain ``args.json``,
        ``best_model.pt``, and ``vocab.json``).
    gene_col
        Column in ``adata.var`` that holds gene names matching the scGPT
        vocabulary.  Use ``"index"`` to use ``adata.var_names``.
    """
    import scgpt as scg  # noqa: F811

    logger.info("Embedding cells with scGPT (model_dir=%s) …", model_dir)

    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir=Path(model_dir),
        gene_col=gene_col,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        return_new_adata=True,
    )
    return np.asarray(embed_adata.X)


# ---------------------------------------------------------------------------
# Geneformer
# ---------------------------------------------------------------------------


def extract_geneformer(
    adata: AnnData,
    *,
    model_dir: str | Path = "ctheodoris/Geneformer",
    emb_mode: str = "cell",
    emb_layer: int = -1,
    max_ncells: int | None = None,
    forward_batch_size: int = 100,
    **_kwargs: Any,
) -> np.ndarray:
    """Embed cells with a pretrained Geneformer model.

    Parameters
    ----------
    model_dir
        HuggingFace repo id or local path to the Geneformer model.
    emb_mode
        ``"cell"`` for mean-pooled cell embeddings or ``"cls"`` for CLS token.
    emb_layer
        Transformer layer to pull embeddings from (``-1`` = last).
    max_ncells
        Cap on cells to embed.  ``None`` embeds all cells.
    """
    from geneformer import EmbExtractor, TranscriptomeTokenizer

    logger.info("Tokenizing AnnData for Geneformer …")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "h5ad"
        data_dir.mkdir()
        token_dir = tmpdir / "tokenized"
        token_dir.mkdir()
        emb_dir = tmpdir / "embeddings"
        emb_dir.mkdir()

        h5ad_path = data_dir / "input.h5ad"
        adata.write_h5ad(h5ad_path)

        tokenizer = TranscriptomeTokenizer(
            special_token=True,
            model_version="V2",
        )
        tokenizer.tokenize_data(
            data_directory=data_dir,
            output_directory=token_dir,
            output_prefix="input",
            file_format="h5ad",
        )

        tokenized_path = token_dir / "input.dataset"

        extractor_kwargs: dict[str, Any] = {
            "emb_mode": emb_mode,
            "emb_layer": emb_layer,
            "forward_batch_size": forward_batch_size,
        }
        if max_ncells is not None:
            extractor_kwargs["max_ncells"] = max_ncells

        embex = EmbExtractor(**extractor_kwargs)

        logger.info("Extracting Geneformer cell embeddings …")
        embs_df = embex.extract_embs(
            model_directory=str(model_dir),
            input_data_file=str(tokenized_path),
            output_directory=str(emb_dir),
            output_prefix="emb",
        )

    emb_cols = [c for c in embs_df.columns if c not in {"cell_id", "dataset"}]
    return embs_df[emb_cols].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# UCE  (Universal Cell Embeddings)
# ---------------------------------------------------------------------------


def extract_uce(
    adata: AnnData,
    *,
    model_size: str = "small",
    species: str = "human",
    batch_size: int = 64,
    device: str = "cuda",
    **_kwargs: Any,
) -> np.ndarray:
    """Embed cells with a pretrained UCE model.

    Parameters
    ----------
    model_size
        ``"small"`` (4-layer) or ``"large"`` (33-layer).
    species
        Species of the input data (e.g. ``"human"``, ``"mouse"``).
    batch_size
        Per-device batch size for inference.
    """
    import torch
    import uce

    logger.info("Embedding cells with UCE (model=%s) …", model_size)

    model = uce.get_pretrained(model_size)
    model.eval()
    if device != "cpu":
        model = model.to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        h5ad_path = tmpdir / "input.h5ad"
        adata.write_h5ad(h5ad_path)

        dataset, dataloader = uce.get_processed_dataset(
            adata_path=str(h5ad_path),
            batch_size=batch_size,
            species=species,
        )

        all_embs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = [
                        b.to(device) if isinstance(b, torch.Tensor) else b
                        for b in batch
                    ]
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(device)

                embs = model(batch)
                if isinstance(embs, tuple):
                    embs = embs[0]
                all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0)
