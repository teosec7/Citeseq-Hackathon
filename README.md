# Cell Embedding Extractor

A unified Python interface for extracting cell embeddings from single-cell RNA-seq data using 11 methods across three categories: simple dimensionality reduction, classical VAE models, and foundation models.

## Supported Methods

| Category | Method | Key | Package | Output dims |
|---|---|---|---|---|
| Simple | PCA | `pca` | scanpy | 50 |
| Simple | UMAP | `umap` | scanpy | 2 |
| Simple | t-SNE | `tsne` | scanpy | 2 |
| Simple | Diffusion Map | `diffmap` | scanpy | 15 |
| Classical | scVI | `scvi` | scvi-tools | 30 |
| Classical | scANVI | `scanvi` | scvi-tools | 30 |
| Classical | TOTALVI | `totalvi` | scvi-tools | 30 |
| Classical | PeakVI | `peakvi` | scvi-tools | 30 |
| Foundation | scGPT | `scgpt` | scgpt | 512 |
| Foundation | Geneformer | `geneformer` | geneformer | 512 |
| Foundation | UCE | `uce` | uce-model | 1280 |

## Installation

Install core dependencies:

```bash
pip install numpy anndata scanpy
```

For classical VAE models:

```bash
pip install scvi-tools torch
```

For foundation models (install only what you need):

```bash
pip install scgpt          # scGPT
pip install geneformer     # Geneformer
pip install uce-model      # UCE
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import scanpy as sc
from embeddings import extract_all_embeddings

adata = sc.read_h5ad("my_data.h5ad")

# Run all methods whose dependencies are installed
results = extract_all_embeddings(adata)

for name, emb in results.items():
    print(f"{name}: {emb.shape}")
# pca: (5000, 50)
# umap: (5000, 2)
# scvi: (5000, 30)
# ...
```

## Usage

### Run specific methods

```python
results = extract_all_embeddings(adata, methods=["pca", "scvi", "scgpt"])
```

### Configure VAE training

```python
results = extract_all_embeddings(
    adata,
    methods=["scvi", "scanvi"],
    batch_key="batch",
    labels_key="cell_type",
    n_latent=20,
    max_epochs=200,
)
```

### Use foundation model checkpoints

```python
results = extract_all_embeddings(
    adata,
    methods=["scgpt", "geneformer", "uce"],
    model_paths={
        "scgpt": "/data/checkpoints/scGPT_human",
        "geneformer": "ctheodoris/Geneformer",
    },
    gene_col="feature_name",
    species="human",
    device="cuda",
)
```

### List available methods

```python
from embeddings import list_methods

# All known methods
list_methods()
# ['pca', 'umap', 'tsne', 'diffmap', 'scvi', 'scanvi', 'totalvi', 'peakvi', 'scgpt', 'geneformer', 'uce']

# Only methods with installed dependencies
list_methods(available_only=True)
# ['pca', 'umap', 'tsne', 'diffmap', 'scvi', ...]
```

### Store embeddings back into AnnData

```python
results = extract_all_embeddings(adata, methods=["pca", "scvi", "scgpt"])

for name, emb in results.items():
    adata.obsm[f"X_{name}"] = emb
```

## API Reference

### `extract_all_embeddings`

```python
extract_all_embeddings(
    adata,
    methods=None,
    *,
    force_recompute=False,
    batch_key=None,
    labels_key=None,
    unlabeled_category="Unknown",
    protein_expression_obsm_key="protein_expression",
    n_latent=30,
    max_epochs=100,
    n_comps=50,
    model_paths=None,
    device="auto",
    species="human",
    gene_col="index",
) -> dict[str, np.ndarray]
```

| Parameter | Description |
|---|---|
| `adata` | AnnData object (cells x genes). |
| `methods` | List of method keys to run. `None` runs all available. |
| `force_recompute` | Recompute simple embeddings even if they exist in `adata.obsm`. |
| `batch_key` | `adata.obs` column for batch correction (scVI family). |
| `labels_key` | `adata.obs` column for cell-type labels (required by scANVI). |
| `unlabeled_category` | Label value for unlabeled cells in scANVI. |
| `protein_expression_obsm_key` | `adata.obsm` key for protein counts (TOTALVI). |
| `n_latent` | Latent dimensions for VAE models. |
| `max_epochs` | Training epochs for VAE models. |
| `n_comps` | Number of PCA components. |
| `model_paths` | Dict mapping method keys to checkpoint paths. |
| `device` | `"cuda"`, `"cpu"`, or `"auto"`. |
| `species` | Species for UCE (`"human"`, `"mouse"`, ...). |
| `gene_col` | `adata.var` column with gene names for scGPT. `"index"` uses `var_names`. |

**Returns:** `dict[str, np.ndarray]` -- each value has shape `(n_cells, n_dims)`.

## Method-Specific Notes

### scANVI

Requires `labels_key` pointing to a cell-type annotation column in `adata.obs`. Skipped with a warning if not provided.

### TOTALVI

Requires protein surface counts in `adata.obsm["protein_expression"]` (or a custom key via `protein_expression_obsm_key`). Designed for CITE-seq data. Skipped if protein data is absent.

### PeakVI

Expects `adata.X` to be a peak-count matrix (cells x peaks) from ATAC-seq. Not applicable to standard scRNA-seq data.

### Foundation Models

All foundation models require pretrained checkpoints. scGPT expects a local folder with model weights; Geneformer accepts a HuggingFace model ID or local path; UCE downloads weights automatically. A GPU is strongly recommended.

## Logging

The package logs progress via Python's `logging` module under the `embeddings` namespace. To see output:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Project Structure

```
embeddings/
    __init__.py       Public API
    extract.py        Orchestrator and method registry
    simple.py         PCA, UMAP, t-SNE, Diffusion Map
    classical.py      scVI, scANVI, TOTALVI, PeakVI
    foundation.py     scGPT, Geneformer, UCE
requirements.txt      Dependencies
```

## Adding a New Method

1. Write an extractor function with signature `(adata: AnnData, **kwargs) -> np.ndarray` in the appropriate module.
2. Add a one-line entry to the registry in `extract.py`:

```python
FOUNDATION_METHODS["my_method"] = ("embeddings.foundation", "extract_my_method")
```

The orchestrator picks it up automatically.
