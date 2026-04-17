# CITE-seq CLIP Explorer

Interactive Streamlit app to probe the CLIP-style model trained in `Nick/v2.ipynb`.

Type (or build) a protein-expression query, see the similarity over cells via:
- **UMAP** — cells colored by query similarity (or by CellType), with the query point projected as a gold star
- **Heatmap** — mean query similarity per known cell type
- **Violin** — full distribution of similarity per cell type

Two queries can run side-by-side for A/B comparison.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run streamlit_app.py
```

The first launch computes all cell embeddings and fits UMAP, then caches both
under `cache/` (`all_rna_embeddings.npy`, `umap_coords_all.npy`,
`umap_reducer.joblib`). Subsequent launches are fast.

## Data

Expected in the App folder (where `streamlit_app.py` lives):

- `cache/clip_cite_v2_best.pt` — model weights (already there)
- `cache/gse_rna_hvg_encodings.npy` — per-cell 2000-HVG inputs (already there)
- `*.h5mu` — CITE-seq dataset with `rna` + `protein` modalities. Drop it anywhere
  in the App folder once the download finishes and hit the **↻ Re-scan for
  dataset** button in the sidebar (or restart the app).

While the `.h5mu` is missing, free-text queries + UMAP still work using a
fallback marker list. CellType-based views (heatmap, violin, UMAP CellType
coloring) will activate automatically once the file appears.

## Adding a new visualization

Drop a file in `viz_app/viz/` with a class that implements the `Viz` protocol
(`base.py`), then register it in `viz_app/viz/__init__.py`'s `REGISTRY` list.
It will show up as a new tab in the main area.
