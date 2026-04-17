"""CITE-seq CLIP explorer.

Interactive app for the Lemanic Life Sciences Hackathon 2026 project.
Takes a free-text or builder-style protein query, embeds it with BioBERT +
the trained CLIP protein head, and renders per-query similarity across cells.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from viz_app import cohort_panel, config, summary
from viz_app.data import Dataset, load_dataset
from viz_app.embeddings import (
    QueryEncoder,
    compute_all_rna_embeddings,
    fit_or_load_umap,
    project_query_to_umap,
    similarity,
)
from viz_app.model import CLIPCITE, load_clip_model
from viz_app.viz import QueryResult, VizContext
from viz_app.viz.heatmap_viz import HeatmapViz
from viz_app.viz.umap_viz import UMAPViz
from viz_app.viz.violin_viz import ViolinViz


st.set_page_config(
    page_title="Predicting protein expression from gene expression",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Global styling
# -----------------------------------------------------------------------------

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
  :root { color-scheme: light; }

  html, body, [class*="css"], .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMarkdownContainer"],
  h1, h2, h3, h4, h5, h6, p, span, label, div,
  input, textarea, select, button {
    font-family: 'Archivo Narrow', 'Helvetica Neue', Arial, sans-serif !important;
  }

  /* Palette — cream bg, near-black text */
  .stApp { background-color: #E0DED8 !important; }
  .stApp, body, p, span, label, div, h1, h2, h3, h4, h5, h6 { color: #111111 !important; }
  [data-testid="stCaptionContainer"], .caption, small { color: #333333 !important; }

  /* Hide sidebar entirely */
  [data-testid="stSidebar"] { display: none !important; }
  [data-testid="collapsedControl"] { display: none !important; }
  header[data-testid="stHeader"] { background: transparent; }

  div.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1400px;
  }

  h1.app-title {
    text-align: center;
    font-weight: 700;
    font-size: 4.5rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    line-height: 1;
    margin: 0.5rem 0 0.25rem 0;
  }
  p.app-subtitle {
    text-align: center;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 1.75rem;
  }

  /* Query label beside the text input */
  p.query-label {
    text-align: right;
    padding-top: 0.55rem;
    margin: 0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    white-space: nowrap;
  }

  /* Text inputs on the cream background */
  input[type="text"], textarea, [data-baseweb="input"] input {
    background-color: #FFFDF6 !important;
    color: #111111 !important;
    border-color: #111111 !important;
  }

  /* Radio / multiselect tweaks */
  div[data-testid="stRadio"] label p { font-size: 1.05rem; }
  [data-baseweb="tag"] { background-color: #111111 !important; color: #E0DED8 !important; }

  /* Bump base font everywhere except the title panel */
  .stApp, p, span, label, div, input, button, [data-testid="stMarkdownContainer"] p {
    font-size: 1.08rem;
  }
  h1.app-title, h1.app-title * { font-size: 4.5rem !important; }
  p.app-subtitle, p.app-subtitle * { font-size: 0.9rem !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
    font-size: 1.05rem !important;
    font-weight: 600;
  }

  /* Tabs: keep the tab bar on cream, flip the panels below to white */
  [data-baseweb="tab-list"] {
    background-color: #E0DED8 !important;
    border-bottom: 2px solid #111111;
    gap: 0.25rem;
  }
  [data-baseweb="tab"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    background-color: transparent !important;
    padding: 0.6rem 1.2rem !important;
  }
  [aria-selected="true"][data-baseweb="tab"] { color: #111111 !important; }

  [data-baseweb="tab-panel"] {
    background-color: #FFFFFF !important;
    padding: 1.5rem 1.25rem !important;
    border-radius: 0 0 10px 10px;
  }
  /* The stApp background reads through between tab-panel and inner containers;
     widen the white area by styling the inner block wrappers too. */
  [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] { background-color: transparent; }

  /* Vertical separator between legend column and right rail (Visualisation tab only) */
  [data-baseweb="tab-panel"]:first-of-type [data-testid="stHorizontalBlock"] >
    [data-testid="stColumn"]:last-child {
    border-left: 1px solid #111111;
    padding-left: 14px;
  }

  /* Plotly toolbar text contrast on white */
  .modebar { background-color: transparent !important; }
</style>
"""


# -----------------------------------------------------------------------------
# Cached heavy resources
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CLIP model weights…")
def _get_model_and_device():
    return load_clip_model()


@st.cache_resource(show_spinner="Reading dataset…")
def _get_dataset() -> Dataset:
    return load_dataset()


@st.cache_resource(show_spinner="Loading BioBERT for query encoding…")
def _get_query_encoder(_model: CLIPCITE, _device) -> QueryEncoder:
    return QueryEncoder(_model, _device)


@st.cache_resource(show_spinner="Computing cell embeddings…")
def _get_all_rna_embeddings(_model: CLIPCITE, _device, rna_encodings_path: str):
    ds = _get_dataset()
    return compute_all_rna_embeddings(_model, ds.rna_encodings, _device)


@st.cache_resource(show_spinner="Fitting UMAP (first run only; cached after)…")
def _get_umap(rna_emb_key: int):
    embs = _get_all_rna_embeddings(*_get_model_and_device(), str(config.RNA_ENCODINGS))
    reducer, coords = fit_or_load_umap(embs)
    return reducer, coords, embs


# -----------------------------------------------------------------------------
# Query input
# -----------------------------------------------------------------------------

def _query_input(protein_names: list[str]) -> str | None:
    """Centered free-text query input. Empty by default; placeholder serves as the prompt."""
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        text = st.text_input(
            "Query",
            value="",
            key="query_text_free",
            label_visibility="collapsed",
            placeholder="Search cells by protein expression",
        )
    return text.strip() or None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        "<h1 class='app-title'>Loss in translation</h1>"
        "<p class='app-subtitle'>Predicting protein expression from gene expression</p>",
        unsafe_allow_html=True,
    )

    ds = _get_dataset()
    model, device = _get_model_and_device()
    query_encoder = _get_query_encoder(model, device)
    reducer, cell_umap, all_rna_embs = _get_umap(rna_emb_key=ds.n_cells)

    if not ds.has_h5mu:
        st.warning(
            f"No .h5mu file found in {config.ROOT}. "
            "The UMAP works from the cached encodings, but cell-type coloring, "
            "heatmap, violin, and the protein builder dropdown will use a fallback list "
            "until the dataset is present."
        )

    query_text = _query_input(ds.protein_names)
    if not query_text:
        return  # Nothing else appears until a query is entered.

    query = _encode_query(query_text, query_encoder, all_rna_embs, reducer)

    ctx = VizContext(
        cell_umap=cell_umap,
        cell_types=ds.cell_types,
        n_cells=ds.n_cells,
        has_h5mu=ds.has_h5mu,
        obs=ds.obs,
    )

    st.write("")  # small spacer

    vis_tab, meta_tab, summary_tab = st.tabs(["Visualisation", "Metadata", "Summary"])

    with vis_tab:
        plot_col, legend_col, rail_col = st.columns([5, 1.2, 1.4])

        legend_container = None
        with rail_col:
            st.markdown("**View**")
            view = st.radio(
                "View",
                ["UMAP", "Heatmap", "Violin"],
                label_visibility="collapsed",
                key="selected_viz",
            )
            if view == "UMAP":
                st.markdown("---")
                st.radio(
                    "Color cells by",
                    ["Similarity score", "Cell type", "Disease Status (cohort)"],
                    key="umap_color_mode",
                )
                st.slider("Point size", 1, 6, 2, key="umap_point_size")
                st.slider(
                    "Top cell types in cohort",
                    min_value=0, max_value=10, value=1, step=1,
                    key="cohort_n_top",
                    help="Cohort = cells from the top-N cell types ranked by mean similarity to the query.",
                )

        with legend_col:
            if view == "UMAP":
                legend_container = st.container()

        with plot_col:
            if view == "UMAP":
                UMAPViz().render(ctx, query, None, legend_container=legend_container)
            elif view == "Heatmap":
                HeatmapViz().render(ctx, query, None)
            else:
                ViolinViz().render(ctx, query, None)

    with meta_tab:
        cohort_panel.render_cohort_panel(ctx, query, None)

    with summary_tab:
        summary.render_summary_tab(ctx, query)


def _encode_query(
    text: str,
    encoder: QueryEncoder,
    all_rna_embs: np.ndarray,
    reducer,
) -> QueryResult:
    key = ("enc", text)
    cache = st.session_state.setdefault("_query_cache", {})
    if key in cache:
        return cache[key]

    emb = encoder.encode(text)
    sims = similarity(emb, all_rna_embs)
    umap_point = project_query_to_umap(reducer, emb)
    qr = QueryResult(text=text, embedding=emb, similarity=sims, umap_point=umap_point)
    cache[key] = qr
    return qr


if __name__ == "__main__":
    main()
