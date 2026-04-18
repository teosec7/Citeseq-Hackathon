"""CITE-seq CLIP explorer.

Interactive Streamlit app for exploring a trained CLIP model that aligns
single-cell RNA embeddings with natural-language queries.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from citeseq_explorer import cohort_panel, config, summary
from citeseq_explorer.data import Dataset, load_dataset
from citeseq_explorer.embeddings import (
    QueryEncoder,
    compute_all_rna_embeddings,
    fit_or_load_umap,
    project_query_to_umap,
    similarity,
)
from citeseq_explorer.model import CLIP, load_clip_model
from citeseq_explorer.viz import QueryResult, VizContext
from citeseq_explorer.viz.heatmap_viz import HeatmapViz
from citeseq_explorer.viz.umap_viz import UMAPViz
from citeseq_explorer.viz.violin_viz import ViolinViz


st.set_page_config(
    page_title="Predicting protein expression from gene expression",
    layout="wide",
    initial_sidebar_state="collapsed",
)


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

  .stApp { background-color: #E0DED8 !important; }
  .stApp, body, p, span, label, div, h1, h2, h3, h4, h5, h6 { color: #111111 !important; }
  h1.app-title, h1.app-title * {
    -webkit-text-stroke: 0 !important;
  }
  h1.app-title .title-i {
    text-transform: lowercase !important;
    position: relative;
    display: inline-block;
  }
  h1.app-title .title-i::after {
    content: '';
    position: absolute;
    top: 0.12em;
    left: 50%;
    transform: translateX(-50%);
    width: 0.16em;
    height: 0.16em;
    background: #2EC4B6;
    border-radius: 50%;
  }
  [data-testid="stCaptionContainer"], .caption, small { color: #333333 !important; }

  [data-testid="stSidebar"] { display: none !important; }
  [data-testid="collapsedControl"] { display: none !important; }
  header[data-testid="stHeader"] { background: transparent; }

  div.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1400px;
  }

  .st-key-viz_card, .st-key-summary_card {
    background: #FFFFFF !important;
    border: 2px solid #2EC4B6;
    border-radius: 10px;
    padding: 1.25rem 1.25rem 1.5rem 1.25rem;
    margin-top: 2.5rem;
  }
  .st-key-summary_card { margin-top: 1.25rem; }

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
    margin-top: -1.2rem !important;
    margin-bottom: 3.5rem;
  }

  input[type="text"], textarea, [data-baseweb="input"] input {
    background-color: #FFFDF6 !important;
    color: #111111 !important;
    border-color: #111111 !important;
  }

  div[data-testid="stRadio"] label p { font-size: 1.05rem; }
  [data-baseweb="tag"] { background-color: #111111 !important; color: #E0DED8 !important; }

  .stApp, p, span, label, div, input, button, [data-testid="stMarkdownContainer"] p {
    font-size: 1.08rem;
  }
  h1.app-title, h1.app-title * { font-size: 4.5rem !important; }
  p.app-subtitle, p.app-subtitle * { font-size: 0.9rem !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
    font-size: 1.05rem !important;
    font-weight: 600;
  }

  .vis-divider {
    width: 1px;
    background: #111111;
    min-height: 900px;
    height: 100%;
    margin: 0 auto 0 0;
  }

  .modebar { background-color: transparent !important; }
</style>
"""


@st.cache_resource(show_spinner="Loading CLIP model weights…")
def _get_model_and_device():
    return load_clip_model()


@st.cache_resource(show_spinner="Reading dataset…")
def _get_dataset() -> Dataset:
    return load_dataset()


@st.cache_resource(show_spinner="Loading text encoder for query encoding…")
def _get_query_encoder(_model: CLIP, _device) -> QueryEncoder:
    return QueryEncoder(_model, _device)


@st.cache_resource(show_spinner="Computing cell embeddings…")
def _get_all_rna_embeddings(_model: CLIP, _device, rna_encodings_path: str):
    ds = _get_dataset()
    return compute_all_rna_embeddings(_model, ds.rna_encodings, _device)


@st.cache_resource(show_spinner="Fitting UMAP (first run only; cached after)…")
def _get_umap(rna_emb_key: int):
    embs = _get_all_rna_embeddings(*_get_model_and_device(), str(config.RNA_ENCODINGS))
    reducer, coords = fit_or_load_umap(embs)
    return reducer, coords, embs


def _query_input(protein_names: list[str]) -> str | None:
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        text = st.text_input(
            "Query",
            value="",
            key="query_text_free",
            label_visibility="collapsed",
            placeholder="This cell expresses … and does not express …",
        )
        st.markdown(
            "<div style='text-align:center;font-size:0.85rem;color:#444444;"
            "margin-top:-0.6rem'>"
            "e.g. This cell expresses CD3, CD4 and does not express CD8."
            "</div>",
            unsafe_allow_html=True,
        )
    return text.strip() or None


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        "<h1 class='app-title'>Loss <span class='title-i'>ı</span>n translation</h1>"
        "<p class='app-subtitle'>Predicting protein abundance from single-cell RNA data</p>",
        unsafe_allow_html=True,
    )

    ds = _get_dataset()
    model, device = _get_model_and_device()
    query_encoder = _get_query_encoder(model, device)
    reducer, cell_umap, all_rna_embs = _get_umap(rna_emb_key=ds.n_cells)

    if not ds.has_h5mu:
        st.warning(
            f"No .h5mu file found in {config.DATA_DIR}. "
            "The UMAP works from the cached encodings, but cell-type coloring, "
            "heatmap, violin, and the protein builder dropdown will use a fallback list "
            "until the dataset is present."
        )

    query_text = _query_input(ds.protein_names)
    if not query_text:
        return

    query = _encode_query(query_text, query_encoder, all_rna_embs, reducer)

    ctx = VizContext(
        cell_umap=cell_umap,
        cell_types=ds.cell_types,
        n_cells=ds.n_cells,
        has_h5mu=ds.has_h5mu,
        obs=ds.obs,
    )

    viz_card = st.container(key="viz_card")
    with viz_card:
        plot_col, legend_col, divider_col, rail_col = st.columns([5, 1.2, 0.06, 1.4])
    with divider_col:
        st.markdown("<div class='vis-divider'></div>", unsafe_allow_html=True)

    legend_container = None
    with rail_col:
        st.markdown("**View**")
        view = st.radio(
            "View",
            ["UMAP", "Heatmap", "Violin", "Bar plots"],
            label_visibility="collapsed",
            key="selected_viz",
        )
        if view == "UMAP":
            st.markdown("---")
            st.radio(
                "Color cells by",
                ["Cell type", "Similarity score", "Disease Status"],
                key="umap_color_mode",
            )
            st.slider("Point size", 1, 6, 2, key="umap_point_size")
        else:
            st.markdown("---")
        if view in ("UMAP", "Bar plots"):
            st.slider(
                "Top cell types",
                min_value=0, max_value=10, value=10, step=1,
                key="cohort_n_top",
                help="Cohort = cells from the top-N cell types ranked by mean similarity to the query.",
            )
        show_matched = ds.cell_types is not None and (
            view == "Bar plots"
            or (view == "UMAP" and st.session_state.get("umap_color_mode") == "Disease Status")
        )
        if show_matched:
            from citeseq_explorer.viz.base import cohort_mask_for_query
            n_top = int(st.session_state.get("cohort_n_top", 1))
            mask, chosen = cohort_mask_for_query(query, ds.cell_types, n_top_types=n_top)
            total = int(mask.sum())
            st.markdown(
                "<div style='text-align:center;color:#111111;"
                "font-size:1.05rem;font-weight:600;margin-top:0.6rem'>"
                f"Matched cell type(s)</div>"
                "<div style='text-align:center;color:#111111;"
                f"font-size:0.95rem;margin-top:0.15rem'>"
                f"{', '.join(chosen)} · n={total:,}</div>",
                unsafe_allow_html=True,
            )

    with legend_col:
        if view == "UMAP":
            legend_container = st.container()

    with plot_col:
        if view == "UMAP":
            UMAPViz().render(ctx, query, None, legend_container=legend_container)
        elif view == "Heatmap":
            HeatmapViz().render(ctx, query, None)
        elif view == "Violin":
            ViolinViz().render(ctx, query, None)
        else:
            cohort_panel.render_cohort_panel(ctx, query, None)

    with st.container(key="summary_card"):
        if ctx.cell_types is None:
            st.info("Summary needs CellType labels from the .h5mu file.")
        else:
            n_top = int(st.session_state.get("cohort_n_top", 1))
            facts = summary._build_facts(ctx, query, n_top)
            cache_key = (query.text, n_top, summary.OLLAMA_MODEL)
            cache = st.session_state.setdefault("_summary_cache", {})

            btn_col, out_col = st.columns([1, 3])
            with btn_col:
                regen = st.button("Generate summary", type="primary", use_container_width=True)
                if cache_key in cache and st.button("Clear", use_container_width=True):
                    cache.pop(cache_key, None)
                    st.rerun()
            with out_col:
                placeholder = st.empty()
                if regen or cache_key in cache:
                    if cache_key in cache and not regen:
                        placeholder.markdown(cache[cache_key])
                    else:
                        try:
                            text = summary._stream_summary(facts, placeholder)
                            cache[cache_key] = text
                        except Exception as e:
                            placeholder.error(
                                f"Could not reach Ollama at localhost:11434 — is `ollama serve` "
                                f"running and is `{summary.OLLAMA_MODEL}` pulled?\n\n{e}"
                            )


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
