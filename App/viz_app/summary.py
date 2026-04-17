"""LLM-generated summary tab.

Builds a numbers-only fact sheet from the active query + cohort, then asks a
local Ollama model to write a short paragraph. The prompt is intentionally
restrictive — the LLM is told to use only the supplied numbers and to avoid
inventing marker-gene biology.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from .config import STATUS_LABELS, STATUS_ORDER
from .viz.base import QueryResult, VizContext, cohort_mask_for_query, rank_cell_types

OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You summarise single-cell results from a CITE-seq exploration app. "
    "You will be given a small JSON fact sheet. Write a short paragraph "
    "(3–4 sentences) that simply states the facts in plain English. Rules:\n"
    "  • Never use first person ('I', 'we', 'us', 'our'). Never address the "
    "reader ('you'). State facts in the third person only.\n"
    "  • Simple words. No jargon unless it's already in the facts.\n"
    "  • Use ONLY the numbers and labels given. Don't invent genes or biology.\n"
    "  • State: the query, the cell type(s) it matched, the cohort size, and "
    "whether any disease group is over- or under-represented vs the whole "
    "dataset ('more', 'less', or 'about the same').\n"
    "  • No bullet points, no headings, no markdown. One short paragraph."
)


def render_summary_tab(ctx: VizContext, query: QueryResult) -> None:
    if ctx.cell_types is None:
        st.info("Summary needs CellType labels from the .h5mu file.")
        return

    n_top = int(st.session_state.get("cohort_n_top", 1))
    facts = _build_facts(ctx, query, n_top)

    cache_key = (query.text, n_top, OLLAMA_MODEL)
    cache = st.session_state.setdefault("_summary_cache", {})

    cols = st.columns([1, 1, 6])
    with cols[0]:
        regen = st.button("Generate", type="primary")
    with cols[1]:
        if cache_key in cache and st.button("Clear"):
            cache.pop(cache_key, None)
            st.rerun()

    if regen or cache_key in cache:
        if cache_key in cache and not regen:
            st.markdown(cache[cache_key])
        else:
            placeholder = st.empty()
            try:
                text = _stream_summary(facts, placeholder)
                cache[cache_key] = text
            except Exception as e:
                placeholder.error(
                    f"Could not reach Ollama at localhost:11434 — is `ollama serve` "
                    f"running and is `{OLLAMA_MODEL}` pulled?\n\n{e}"
                )


def _build_facts(ctx: VizContext, query: QueryResult, n_top: int) -> dict:
    ranked = rank_cell_types(query, ctx.cell_types)
    cohort_mask, chosen = cohort_mask_for_query(query, ctx.cell_types, n_top_types=max(n_top, 1))

    facts: dict = {
        "query": query.text,
        "n_total_cells": int(ctx.n_cells),
        "top_cell_types_by_mean_similarity": [
            {"cell_type": ct, "mean_similarity": round(s, 4)}
            for ct, s in ranked[:10]
        ],
        "cohort": {
            "n_top_used": n_top,
            "cell_types_in_cohort": chosen,
            "cohort_n_cells": int(cohort_mask.sum()),
        },
    }

    if ctx.obs is not None and "Status" in ctx.obs.columns:
        status_all = ctx.obs["Status"].astype(str).values
        cohort_status = status_all[cohort_mask]
        present = [s for s in STATUS_ORDER if s in np.unique(status_all)]
        total_cohort = max(len(cohort_status), 1)
        total_all = len(status_all)
        facts["disease_status"] = [
            {
                "status": s,
                "label": STATUS_LABELS.get(s, s),
                "cohort_pct": round(100 * (cohort_status == s).sum() / total_cohort, 1),
                "cohort_n": int((cohort_status == s).sum()),
                "whole_dataset_pct": round(100 * (status_all == s).sum() / total_all, 1),
            }
            for s in present
        ]

    return facts


def _stream_summary(facts: dict, placeholder) -> str:
    import ollama  # imported lazily so the app still runs without ollama installed

    user_prompt = (
        "Here is the fact sheet:\n\n"
        f"{facts}\n\n"
        "Write the paragraph now."
    )
    accumulated = ""
    for chunk in ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    ):
        token = chunk.get("message", {}).get("content", "")
        if token:
            accumulated += token
            placeholder.markdown(accumulated)
    return accumulated
