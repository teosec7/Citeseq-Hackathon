"""Always-on cohort metadata panel shown below the viz tabs.

For each active query, the cohort is the cells belonging to the top-N cell
types ranked by mean similarity to the query. This module renders the cohort's
disease-Status composition and per-Subject breakdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import STATUS_LABELS, STATUS_ORDER
from .viz.base import QueryResult, VizContext, cohort_mask_for_query

STATUS_COLORS = {
    "Healthy": "#A8D5E2",
    "PSO":     "#F6C28B",
    "PSA":     "#F2A8A8",
    "PSX":     "#C5B0D5",
}


def render_cohort_panel(
    ctx: VizContext,
    query_a: QueryResult | None,
    query_b: QueryResult | None,
) -> None:
    if query_a is None:
        return
    if ctx.obs is None or ctx.cell_types is None:
        st.info("Cohort metadata needs the .h5mu file (Status/Subject columns).")
        return
    if "Status" not in ctx.obs.columns:
        st.info("No `Status` column found in the dataset's obs.")
        return

    n_top = int(st.session_state.get("cohort_n_top", 1))

    queries = [("A", query_a)] + ([("B", query_b)] if query_b is not None else [])
    cols = st.columns(len(queries))
    for col, (label, q) in zip(cols, queries):
        with col:
            _render_for_query(ctx, label, q, n_top)


def _render_for_query(ctx: VizContext, label: str, q: QueryResult, n_top: int) -> None:
    mask, chosen = cohort_mask_for_query(q, ctx.cell_types, n_top_types=n_top)
    total = int(mask.sum())

    if total == 0:
        st.warning("Cohort is empty.")
        return

    status_all = ctx.obs["Status"].astype(str).values
    cohort_status = status_all[mask]
    present = [s for s in STATUS_ORDER if s in np.unique(status_all)]

    st.plotly_chart(_status_chart(cohort_status, status_all, present), use_container_width=True)

    if "Subject" in ctx.obs.columns:
        st.markdown(
            "<hr style='border:none;border-top:1px solid #111111;margin:1.25rem 0;'>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _subject_chart(ctx.obs.loc[mask], present),
            use_container_width=True,
        )


def _status_chart(cohort_status: np.ndarray, status_all: np.ndarray, present: list[str]) -> go.Figure:
    total_cohort = max(len(cohort_status), 1)
    cohort_pct = [100 * (cohort_status == s).sum() / total_cohort for s in present]
    counts = [int((cohort_status == s).sum()) for s in present]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=present, y=cohort_pct,
            marker=dict(color=[STATUS_COLORS.get(s) for s in present]),
            text=[f"{p:.1f}%<br>n={c}" for p, c in zip(cohort_pct, counts)],
            textposition="outside",
            name="Cohort",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Disease Status composition", font=dict(size=18, color="#111111"), x=0.5, xanchor="center"),
        height=420,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Archivo Narrow, sans-serif", color="#111111", size=12),
        margin=dict(l=20, r=10, t=40, b=50),
        xaxis=dict(
            ticktext=[f"{s}<br>{STATUS_LABELS.get(s, s)}" for s in present],
            tickvals=present,
            showgrid=False, zeroline=False, showline=True, linecolor="#111111", linewidth=1,
            tickfont=dict(size=14, color="#111111"),
            title=dict(font=dict(size=15, color="#111111")),
        ),
        yaxis=dict(
            title=dict(text="% of matched cell type(s)", font=dict(size=15, color="#111111")),
            tickfont=dict(size=14, color="#111111"),
            showgrid=False, zeroline=False, showline=True, linecolor="#111111", linewidth=1,
        ),
        showlegend=False,
    )
    return fig


def _subject_chart(cohort_obs: pd.DataFrame, present: list[str]) -> go.Figure:
    subj = cohort_obs["Subject"].astype(str)
    status = cohort_obs["Status"].astype(str)

    counts = (
        pd.crosstab(subj, status)
        .reindex(columns=present, fill_value=0)
    )
    subj_status = cohort_obs.groupby("Subject")["Status"].first().astype(str)
    order_key = pd.DataFrame({
        "subject": counts.index,
        "status_rank": [STATUS_ORDER.index(subj_status.get(s, present[0])) if subj_status.get(s) in STATUS_ORDER else 99 for s in counts.index],
        "total": counts.sum(axis=1).values,
    }).sort_values(["status_rank", "total"], ascending=[True, False])
    counts = counts.loc[order_key["subject"].values]

    fig = go.Figure()
    for s in present:
        fig.add_trace(
            go.Bar(
                x=counts.index.astype(str),
                y=counts[s].values,
                name=s,
                marker=dict(color=STATUS_COLORS.get(s)),
                hovertemplate="%{x}: %{y} cells<extra>" + s + "</extra>",
            )
        )
    fig.update_layout(
        title=dict(text="Cells per Patient (coloured by Disease status)", font=dict(size=18, color="#111111"), x=0.5, xanchor="center"),
        barmode="stack",
        height=420,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Archivo Narrow, sans-serif", color="#111111", size=12),
        margin=dict(l=20, r=100, t=50, b=80),
        xaxis=dict(
            title=dict(text="Patient", font=dict(size=15, color="#111111")),
            tickangle=-90,
            tickfont=dict(size=13, color="#111111"),
            showgrid=False, zeroline=False, showline=True, linecolor="#111111", linewidth=1,
        ),
        yaxis=dict(
            title=dict(text="cells in matched cell type(s)", font=dict(size=15, color="#111111")),
            tickfont=dict(size=14, color="#111111"),
            showgrid=False, zeroline=False, showline=True, linecolor="#111111", linewidth=1,
        ),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    )
    return fig
