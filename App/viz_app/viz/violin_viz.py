import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .base import QueryResult, VizContext


class ViolinViz:
    name = "Violin"
    description = "Distribution of similarity scores within each cell type."
    requires_cell_types = True
    requires_query = True

    def render(self, ctx: VizContext, query_a: QueryResult | None, query_b: QueryResult | None) -> None:
        if query_a is None:
            st.info("Enter a query above to see this view.")
            return
        if ctx.cell_types is None:
            st.info("Violin plot needs CellType labels from the .h5mu file.")
            return

        unique_types = sorted(np.unique(ctx.cell_types).tolist())
        means = {ct: float(query_a.similarity[ctx.cell_types == ct].mean()) for ct in unique_types}
        unique_types = sorted(unique_types, key=lambda c: -means[c])

        fig = go.Figure()
        for ct in unique_types:
            mask = ctx.cell_types == ct
            fig.add_trace(
                go.Violin(
                    y=query_a.similarity[mask],
                    x=[ct] * int(mask.sum()),
                    name=ct,
                    line_color="#1f4e79",
                    fillcolor="rgba(70,130,180,0.55)",
                    points=False,
                    width=0.85,
                    spanmode="hard",
                    showlegend=False,
                )
            )

        fig.update_traces(meanline_visible=True, box_visible=False)
        fig.update_layout(
            violinmode="group",
            violingap=0.05,
            violingroupgap=0.05,
            height=700,
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Archivo Narrow, sans-serif", color="#111111", size=12),
            margin=dict(l=30, r=10, t=20, b=120),
            xaxis=dict(tickangle=-45, title="", showgrid=False, zeroline=False),
            yaxis=dict(title="cosine similarity", showgrid=False, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)
