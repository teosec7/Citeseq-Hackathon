import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .base import QueryResult, VizContext


class HeatmapViz:
    name = "Heatmap"
    description = "Mean query–cell-type similarity. Needs the CITE-seq data to be loaded."
    requires_cell_types = True
    requires_query = True

    def render(self, ctx: VizContext, query_a: QueryResult | None, query_b: QueryResult | None) -> None:
        if query_a is None:
            st.info("Enter a query above to see this view.")
            return
        if ctx.cell_types is None:
            st.info("Heatmap needs CellType labels from the .h5mu file. "
                    "Drop the dataset in the App folder and restart the app.")
            return

        unique_types = sorted(np.unique(ctx.cell_types).tolist())

        means = np.array(
            [float(query_a.similarity[ctx.cell_types == ct].mean()) for ct in unique_types],
            dtype=np.float32,
        )
        order = np.argsort(-means)
        unique_types = [unique_types[i] for i in order]
        means = means[order]

        hover = [
            f"{query_a.text}<br>{ct}<br>mean sim={m:.3f}"
            for ct, m in zip(unique_types, means)
        ]
        fig = go.Figure(
            data=go.Heatmap(
                z=[means],
                x=unique_types,
                y=["mean sim"],
                colorscale="RdBu_r",
                zmid=0.0,
                text=[[f"{v:.2f}" for v in means]],
                texttemplate="%{text}",
                textfont=dict(size=10),
                hovertext=[hover],
                hoverinfo="text",
                colorbar=dict(title="mean sim"),
            )
        )
        fig.update_layout(
            height=180,
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Archivo Narrow, sans-serif", color="#111111", size=12),
            margin=dict(l=30, r=10, t=20, b=100),
            xaxis=dict(tickangle=-45, showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)
