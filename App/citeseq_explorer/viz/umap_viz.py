import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..config import STATUS_LABELS, STATUS_ORDER
from .base import QueryResult, VizContext, cohort_mask_for_query

STATUS_COLORS = {
    "Healthy": "#2EC4B6",
    "PSO":     "#F77F00",
    "PSA":     "#D62828",
    "PSX":     "#6A4C93",
}


class UMAPViz:
    name = "UMAP"
    description = (
        "Cells in 2D, colored by query similarity, cell type, or — for the query's matched "
        "cohort only — by disease Status."
    )
    requires_cell_types = False
    requires_query = True

    def render(
        self,
        ctx: VizContext,
        query_a: QueryResult | None,
        query_b: QueryResult | None,
        legend_container=None,
    ) -> None:
        if query_a is None:
            st.info("Enter a query above to see this view.")
            return

        options = ["Similarity score"]
        if ctx.has_h5mu:
            options.append("Cell type")
        has_status = (
            ctx.obs is not None
            and "Status" in ctx.obs.columns
            and ctx.cell_types is not None
        )
        if has_status:
            options.append("Disease Status")

        color_mode = st.session_state.get("umap_color_mode", "Similarity score")
        if color_mode not in options:
            color_mode = options[0]
        point_size = int(st.session_state.get("umap_point_size", 2))
        n_top = int(st.session_state.get("cohort_n_top", 1))

        fig, legend_items = _build_figure(
            ctx=ctx, q=query_a,
            color_mode=color_mode,
            point_size=point_size,
            n_top=n_top,
        )
        st.plotly_chart(fig, use_container_width=False)

        if legend_container is not None:
            with legend_container:
                _render_legend(legend_items, color_mode)


def _build_figure(
    ctx: VizContext,
    q: QueryResult,
    color_mode: str,
    point_size: int,
    n_top: int,
) -> tuple[go.Figure, list[dict]]:
    xy = ctx.cell_umap
    fig = go.Figure()
    legend_items: list[dict] = []

    if color_mode == "Cell type" and ctx.cell_types is not None:
        types = ctx.cell_types
        unique = list(np.unique(types))
        palette = _qualitative_palette(len(unique))
        centroids: list[tuple[str, float, float, int]] = []
        for i, ct in enumerate(unique):
            mask = types == ct
            color = palette[i]
            fig.add_trace(
                go.Scattergl(
                    x=xy[mask, 0], y=xy[mask, 1],
                    mode="markers", name=str(ct),
                    marker=dict(size=point_size, opacity=0.6, color=color),
                    hovertext=[str(ct)] * int(mask.sum()),
                    hoverinfo="text",
                )
            )
            legend_items.append({"name": str(ct), "color": color})
            cx = float(np.median(xy[mask, 0]))
            cy = float(np.median(xy[mask, 1]))
            centroids.append((str(ct), cx, cy, int(mask.sum())))

        for name, lx, ly in _place_labels(centroids, xy):
            fig.add_annotation(
                x=lx, y=ly, text=name, showarrow=False,
                font=dict(size=11, color="#111111", family="Archivo Narrow, sans-serif"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.35)",
                borderwidth=0.6, borderpad=2,
            )
    elif color_mode == "Disease Status":
        cohort, chosen = cohort_mask_for_query(q, ctx.cell_types, n_top_types=n_top)
        status_all = ctx.obs["Status"].astype(str).values
        bg = ~cohort
        fig.add_trace(
            go.Scattergl(
                x=xy[bg, 0], y=xy[bg, 1],
                mode="markers", name="other cells",
                marker=dict(size=max(1, point_size - 1), color="#BFBFBF", opacity=0.18),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        legend_items.append({"name": "Other cells", "color": "#BFBFBF"})

        status_cohort = status_all[cohort]
        xy_cohort = xy[cohort]
        total = int(cohort.sum()) or 1
        present = [s for s in STATUS_ORDER if s in np.unique(status_cohort)]
        for s in present:
            m = status_cohort == s
            n = int(m.sum())
            color = STATUS_COLORS.get(s, "#444")
            fig.add_trace(
                go.Scattergl(
                    x=xy_cohort[m, 0], y=xy_cohort[m, 1],
                    mode="markers",
                    name=f"{s} · {n} ({100*n/total:.0f}%)",
                    marker=dict(size=point_size, color=color, opacity=0.85),
                    hovertext=[f"{s} — {STATUS_LABELS.get(s, s)}"] * n,
                    hoverinfo="text",
                )
            )
            legend_items.append({
                "name": f"{s} · {n:,} ({100*n/total:.0f}%)",
                "sub": STATUS_LABELS.get(s, s),
                "color": color,
            })
    else:
        legend_items.append({
            "name": "Cells",
            "sub": "color = cosine similarity",
            "color": "#444444",
            "shape": "gradient",
        })
        sims = q.similarity
        hover = [
            f"{ct}<br>sim={s:.3f}"
            if ctx.cell_types is not None else f"sim={s:.3f}"
            for ct, s in zip(
                (ctx.cell_types if ctx.cell_types is not None else [""] * len(sims)),
                sims,
            )
        ]
        fig.add_trace(
            go.Scattergl(
                x=xy[:, 0], y=xy[:, 1],
                mode="markers",
                marker=dict(
                    size=point_size, color=sims,
                    colorscale=[
                        [0.0, "#E58A8A"],
                        [0.5, "#F5F1E6"],
                        [1.0, "#2EC4B6"],
                    ],
                    cmid=0.0,
                    showscale=True, colorbar=dict(title="sim", x=1.02, len=0.7),
                    opacity=0.75,
                ),
                hovertext=hover, hoverinfo="text", name="cells",
            )
        )

    if q.umap_point is not None and color_mode != "Disease Status":
        qx, qy = float(q.umap_point[0]), float(q.umap_point[1])
        fig.add_annotation(
            x=qx, y=qy, text="★", showarrow=False,
            font=dict(size=30, color="black", family="Archivo Narrow, sans-serif"),
        )
        fig.add_annotation(
            x=qx, y=qy, text="★", showarrow=False,
            font=dict(size=24, color="gold", family="Archivo Narrow, sans-serif"),
        )
        legend_items.insert(0, {"name": "Query position", "color": "gold", "shape": "star"})

    fig.update_layout(
        width=850, height=850, autosize=False,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Archivo Narrow, sans-serif", color="#111111", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            title="UMAP1", showgrid=False, zeroline=False,
            showticklabels=False, scaleanchor="y", scaleratio=1,
            showline=True, linecolor="#111111", linewidth=1.5, mirror=False,
        ),
        yaxis=dict(
            title="UMAP2", showgrid=False, zeroline=False,
            showticklabels=False,
            showline=True, linecolor="#111111", linewidth=1.5, mirror=False,
        ),
    )
    return fig, legend_items


def _place_labels(
    centroids: list[tuple[str, float, float, int]],
    xy: np.ndarray,
) -> list[tuple[str, float, float]]:
    extent = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])))
    min_dist = max(extent * 0.06, 0.5)

    items = sorted(centroids, key=lambda t: -t[3])
    placed: list[tuple[str, float, float]] = []
    for name, x, y, _ in items:
        for _ in range(40):
            collision = None
            for _, px, py in placed:
                d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                if d < min_dist:
                    collision = (px, py, d)
                    break
            if collision is None:
                break
            px, py, d = collision
            if d < 1e-6:
                x += min_dist
            else:
                push = (min_dist - d) + 0.02 * extent
                x += (x - px) / d * push
                y += (y - py) / d * push
        placed.append((name, x, y))
    return placed


def _qualitative_palette(n: int) -> list[str]:
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194",
        "#6b6ecf", "#b5cf6b",
    ]
    if n <= len(base):
        return base[:n]
    return (base * (n // len(base) + 1))[:n]


def _render_legend(items: list[dict], color_mode: str) -> None:
    if not items:
        return
    st.markdown("**Legend**")
    rows = ["<div style='max-height:820px;overflow-y:auto;padding-right:6px'>"]
    for item in items:
        shape = item.get("shape", "square")
        if shape == "star":
            swatch = (
                "<svg width='18' height='18' viewBox='0 0 24 24' "
                "style='display:inline-block;vertical-align:middle;margin-right:8px'>"
                "<polygon points='12,2 14.6,9 22,9.3 16.2,13.9 18.2,21 12,17 5.8,21 "
                "7.8,13.9 2,9.3 9.4,9' fill='gold' stroke='black' "
                "stroke-width='1.5' stroke-linejoin='round'/></svg>"
            )
        elif shape == "gradient":
            swatch = (
                "<span style='display:inline-block;width:16px;height:16px;"
                "background:linear-gradient(90deg,#2166ac,#f7f7f7,#b2182b);"
                "border:1px solid #999;border-radius:3px;margin-right:8px;"
                "vertical-align:middle'></span>"
            )
        else:
            swatch = (
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{item['color']};border-radius:3px;margin-right:8px;"
                f"vertical-align:middle'></span>"
            )
        sub = (
            f"<div style='font-size:0.78rem;opacity:0.65;margin-left:24px;"
            f"line-height:1.1'>{item['sub']}</div>"
            if item.get("sub") else ""
        )
        rows.append(
            f"<div style='margin-bottom:6px;font-size:0.95rem'>"
            f"{swatch}<span style='vertical-align:middle'>{item['name']}</span>"
            f"{sub}</div>"
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
