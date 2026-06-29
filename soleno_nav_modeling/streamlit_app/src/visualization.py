"""Graphiques Plotly enrichis pour le MVP (outliers, ECDF, violons)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.outlier_viz import (
    POINT_STATUS_COLORS,
    POINT_STATUS_LABELS,
    classify_point_status,
    classify_row_status,
    column_summary_stats,
    percentile_bounds,
    physical_bounds,
)


def _add_percentile_lines(fig: go.Figure, stats: dict, row: int = 1, col: int = 1) -> None:
    for p, color, name in (
        (stats.get("p5"), "#54A24B", "P5"),
        (stats.get("median"), "#72B7B2", "Médiane"),
        (stats.get("p95"), "#54A24B", "P95"),
    ):
        if p is None or np.isnan(p):
            continue
        fig.add_vline(
            x=p,
            line_dash="dot",
            line_color=color,
            opacity=0.8,
            annotation_text=name,
            annotation_position="top",
            row=row,
            col=col,
        )


def _add_physical_lines(fig: go.Figure, column: str, row: int = 1, col: int = 1) -> None:
    lo, hi = physical_bounds(column)
    for val, label in ((lo, "min phys."), (hi, "max phys.")):
        if val is None:
            continue
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color="#E45756",
            opacity=0.9,
            annotation_text=label,
            annotation_position="bottom",
            row=row,
            col=col,
        )


def distribution_dual_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    log_scale: bool = False,
) -> go.Figure | None:
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return None

    stats = column_summary_stats(df[column], column)
    title = (
        f"Distribution — {column} | n={stats['n']:,} | "
        f"IQR outliers={stats['iqr_outliers']} | hors plage phys.≈{stats['physical_outliers']}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Toutes les valeurs", "Zoom P5–P95"),
    )

    x_full = np.log1p(s.clip(lower=0)) if log_scale else s
    x_zoom = s[(s >= stats["p5"]) & (s <= stats["p95"])]
    if log_scale:
        x_zoom = np.log1p(x_zoom.clip(lower=0))

    x_label = f"log1p({column})" if log_scale else column

    fig.add_trace(
        go.Histogram(x=x_full, nbinsx=40, marker_color="#4C78A8", name="Toutes"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=x_zoom, nbinsx=35, marker_color="#72B7B2", name="Zoom"),
        row=1,
        col=2,
    )

    if not log_scale:
        _add_percentile_lines(fig, stats, row=1, col=2)
        _add_physical_lines(fig, column, row=1, col=2)

    fig.update_layout(
        title=title,
        showlegend=False,
        height=420,
        bargap=0.05,
    )
    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_xaxes(title_text=x_label, row=1, col=2)
    return fig


def ecdf_comparison(
    df: pd.DataFrame,
    column: str,
) -> go.Figure | None:
    s_all = pd.to_numeric(df[column], errors="coerce").dropna().sort_values()
    if s_all.empty:
        return None

    fig = go.Figure()
    y_all = np.arange(1, len(s_all) + 1) / len(s_all)
    fig.add_trace(
        go.Scatter(
            x=s_all,
            y=y_all,
            mode="lines",
            name="Toutes les données",
            line=dict(color="#4C78A8", width=2),
        )
    )

    if "is_outlier_feature" in df.columns:
        mask = ~df["is_outlier_feature"].fillna(False)
        s_clean = pd.to_numeric(df.loc[mask, column], errors="coerce").dropna().sort_values()
        if len(s_clean) >= 5:
            y_clean = np.arange(1, len(s_clean) + 1) / len(s_clean)
            fig.add_trace(
                go.Scatter(
                    x=s_clean,
                    y=y_clean,
                    mode="lines",
                    name="Sans outliers pipeline",
                    line=dict(color="#54A24B", width=2, dash="dash"),
                )
            )

    stats = column_summary_stats(df[column], column)
    for p, lbl in ((stats.get("p5"), "P5"), (stats.get("median"), "P50"), (stats.get("p95"), "P95")):
        if p is not None:
            fig.add_vline(x=p, line_dash="dot", line_color="#B279A2", opacity=0.7, annotation_text=lbl)

    lo, hi = physical_bounds(column)
    if hi is not None:
        fig.add_vline(x=hi, line_dash="dash", line_color="#E45756", annotation_text="max phys.")
    if lo is not None:
        fig.add_vline(x=lo, line_dash="dash", line_color="#E45756", annotation_text="min phys.")

    fig.update_layout(
        title=f"ECDF — {column} (proportion de lots ≤ x)",
        xaxis_title=column,
        yaxis_title="Proportion cumulée",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def violin_boxplot_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    clip_percentiles: tuple[float, float] = (1, 99),
) -> go.Figure | None:
    sub = df[[value_col, group_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return None

    if sub[group_col].nunique() > 20:
        top = sub[group_col].astype(str).value_counts().head(15).index
        sub = sub[sub[group_col].astype(str).isin(top)]

    lo, hi = percentile_bounds(sub[value_col], clip_percentiles[0], clip_percentiles[1])
    n_outside = int(((sub[value_col] < lo) | (sub[value_col] > hi)).sum())

    fig = px.violin(
        sub,
        x=group_col,
        y=value_col,
        box=True,
        points="outliers",
        color=group_col,
        title=(
            f"{value_col} par {group_col} | violon + box | "
            f"{n_outside} points hors P{clip_percentiles[0]}–P{clip_percentiles[1]} masqués visuellement"
        ),
    )
    fig.update_layout(
        showlegend=False,
        height=480,
        yaxis=dict(range=[lo, hi]),
        xaxis_tickangle=-35,
    )
    return fig


def scatter_colored(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hover_cols: list[str] | None = None,
    clip_percentiles: tuple[float, float] = (1, 99),
) -> go.Figure | None:
    cols = [x, y]
    extra = [c for c in (hover_cols or []) if c in df.columns and c not in cols]
    sub = df[cols + extra].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=cols)
    if sub.empty:
        return None

    sub["_status"] = classify_row_status(df.loc[sub.index], cols)
    sub["_status_label"] = sub["_status"].map(POINT_STATUS_LABELS)

    lo_x, hi_x = percentile_bounds(sub[x], clip_percentiles[0], clip_percentiles[1])
    lo_y, hi_y = percentile_bounds(sub[y], clip_percentiles[0], clip_percentiles[1])

    fig = px.scatter(
        sub,
        x=x,
        y=y,
        color="_status_label",
        color_discrete_map={POINT_STATUS_LABELS[k]: POINT_STATUS_COLORS[k] for k in POINT_STATUS_COLORS},
        hover_data=extra if extra else None,
        title=f"{x} vs {y} — points colorés par qualité",
    )
    fig.update_layout(
        height=480,
        legend_title_text="Statut",
        xaxis=dict(range=[lo_x, hi_x]),
        yaxis=dict(range=[lo_y, hi_y]),
    )
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    *,
    method: str = "pearson",
    winsorize: bool = False,
    p_low: float = 1,
    p_high: float = 99,
) -> go.Figure | None:
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    if winsorize:
        for c in num.columns:
            lo, hi = percentile_bounds(num[c], p_low, p_high)
            num[c] = num[c].clip(lo, hi)
    if method == "spearman":
        corr = num.corr(method="spearman")
        title = f"Corrélation Spearman (winsorisé P{p_low}–P{p_high})" if winsorize else "Corrélation Spearman"
    else:
        corr = num.corr(method="pearson")
        title = f"Corrélation Pearson (winsorisé P{p_low}–P{p_high})" if winsorize else "Corrélation Pearson"
    if corr.empty:
        return None
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title=title, height=520)
    return fig


def status_breakdown_bar(df: pd.DataFrame, column: str) -> go.Figure | None:
    if column not in df.columns:
        return None
    status = classify_point_status(df, column)
    counts = status.value_counts().reindex(POINT_STATUS_LABELS.keys(), fill_value=0)
    labels = [POINT_STATUS_LABELS.get(k, k) for k in counts.index]
    colors = [POINT_STATUS_COLORS.get(k, "#999") for k in counts.index]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts.values,
            marker_color=colors,
            text=counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Répartition qualité — {column}",
        yaxis_title="Nombre de lots",
        height=360,
    )
    return fig


def bar_metrics(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None):
    if df.empty or y not in df.columns:
        return None
    return px.bar(df, x=x, y=y, color=color, title=title)


def reliability_comparison_bars(
    metrics_long: pd.DataFrame,
    target: str,
    metric: str = "r2_mean",
) -> go.Figure | None:
    """Compare random CV vs group supplier pour une cible."""
    if metrics_long.empty:
        return None
    sub = metrics_long[
        (metrics_long["target"].astype(str) == target)
        & (metrics_long["model_version"].astype(str) == "A")
    ].copy()
    if sub.empty:
        sub = metrics_long[metrics_long["target"].astype(str) == target].copy()
    schemes = ["kfold_random", "group_supplier", "group_grade", "temporal"]
    sub = sub[sub["validation_scheme"].isin(schemes)]
    if sub.empty:
        return None
    sub = sub.sort_values("validation_priority")
    fig = px.bar(
        sub,
        x="validation_label",
        y=metric,
        color="validation_scheme",
        title=f"{target} — {metric} par protocole de validation",
        labels={metric: metric.replace("_", " ").upper(), "validation_label": "Protocole"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(showlegend=False, height=420, xaxis_tickangle=-25)
    return fig


def parity_plot(
    pred_df: pd.DataFrame,
    target: str,
    title: str | None = None,
) -> go.Figure | None:
    """Prédit vs réel à partir des prédictions CV robustes."""
    if pred_df.empty:
        return None
    sub = pred_df[pred_df["target"].astype(str) == target].copy()
    if sub.empty or "y_true" not in sub.columns or "y_pred" not in sub.columns:
        return None
    sub["y_true"] = pd.to_numeric(sub["y_true"], errors="coerce")
    sub["y_pred"] = pd.to_numeric(sub["y_pred"], errors="coerce")
    sub = sub.dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        return None
    scheme = sub["validation_scheme"].iloc[0] if "validation_scheme" in sub.columns else ""
    fig = px.scatter(
        sub,
        x="y_true",
        y="y_pred",
        opacity=0.55,
        title=title or f"Parity plot — {target} ({scheme})",
        labels={"y_true": "Mesuré", "y_pred": "Prédit (CV)"},
    )
    lo = min(sub["y_true"].min(), sub["y_pred"].min())
    hi = max(sub["y_true"].max(), sub["y_pred"].max())
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y=x",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(height=460)
    return fig


def deployment_status_chart(deployment: pd.DataFrame) -> go.Figure | None:
    """Vue synthétique des statuts de déploiement."""
    if deployment.empty or "deployment_status" not in deployment.columns:
        return None
    from config import DEPLOYMENT_STATUS_COLORS, DEPLOYMENT_STATUS_LABELS

    sub = deployment.copy()
    if "model_version" in sub.columns:
        sub = sub[sub["model_version"].astype(str) == "A"]
    sub["label"] = sub["deployment_status"].map(DEPLOYMENT_STATUS_LABELS).fillna(sub["deployment_status"])
    sub["color"] = sub["deployment_status"].map(DEPLOYMENT_STATUS_COLORS).fillna("#999")
    fig = go.Figure(
        go.Bar(
            x=sub["target"],
            y=[1] * len(sub),
            marker_color=sub["color"],
            text=sub["label"],
            textposition="outside",
            hovertext=sub.get("status_reason", sub["label"]),
        )
    )
    fig.update_layout(
        title="Statut de déploiement par cible (modèle A)",
        yaxis_visible=False,
        height=380,
    )
    return fig


# Rétrocompatibilité
def histogram(df: pd.DataFrame, column: str):
    return distribution_dual_histogram(df, column)


def boxplot_by_group(df: pd.DataFrame, value_col: str, group_col: str):
    return violin_boxplot_by_group(df, value_col, group_col)


def scatter(df: pd.DataFrame, x: str, y: str):
    hover = [c for c in ("numero", "supplier_code", "description_fr") if c in df.columns]
    return scatter_colored(df, x, y, hover_cols=hover)
