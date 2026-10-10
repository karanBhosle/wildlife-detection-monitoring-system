"""Reusable Streamlit UI components."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui.theme import COLORS, inject_global_css


def apply_theme() -> None:
    st.markdown(f"<style>{inject_global_css()}</style>", unsafe_allow_html=True)


def top_brand_bar(brand_name: str, tagline: str | None = None) -> None:
    tagline_html = (
        f'<span class="kv-topbar-tagline">{tagline}</span>' if tagline else ""
    )
    st.markdown(
        f"""
        <div class="kv-topbar">
            <div class="kv-topbar-left">
                <span class="brand-karanverse kv-topbar-brand">{brand_name}</span>
                {tagline_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand(
    brand_name: str,
    product_title: str,
    subtitle: str = "YOLO11 · ByteTrack · OpenCV",
) -> None:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="brand-karanverse">{brand_name}</div>
            <div class="sidebar-brand-title">{product_title}</div>
            <div class="sidebar-brand-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_status_pill(loaded: bool, device: str) -> None:
    dot_color = COLORS["accent"] if loaded else COLORS["danger"]
    label = "Model ready" if loaded else "Model offline"
    st.markdown(
        f"""
        <div class="status-pill" style="margin-top:1rem;">
            <span class="status-dot" style="background:{dot_color}; box-shadow:0 0 8px {dot_color};"></span>
            {label} · {device}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_hero(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f'<span class="hero-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="hero-banner">
            <p class="hero-title">{title}</p>
            <p class="hero-sub">{subtitle}</p>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_cards(items: list[dict[str, Any]]) -> None:
    cards = []
    for item in items:
        hint = item.get("hint", "")
        cards.append(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{item['label']}</div>
                <div class="kpi-value">{item['value']}</div>
                <div class="kpi-hint">{hint}</div>
            </div>
            """
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def panel_start(title: str) -> None:
    st.markdown(f'<div class="panel"><p class="panel-title">{title}</p>', unsafe_allow_html=True)


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def empty_state(title: str, message: str, icon: str = "📭") -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.bar(df, x=x, y=y, title=title, color=y, color_continuous_scale=[COLORS["surface_2"], COLORS["accent"]])
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=360)
    return fig


def styled_pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
    fig = px.pie(df, names=names, values=values, title=title, hole=0.45, color_discrete_sequence=COLORS["chart"])
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=360)
    return fig


def styled_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.area(df, x=x, y=y, title=title, color_discrete_sequence=[COLORS["accent"]])
    fig.update_traces(line=dict(width=2))
    fig.update_layout(height=320)
    return fig


def detection_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No rows to display.")
        return
    display_cols = [c for c in ["display_name", "class_name", "confidence", "track_id", "x1", "y1", "x2", "y2"] if c in df.columns]
    st.dataframe(
        df[display_cols] if display_cols else df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
            "display_name": st.column_config.TextColumn("Species", width="medium"),
        },
    )


def session_confidence(default: float) -> float:
    if "ui_confidence" not in st.session_state:
        st.session_state.ui_confidence = float(default)
    return float(st.session_state.ui_confidence)


def confidence_control(label: str, default: float, key: str) -> float:
    val = st.slider(
        label,
        0.05,
        0.95,
        session_confidence(default),
        0.05,
        key=key,
        help="Minimum score to keep a detection.",
    )
    st.session_state.ui_confidence = val
    return val
