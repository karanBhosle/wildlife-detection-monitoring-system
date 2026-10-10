"""Visual theme: karanverse CSS tokens and Plotly styling."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# karanverse — deep space indigo with violet + aurora cyan accents
COLORS = {
    "bg": "#08091A",
    "surface": "#11132A",
    "surface_2": "#171A38",
    "border": "#2A2F5A",
    "text": "#E8EAFF",
    "muted": "#9BA3D9",
    "accent": "#7C5CFF",
    "accent_2": "#38BDF8",
    "danger": "#FF6B8A",
    "info": "#818CF8",
    "chart": ["#7C5CFF", "#38BDF8", "#F472B6", "#A78BFA", "#34D399", "#FBBF24", "#60A5FA"],
}

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, system-ui, sans-serif", color=COLORS["text"], size=13),
        colorway=COLORS["chart"],
        margin=dict(l=24, r=24, t=48, b=24),
        xaxis=dict(
            gridcolor="rgba(42,47,90,0.55)",
            zerolinecolor="rgba(42,47,90,0.75)",
            linecolor=COLORS["border"],
        ),
        yaxis=dict(
            gridcolor="rgba(42,47,90,0.55)",
            zerolinecolor="rgba(42,47,90,0.75)",
            linecolor=COLORS["border"],
        ),
        title=dict(font=dict(size=16, color=COLORS["text"])),
        legend=dict(bgcolor="rgba(17,19,42,0.85)", bordercolor=COLORS["border"]),
    )
)

pio.templates["karanverse"] = PLOTLY_TEMPLATE
pio.templates.default = "karanverse"


def inject_global_css() -> str:
    c = COLORS
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', system-ui, sans-serif;
    }}

    .stApp {{
        background: radial-gradient(1200px 600px at 10% -10%, rgba(124,92,255,0.12), transparent 55%),
                    radial-gradient(900px 500px at 100% 0%, rgba(56,189,248,0.08), transparent 50%),
                    {c['bg']};
    }}

    .block-container {{
        padding-top: 3.75rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }}

    .kv-topbar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 52px;
        z-index: 999991;
        display: flex;
        align-items: center;
        padding: 0 1.25rem;
        background: linear-gradient(90deg, {c['surface_2']} 0%, {c['surface']} 70%);
        border-bottom: 1px solid {c['border']};
        box-shadow: 0 4px 24px rgba(8, 9, 26, 0.45);
    }}

    .kv-topbar-left {{
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        min-width: 0;
    }}

    .kv-topbar-brand {{
        font-size: 1.25rem;
        font-weight: 800;
        white-space: nowrap;
    }}

    .kv-topbar-tagline {{
        color: {c['muted']};
        font-size: 0.78rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    @media (max-width: 720px) {{
        .kv-topbar-tagline {{ display: none; }}
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {c['surface_2']} 0%, {c['bg']} 100%);
        border-right: 1px solid {c['border']};
        padding-top: 52px;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0.75rem;
    }}

    [data-testid="stSidebar"] .stRadio label {{
        background: transparent;
        border-radius: 10px;
        padding: 0.35rem 0.5rem;
        transition: background 0.15s ease;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: {c['text']} !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {c['muted']} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid {c['border']};
        border-radius: 12px;
        background: {c['surface']};
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {c['surface']};
        border-radius: 12px;
        padding: 6px;
        border: 1px solid {c['border']};
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }}

    .stDownloadButton button, .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid {c['border']};
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}

    .stDownloadButton button:hover, .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 22px rgba(124, 92, 255, 0.22);
    }}

    .hero-banner {{
        background: linear-gradient(135deg, {c['surface_2']} 0%, {c['surface']} 52%, rgba(124,92,255,0.1) 100%);
        border: 1px solid {c['border']};
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }}

    .hero-banner::after {{
        content: '';
        position: absolute;
        top: -40%;
        right: -10%;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(56,189,248,0.14) 0%, transparent 70%);
        pointer-events: none;
    }}

    .hero-title {{
        font-size: 1.65rem;
        font-weight: 700;
        color: {c['text']};
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }}

    .hero-sub {{
        color: {c['muted']};
        font-size: 0.95rem;
        margin: 0;
        max-width: 52rem;
        line-height: 1.5;
    }}

    .hero-badge {{
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: rgba(124,92,255,0.14);
        color: {c['accent_2']};
        border: 1px solid rgba(124,92,255,0.4);
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.25rem;
    }}

    @media (max-width: 900px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    .kpi-card {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 96px;
        position: relative;
    }}

    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
        background: linear-gradient(90deg, {c['accent']}, {c['accent_2']});
        opacity: 0.9;
    }}

    .kpi-label {{
        color: {c['muted']};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}

    .kpi-value {{
        color: {c['text']};
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 0.25rem 0;
    }}

    .kpi-hint {{
        color: {c['muted']};
        font-size: 0.78rem;
    }}

    .panel {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }}

    .panel-title {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {c['text']};
        margin: 0 0 0.75rem 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .sidebar-brand {{
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid {c['border']};
        margin-bottom: 1rem;
    }}

    .brand-karanverse {{
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        background: linear-gradient(90deg, {c['accent']} 0%, {c['accent_2']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.35rem;
    }}

    .sidebar-brand-title {{
        font-size: 0.92rem;
        font-weight: 600;
        color: {c['text']};
        line-height: 1.35;
    }}

    .sidebar-brand-sub {{
        font-size: 0.75rem;
        color: {c['muted']};
        margin-top: 0.35rem;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid {c['border']};
        background: {c['surface_2']};
        color: {c['muted']};
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {c['accent']};
        box-shadow: 0 0 8px {c['accent']};
    }}

    .empty-state {{
        text-align: center;
        padding: 2.5rem 1.5rem;
        border: 1px dashed {c['border']};
        border-radius: 14px;
        color: {c['muted']};
        background: rgba(17,19,42,0.55);
    }}

    .empty-state-title {{
        color: {c['text']};
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.35rem;
    }}

    code, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    """
