"""karanverse Wildlife Detection & Monitoring System — Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ui.components import apply_theme, model_status_pill, top_brand_bar
from src.ui.streamlit_app import (
    get_config,
    get_detector,
    render_analytics,
    render_batch,
    render_dashboard,
    render_image_detection,
    render_model_info,
    render_settings,
    render_video_tracking,
    _config_cache_key,
)


def main() -> None:
    cfg = load_config()
    st.set_page_config(
        page_title=cfg.app.title,
        page_icon=cfg.app.page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    runtime_cfg = get_config()
    top_brand_bar(runtime_cfg.app.brand_name, runtime_cfg.app.brand_tagline)

    detector = get_detector(_config_cache_key(runtime_cfg))
    info = detector.model_info

    with st.sidebar:
        st.caption("Navigation")
        model_status_pill(info.loaded, info.device if info.loaded else "—")
        if st.session_state.get("model_load_error"):
            st.error(st.session_state["model_load_error"])
        st.divider()
        st.caption("Quick tips")
        st.markdown(
            """
            - Use **Image** for single trap shots  
            - Use **Video** for motion + track IDs  
            - **Analytics** reads stored SQLite events  
            """
        )

    pages = {
        "Overview": [
            st.Page(render_dashboard, title="Dashboard", icon="📊", default=True),
        ],
        "Detection": [
            st.Page(render_image_detection, title="Image Detection", icon="🖼️"),
            st.Page(render_video_tracking, title="Video Tracking", icon="🎬"),
            st.Page(render_batch, title="Camera-Trap Batch", icon="📦"),
        ],
        "Insights": [
            st.Page(render_analytics, title="Analytics", icon="📈"),
        ],
        "System": [
            st.Page(render_model_info, title="Model Information", icon="🧠"),
            st.Page(render_settings, title="Settings", icon="⚙️"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
