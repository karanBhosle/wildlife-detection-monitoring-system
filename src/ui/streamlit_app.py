"""Streamlit UI pages and shared resources."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.report_generator import ReportGenerator
from src.analytics.species_analytics import (
    activity_by_date,
    activity_by_hour,
    compute_summary,
    filter_detections_df,
    per_input_stats,
    species_distribution,
)
from src.config import WildlifeConfig, load_config
from src.detection.detector import WildlifeDetector
from src.logging_config import setup_logging
from src.processing.batch_processor import BatchProcessor
from src.processing.image_processor import ImageProcessor
from src.processing.video_processor import VideoProcessor
from src.storage.repository import WildlifeRepository
from src.ui.components import (
    confidence_control,
    detection_table,
    empty_state,
    kpi_cards,
    page_hero,
    styled_bar_chart,
    styled_line_chart,
    styled_pie_chart,
)
from src.ui.theme import COLORS
from src.utils.file_validation import (
    sanitize_filename,
    validate_extension,
    validate_file_size,
    validate_image_readable,
    validate_video_readable,
)
from src.utils.visualization import bgr_to_rgb


@st.cache_resource
def get_config() -> WildlifeConfig:
    cfg = load_config()
    setup_logging(cfg)
    cfg.ensure_directories()
    return cfg


@st.cache_resource
def get_detector(_config_version: str) -> WildlifeDetector:
    cfg = get_config()
    detector = WildlifeDetector(cfg)
    try:
        detector.load()
    except Exception as exc:
        st.session_state["model_load_error"] = str(exc)
    return detector


@st.cache_resource
def get_repository(_db_path: str) -> WildlifeRepository:
    cfg = get_config()
    return WildlifeRepository(cfg.paths.database_path)


def _config_cache_key(cfg: WildlifeConfig) -> str:
    conf = st.session_state.get("ui_confidence", cfg.model.confidence_threshold)
    return f"{cfg.model.weights_path}|{cfg.model.device}|{cfg.model.image_size}|{conf}"


def render_dashboard() -> None:
    page_hero(
        "Operations dashboard",
        "Live view of detection events, species mix, and recent processing jobs from your SQLite store.",
        badge="Monitoring",
    )
    cfg = get_config()
    repo = get_repository(cfg.paths.database_path)
    df = repo.fetch_detections_df()
    jobs = repo.fetch_jobs_df()
    summary = compute_summary(df, jobs)

    kpi_cards(
        [
            {
                "label": "Detection events",
                "value": f"{summary.total_detection_events:,}",
                "hint": "Stored model outputs",
            },
            {
                "label": "Species detected",
                "value": str(summary.unique_species),
                "hint": "Unique class labels",
            },
            {
                "label": "Images processed",
                "value": str(summary.total_images),
                "hint": "Single + batch stills",
            },
            {
                "label": "Track IDs (video)",
                "value": str(summary.unique_track_ids),
                "hint": "Session-level IDs",
            },
        ]
    )

    st.caption(summary.metrics_note)

    if df.empty:
        empty_state(
            "No observations yet",
            "Run image, video, or batch detection to populate charts and activity tables.",
            "🌲",
        )
        return

    dist = species_distribution(df)
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            styled_bar_chart(dist, "species_name", "count", "Species distribution"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            styled_pie_chart(dist, "species_name", "count", "Share of detections"),
            use_container_width=True,
        )

    if not jobs.empty:
        st.markdown("#### Recent activity")
        recent = jobs.head(15).copy()
        if "status" in recent.columns:
            recent["status"] = recent["status"].astype(str).str.upper()
        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.TextColumn("Status", width="small"),
                "input_filename": st.column_config.TextColumn("File", width="large"),
                "processing_duration_ms": st.column_config.NumberColumn("Duration (ms)", format="%.0f"),
            },
        )


def render_image_detection() -> None:
    page_hero(
        "Image detection",
        "Upload a camera-trap still, tune confidence, and inspect bounding boxes with exportable metadata.",
        badge="YOLO11",
    )
    cfg = get_config()
    detector = get_detector(_config_cache_key(cfg))
    repo = get_repository(cfg.paths.database_path)
    processor = ImageProcessor(cfg, detector, repo)

    control, preview = st.columns([1, 2.2], gap="large")
    with control:
        st.markdown("##### Controls")
        conf = confidence_control("Confidence threshold", cfg.model.confidence_threshold, "img_conf")
        uploaded = st.file_uploader(
            "Wildlife image",
            type=[e.strip(".") for e in cfg.image.allowed_extensions],
            help=f"Max {cfg.image.max_upload_mb} MB",
        )
        if uploaded:
            if st.session_state.get("img_upload_name") != uploaded.name:
                st.session_state.pop("img_last_result", None)
            st.session_state["img_upload_name"] = uploaded.name
            st.session_state["img_upload_bytes"] = uploaded.getvalue()
        run = st.button("Run detection", type="primary", use_container_width=True)

    with preview:
        if "img_upload_bytes" not in st.session_state:
            empty_state("Awaiting image", "Choose a file and click **Run detection**.", "🖼️")
            return

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(st.session_state["img_upload_name"]).suffix,
        ) as tmp:
            tmp.write(st.session_state["img_upload_bytes"])
            tmp_path = Path(tmp.name)

        if not run and "img_last_result" not in st.session_state:
            st.image(bgr_to_rgb(cv2.imread(str(tmp_path))), caption="Preview", use_container_width=True)
            st.info("Adjust settings, then run detection.")
            return

        if not run:
            result = st.session_state.get("img_last_result")
        else:
            for v in (
                validate_extension(tmp_path, cfg.image.allowed_extensions),
                validate_file_size(tmp_path, cfg.image.max_upload_mb),
                validate_image_readable(tmp_path),
            ):
                if not v.ok:
                    st.error(v.message)
                    return
            with st.spinner("Running YOLO11 inference…"):
                result = processor.process(tmp_path, confidence=conf)
            st.session_state["img_last_result"] = result

        if result.error:
            st.error(result.error)
            return

        tab_vis, tab_data, tab_export = st.tabs(["Visual", "Detections", "Export"])
        frame = cv2.imread(str(tmp_path))
        with tab_vis:
            v1, v2 = st.columns(2)
            with v1:
                st.image(bgr_to_rgb(frame), caption="Original", use_container_width=True)
            with v2:
                if result.annotated_path and result.annotated_path.exists():
                    st.image(str(result.annotated_path), caption="Annotated", use_container_width=True)
        with tab_data:
            if result.detections:
                ddf = pd.DataFrame([d.to_dict() for d in result.detections])
                species_filter = st.multiselect("Filter species", sorted(ddf["display_name"].unique()))
                if species_filter:
                    ddf = ddf[ddf["display_name"].isin(species_filter)]
                detection_table(ddf)
                st.metric("Inference time (ms)", f"{result.inference_time_ms:.1f}")
            else:
                st.warning("No detections above threshold. Lower confidence or use wildlife-trained weights.")
        with tab_export:
            if result.detections:
                ddf = pd.DataFrame([d.to_dict() for d in result.detections])
                reporter = ReportGenerator(cfg.paths.exports_dir)
                csv_path = reporter.export_detections_csv(
                    ddf,
                    sanitize_filename(f"{st.session_state['img_upload_name']}_detections.csv"),
                )
                c1, c2 = st.columns(2)
                c1.download_button("Download CSV", csv_path.read_bytes(), file_name=csv_path.name, use_container_width=True)
                if result.annotated_path:
                    c2.download_button(
                        "Download annotated image",
                        result.annotated_path.read_bytes(),
                        file_name=result.annotated_path.name,
                        use_container_width=True,
                    )


def render_video_tracking() -> None:
    page_hero(
        "Video tracking",
        "Streamed frame processing with ByteTrack IDs. Track numbers reset per video session.",
        badge="ByteTrack",
    )
    cfg = get_config()
    detector = get_detector(_config_cache_key(cfg))
    repo = get_repository(cfg.paths.database_path)
    processor = VideoProcessor(cfg, detector, repo)

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("##### Pipeline")
        conf = confidence_control("Confidence threshold", cfg.model.confidence_threshold, "vid_conf")
        uploaded = st.file_uploader(
            "Video file",
            type=[e.strip(".") for e in cfg.video.allowed_extensions],
            key="video_upload",
        )
        st.caption(
            f"Sampling: every {cfg.tracking.frame_sample_interval} frame(s) · "
            f"Max {cfg.video.max_upload_mb} MB"
        )
        process = st.button("Process with tracking", type="primary", use_container_width=True)

    with right:
        st.markdown("##### Output")
        if not (uploaded and process):
            empty_state(
                "Ready for video",
                "Upload MP4/AVI/MOV and start processing. Progress appears here.",
                "🎬",
            )
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        for v in (
            validate_extension(tmp_path, cfg.video.allowed_extensions),
            validate_file_size(tmp_path, cfg.video.max_upload_mb),
            validate_video_readable(tmp_path),
        ):
            if not v.ok:
                st.error(v.message)
                return

        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        def on_progress(p: float, msg: str) -> None:
            progress.progress(p, text=msg)
            status.caption(msg)

        with st.spinner("Processing frames…"):
            result = processor.process(tmp_path, confidence=conf, progress_callback=on_progress)

        if result.error:
            st.error(result.error)
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Frames sampled", result.frames_processed)
        m2.metric("Detection events", result.total_detections)
        m3.metric("Unique track IDs", result.tracking_summary.get("unique_track_ids", 0))

        if result.output_path and result.output_path.exists():
            st.video(str(result.output_path))
            st.download_button(
                "Download annotated video",
                result.output_path.read_bytes(),
                file_name=result.output_path.name,
                use_container_width=True,
            )


def render_batch() -> None:
    page_hero(
        "Camera-trap batch",
        "Process uploads or a folder of stills with per-file status, species counts, and consolidated exports.",
        badge="Batch",
    )
    cfg = get_config()
    detector = get_detector(_config_cache_key(cfg))
    repo = get_repository(cfg.paths.database_path)
    batch = BatchProcessor(cfg, detector, repo)
    conf = confidence_control("Confidence threshold", cfg.model.confidence_threshold, "batch_conf")

    tab_up, tab_dir = st.tabs(["Upload batch", "Local folder"])

    with tab_up:
        uploads = st.file_uploader(
            "Images",
            accept_multiple_files=True,
            type=[e.strip(".") for e in cfg.image.allowed_extensions],
        )
        if st.button("Process uploads", type="primary") and uploads:
            paths: list[Path] = []
            for up in uploads:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
                    tmp.write(up.getvalue())
                    paths.append(Path(tmp.name))
            _run_batch_ui(batch, repo, cfg, paths, conf)

    with tab_dir:
        st.caption(f"Reads images from `{cfg.batch.input_dir}`")
        if st.button("Process directory", type="primary"):
            _run_batch_ui(batch, repo, cfg, None, conf, use_directory=True)


def _run_batch_ui(
    batch: BatchProcessor,
    repo: WildlifeRepository,
    cfg: WildlifeConfig,
    paths: list[Path] | None,
    conf: float,
    use_directory: bool = False,
) -> None:
    bar = st.progress(0.0, text="Queued…")
    label = st.empty()

    def prog(done: int, total: int, name: str) -> None:
        if total:
            bar.progress(min(1.0, done / max(total, 1)), text=f"{done}/{total}: {name}")
        label.caption(name)

    if use_directory:
        result = batch.process_directory(confidence=conf, progress_callback=prog)
    else:
        result = batch.process_paths(paths or [], confidence=conf, progress_callback=prog)

    kpi_cards(
        [
            {"label": "Total detections", "value": str(result.total_detections), "hint": "This run"},
            {"label": "Files", "value": str(len(result.file_statuses)), "hint": "Processed"},
            {"label": "Duration", "value": f"{result.duration_ms:.0f} ms", "hint": "Wall time"},
            {"label": "Species", "value": str(len(result.species_counts)), "hint": "In batch"},
        ]
    )

    st.dataframe(
        pd.DataFrame([s.__dict__ for s in result.file_statuses]),
        use_container_width=True,
        hide_index=True,
    )

    if result.species_counts:
        dist = pd.DataFrame(
            [{"species_name": k, "count": v} for k, v in result.species_counts.items()]
        )
        st.plotly_chart(styled_bar_chart(dist, "species_name", "count", "Batch species counts"), use_container_width=True)

    df = repo.fetch_detections_df()
    if not df.empty:
        reporter = ReportGenerator(cfg.paths.exports_dir)
        csv_p = reporter.export_detections_csv(df, "batch_all_detections.csv")
        json_p = reporter.export_detections_json(
            repo.fetch_all_detections_as_records(),
            "batch_all_detections.json",
        )
        c1, c2 = st.columns(2)
        c1.download_button("Consolidated CSV", csv_p.read_bytes(), csv_p.name, use_container_width=True)
        c2.download_button("Consolidated JSON", json_p.read_bytes(), json_p.name, use_container_width=True)


def render_analytics() -> None:
    page_hero(
        "Wildlife analytics",
        "Explore stored detections with filters. Time charts require valid observation timestamps.",
        badge="Insights",
    )
    cfg = get_config()
    repo = get_repository(cfg.paths.database_path)
    df = repo.fetch_detections_df()
    if df.empty:
        empty_state("No analytics data", "Complete at least one detection job first.", "📈")
        return

    filt_col, chart_col = st.columns([1, 3], gap="large")
    with filt_col:
        st.markdown("##### Filters")
        species = st.multiselect("Species", sorted(df["species_name"].dropna().unique()))
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
        source = st.selectbox("Input type", ["All", "image", "batch_image", "video"])
    input_type = None if source == "All" else source
    filtered = filter_detections_df(df, species=species or None, min_confidence=min_conf, input_type=input_type)

    with chart_col:
        summary = compute_summary(filtered)
        kpi_cards(
            [
                {"label": "Events", "value": str(summary.total_detection_events), "hint": "Filtered"},
                {"label": "Species", "value": str(summary.unique_species), "hint": "Unique"},
                {"label": "Avg conf.", "value": f"{summary.avg_confidence:.2f}", "hint": "Mean score"},
                {"label": "Avg infer.", "value": f"{summary.avg_inference_ms:.0f} ms", "hint": "Per event"},
            ]
        )

        c1, c2 = st.columns(2)
        dist = species_distribution(filtered)
        with c1:
            st.plotly_chart(styled_bar_chart(dist, "species_name", "count", "Species"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(filtered, x="confidence", nbins=20, title="Confidence"), use_container_width=True)

        hours = activity_by_hour(filtered)
        if hours.empty:
            st.warning("Observation timestamps unavailable — hour-of-day chart hidden.")
        else:
            st.plotly_chart(
                styled_bar_chart(hours, "hour", "detections", "Activity by hour"),
                use_container_width=True,
            )

        dates = activity_by_date(filtered)
        if dates.empty:
            st.warning("No valid observation dates for trend line.")
        else:
            st.plotly_chart(
                styled_line_chart(dates, "date", "detections", "Detections over time"),
                use_container_width=True,
            )

        st.markdown("##### Per input file")
        st.dataframe(per_input_stats(filtered), use_container_width=True, hide_index=True)


def render_model_info() -> None:
    page_hero(
        "Model & classes",
        "Active checkpoint, device, and supported labels. Use custom weights for reserve-specific species.",
        badge="Engine",
    )
    cfg = get_config()
    detector = get_detector(_config_cache_key(cfg))
    info = detector.model_info
    repo = get_repository(cfg.paths.database_path)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", len(info.class_names))
    c2.metric("Device", info.device)
    c3.metric("Loaded", "Yes" if info.loaded else "No")
    c4.metric("Confidence", cfg.model.confidence_threshold)

    st.code(str(info.weights_path), language=None)

    if info.unsupported_filter_warnings:
        for w in info.unsupported_filter_warnings:
            st.warning(w)

    st.info(
        "Default COCO checkpoints detect broad categories (bear, elephant, bird…) "
        "—not tiger, leopard, or wild boar unless you fine-tune."
    )

    if info.class_names:
        class_df = pd.DataFrame([{"class_id": k, "class_name": v} for k, v in sorted(info.class_names.items())])
        st.dataframe(class_df, use_container_width=True, hide_index=True, height=360)

    eval_run = repo.latest_model_run("evaluation")
    train_run = repo.latest_model_run("training")
    if eval_run or train_run:
        t1, t2 = st.tabs(["Evaluation", "Training"])
        with t1:
            if eval_run:
                st.json(eval_run.get("metrics", {}))
            else:
                st.caption("No evaluation runs logged.")
        with t2:
            if train_run:
                st.json(train_run)
            else:
                st.caption("No training runs logged.")


def render_settings() -> None:
    page_hero(
        "Session settings",
        "Overrides apply to this browser session. Persist changes in configs/config.yaml.",
        badge="Config",
    )
    cfg = get_config()
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["override_conf"] = st.number_input(
            "Detection confidence",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.get("ui_confidence", cfg.model.confidence_threshold)),
            step=0.05,
        )
        st.session_state["ui_confidence"] = float(st.session_state["override_conf"])
    with c2:
        st.session_state["override_imgsz"] = st.selectbox(
            "Image size",
            [320, 416, 640, 1280],
            index=[320, 416, 640, 1280].index(cfg.model.image_size)
            if cfg.model.image_size in [320, 416, 640, 1280]
            else 2,
        )

    device = st.selectbox(
        "Device",
        ["auto", "cpu", "cuda", "mps"],
        index=["auto", "cpu", "cuda", "mps"].index(cfg.model.device),
    )

    if st.button("Apply & reload model cache", type="primary"):
        cfg.model.confidence_threshold = float(st.session_state["override_conf"])
        cfg.model.image_size = int(st.session_state["override_imgsz"])
        cfg.model.device = device  # type: ignore[assignment]
        get_detector.cache_clear()
        st.cache_resource.clear()
        st.success("Settings applied. Model cache cleared.")
        st.balloons()

    st.markdown(
        f"""
        <div style="margin-top:1rem;padding:1rem;border-radius:12px;border:1px solid {COLORS['border']};
        background:{COLORS['surface']};color:{COLORS['muted']};font-size:0.9rem;">
        <strong style="color:{COLORS['text']};">Paths</strong><br/>
        DB: <code>{cfg.paths.database_path}</code><br/>
        Exports: <code>{cfg.paths.exports_dir}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
