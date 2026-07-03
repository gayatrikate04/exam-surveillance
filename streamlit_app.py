import tempfile
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from app_pipeline import (
    annotated_png_bytes as _annotated_png_bytes,
    list_available_cameras,
    process_image,
    process_video_sampled,
)
from detector import LateralMotionTracker, process_frame, process_frame_multi


Status = Tuple[str, float]


st.set_page_config(page_title="Exam Activity Monitor", layout="wide")
st.title("Exam activity monitor")
with st.expander("About this app", expanded=False):
    st.markdown(
        "**YOLOv8-Pose** draws each student; **Suspicious** posture on **video/live** uses repeated "
        "**left–right head motion** vs shoulders (scanning). **Photos** use neutral pose except **phones** "
        "near a student. **Objects**: phone, book, laptop, etc. Prototype—tune in **Advanced** and verify "
        "before real exam use."
    )


def _render_activity_dashboard(summary: Dict[str, Any], *, compact: bool = False) -> None:
    if not summary or summary.get("people", 0) == 0 and not summary.get("objects"):
        return
    if compact:
        risk = str(summary.get("overall_risk", "low")).upper()
        st.caption(
            f"**{summary.get('people', 0)}** people · **{summary.get('normal', 0)}** ok · "
            f"**{summary.get('suspicious', 0)}** flagged · risk **{risk}**"
        )
        if int(summary.get("back_turned", 0)) > 0:
            st.warning(f"Back-facing detected: **{int(summary.get('back_turned', 0))}**")
        if summary.get("phone_near_person", 0) > 0:
            st.warning(
                f"**{summary['phone_near_person']}** student(s) — **phone** detected nearby."
            )
        return
    st.subheader("Overall activity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People (pose)", summary.get("people", 0))
    c2.metric("Normal", summary.get("normal", 0))
    c3.metric("Suspicious", summary.get("suspicious", 0))
    risk = summary.get("overall_risk", "low")
    c4.metric("Risk level", risk.upper())
    if int(summary.get("back_turned", 0)) > 0:
        st.warning(f"Back-facing detected for **{int(summary.get('back_turned', 0))}** person(s).")
    if summary.get("phone_near_person", 0) > 0:
        st.warning(
            f"**{summary['phone_near_person']}** student(s) associated with a **detected phone** (overlap / hold)."
        )
    objs = summary.get("objects") or {}
    if objs:
        st.caption("Objects this frame: " + ", ".join(f"{k}: {v}" for k, v in sorted(objs.items())))


_LIVE_REFRESH = timedelta(seconds=0.22)
# OpenCV device indices to probe (0 .. inclusive).
_CAMERA_PROBE_MAX = 10
_LIVE_CAPTURE_W = 1280
_LIVE_CAPTURE_H = 720
_LIVE_DISPLAY_MIN_W = 1280


def _on_live_camera_pick_change() -> None:
    """Changing device while streaming would leave a stale capture; stop cleanly."""
    if st.session_state.get("live_on"):
        _stop_live_camera()


def _stop_live_camera() -> None:
    ss = st.session_state
    ss["live_on"] = False
    ss["_live_back_alert_on"] = False
    cap = ss.get("live_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        ss["live_cap"] = None
    ss["live_tracker"] = None


def _live_preview_fragment() -> None:
    """Runs on a timer while live mode is active (see st.fragment)."""
    ss = st.session_state
    if not ss.get("live_on") or ss.get("_live_mode_selected") != "Live (webcam)":
        return
    cap = ss.get("live_cap")
    tracker = ss.get("live_tracker")
    if cap is None or tracker is None or not cap.isOpened():
        st.warning("Camera not ready. Press Stop and Start again.")
        return
    ret, frame = cap.read()
    if not ret or frame is None:
        st.warning("No frame from camera.")
        return

    params = ss.get("live_params") or {}
    aug = True if params.get("object_tta") else None
    match_pose = not params.get("object_detect_hires", True)
    y_imgsz = min(int(params.get("yolo_imgsz", 960)), 960)
    multi_person = params.get("multi_person", True)
    overlay = str(params.get("exam_overlay", "full"))
    if multi_person:
        annotated, statuses, summary = process_frame_multi(
            frame,
            None,
            max_persons=int(params.get("max_people", 200)),
            conf=float(params.get("yolo_conf", 0.22)),
            imgsz=y_imgsz,
            iou=float(params.get("yolo_iou", 0.55)),
            kpt_conf_floor=float(params.get("kpt_floor", 0.12)),
            object_detect_conf=float(params.get("object_detect_conf", 0.2)),
            object_detect_match_pose_imgsz=match_pose,
            object_detect_augment=aug,
            motion_tracker=tracker,
            exam_overlay=overlay,
        )
    else:
        annotated, statuses, summary = process_frame(
            frame,
            None,
            conf=max(0.2, float(params.get("yolo_conf", 0.22))),
            imgsz=y_imgsz,
            object_detect_conf=float(params.get("object_detect_conf", 0.2)),
            object_detect_match_pose_imgsz=match_pose,
            object_detect_augment=aug,
            motion_tracker=tracker,
            exam_overlay=overlay,
        )

    compact = bool(params.get("minimal_ui"))
    _render_activity_dashboard(summary, compact=compact)
    back_count = int(summary.get("back_turned", 0))
    back_alert_on = back_count > 0
    prev_back_alert_on = bool(ss.get("_live_back_alert_on", False))
    if back_alert_on:
        st.error(f"ALERT: {back_count} person(s) appear to be looking back.")
        if not prev_back_alert_on:
            st.toast("Alert triggered: back-facing behavior detected.")
    ss["_live_back_alert_on"] = back_alert_on
    h, w = annotated.shape[:2]
    if w < _LIVE_DISPLAY_MIN_W:
        scale = _LIVE_DISPLAY_MIN_W / max(1, w)
        annotated = cv2.resize(
            annotated,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=None if compact else "Live preview (this frame)", use_container_width=True)
    if not compact and statuses:
        n = len(statuses)
        if n == 1:
            s0, c0 = statuses[0]
            st.caption(f"**Person 1**: {s0} — {int(c0 * 100)}%")
        else:
            st.caption(
                f"**Per person**: {summary.get('normal', 0)} Normal, "
                f"{summary.get('suspicious', 0)} Suspicious"
            )
            for i, (status_val, conf) in enumerate(statuses):
                st.caption(f"Person {i + 1}: {status_val} ({int(conf * 100)}%)")


st.sidebar.markdown("### Controls")
ui_layout = st.sidebar.radio(
    "Interface",
    ["Minimal", "Advanced"],
    horizontal=True,
    help="Minimal: large preview, simple sidebar. Advanced: all detection sliders.",
)
minimal_ui = ui_layout == "Minimal"
exam_overlay = "minimal" if minimal_ui else "full"

mode = st.sidebar.selectbox("Input", ["Image", "Video", "Live (webcam)"])
prev_mode = st.session_state.get("_prev_input_mode")
if prev_mode is not None and prev_mode != mode and prev_mode == "Live (webcam)":
    _stop_live_camera()
st.session_state["_prev_input_mode"] = mode
st.session_state["_live_mode_selected"] = mode

if minimal_ui:
    multi_person = True
    max_people = 200
    yolo_conf = 0.12
    yolo_imgsz = 1280
    yolo_iou = 0.55
    kpt_floor = 0.12
    object_detect_hires = True
    object_detect_conf = 0.18
    object_tta = False
else:
    multi_person = st.sidebar.checkbox(
        "Detect multiple people",
        value=True,
        help="Unchecked: top detection only. Checked: many students + summary.",
    )
    st.sidebar.markdown("**Classroom / background**")
    crowd = st.sidebar.checkbox("Boost distant detection", value=True)
    max_people = st.sidebar.slider("Max people", 10, 250, 200, 10)
    yolo_conf = st.sidebar.slider(
        "Min box confidence",
        0.05,
        0.45,
        0.12 if crowd else 0.22,
        0.01,
    )
    yolo_imgsz = st.sidebar.select_slider(
        "Inference size (px)",
        options=[640, 960, 1280],
        value=1280 if crowd else 960,
    )
    yolo_iou = st.sidebar.slider("NMS IoU", 0.35, 0.75, 0.55 if crowd else 0.6, 0.05)
    kpt_floor = st.sidebar.slider("Min keypoint confidence (draw)", 0.05, 0.35, 0.12 if crowd else 0.18, 0.01)
    st.sidebar.markdown("**Object detection (phones, books, …)**")
    object_detect_hires = st.sidebar.checkbox(
        "Higher-res object pass (recommended)",
        value=True,
        help="Runs the object model at up to 1920px (vs pose size) so small phones are found more reliably. Uncheck for speed.",
    )
    object_detect_conf = st.sidebar.slider(
        "Object min confidence",
        0.08,
        0.45,
        0.18 if crowd else 0.22,
        0.01,
        help="Lower catches more objects (more false alarms). Higher is stricter.",
    )
    object_tta = st.sidebar.checkbox(
        "Object test-time augmentation (slower)",
        value=False,
        help="Extra inference passes; can sharpen boxes on hard frames. ~2–4× slower.",
    )

if mode == "Live (webcam)":
    st.sidebar.markdown("**Camera**" if minimal_ui else "**Live camera**")
    if not minimal_ui:
        st.sidebar.caption(f"Probes indices **0–{_CAMERA_PROBE_MAX}**.")
    if st.sidebar.button("Refresh list", key="live_cam_refresh"):
        st.session_state["live_camera_list"] = list_available_cameras(_CAMERA_PROBE_MAX)
    if st.session_state.get("live_camera_list") is None:
        st.session_state["live_camera_list"] = list_available_cameras(_CAMERA_PROBE_MAX)
    cam_list = list(st.session_state["live_camera_list"])

    pick_key = "live_camera_pick"
    if cam_list:
        current = st.session_state.get(pick_key)
        if current is not None and current not in cam_list:
            del st.session_state[pick_key]
        live_cam_idx = st.sidebar.selectbox(
            "Device",
            options=cam_list,
            format_func=lambda idx: f"Camera {idx}",
            help="Video input for live mode.",
            key=pick_key,
            on_change=_on_live_camera_pick_change,
        )
        if not minimal_ui:
            st.sidebar.caption("Changing device stops the stream—start again.")
    else:
        st.sidebar.warning("No camera found. Connect one and refresh.")
        live_cam_idx = 0
else:
    live_cam_idx = 0

if minimal_ui:
    st.sidebar.caption("Use **Advanced** for model tuning.")
else:
    st.sidebar.caption(
        "Weights: `YOLO_DETECT_MODEL` (default yolov8s). "
        "`YOLO_DETECT_IMGSZ`, `YOLO_DETECT_AGNOSTIC_NMS=1` for dense desks. "
        "`YOLO_DETECT_OBJECTS=0` disables objects. "
        "`POSTURE_LEGACY_STATIC=1` = old single-frame posture rules. "
        "`LATERAL_SWING_THRESHOLD` / `LATERAL_MIN_FRAMES` / `LATERAL_MIN_EXCURSION` / `LATERAL_MIN_REVERSALS` "
        "tune head-scan vs sustained gaze. `YOLO_POSE_DEDUPE_IOU` merges duplicate pose boxes."
    )

if mode == "Image":
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_image is not None:
        summary: Dict[str, Any] = {}
        with st.status("Processing image...", expanded=True) as status:
            st.write("Loading image...")
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if bgr is None:
                status.update(label="Error: could not read image.", state="error")
                st.error("Could not read the uploaded image.")
            else:
                st.write("Running pose + object detection...")
                annotated, statuses, summary = process_image(
                    bgr,
                    multi_person=multi_person,
                    yolo_conf=yolo_conf,
                    yolo_imgsz=yolo_imgsz,
                    max_people=max_people,
                    yolo_iou=yolo_iou,
                    kpt_floor=kpt_floor,
                    object_detect_conf=object_detect_conf,
                    object_detect_hires=object_detect_hires,
                    object_tta=object_tta,
                    exam_overlay=exam_overlay,
                )
                st.write("Drawing results...")
                orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                track_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                status.update(label="Done!", state="complete")

        if bgr is not None:
            if not minimal_ui and summary.get("posture_mode") == "snapshot_neutral":
                st.caption(
                    "Single image: posture is **not** scored (everyone Normal) except **phone near student**. "
                    "Use **Video** or **Live** for head-scan detection."
                )
            _render_activity_dashboard(summary, compact=minimal_ui)
            if minimal_ui:
                st.image(track_rgb, caption="Result", use_container_width=True)
                with st.expander("Original image"):
                    st.image(orig_rgb, use_container_width=True)
                with st.expander("Download PNG"):
                    st.download_button(
                        label="Download annotated PNG",
                        data=_annotated_png_bytes(annotated),
                        file_name="exam_monitor_annotated.png",
                        mime="image/png",
                    )
            else:
                st.subheader("Annotated view")
                col_orig, col_track = st.columns(2)
                with col_orig:
                    st.image(orig_rgb, caption="Original", use_container_width=True)
                with col_track:
                    st.image(track_rgb, caption="Pose + posture + objects", use_container_width=True)
                st.download_button(
                    label="Download annotated image (full resolution PNG)",
                    data=_annotated_png_bytes(annotated),
                    file_name="exam_monitor_annotated.png",
                    mime="image/png",
                    help="Same resolution as upload.",
                )

            if not statuses:
                st.warning(
                    "No pose detected. Open **Advanced** to tune sensitivity."
                    if minimal_ui
                    else "No pose detected. Try confidence / image size in the sidebar."
                )
            elif not minimal_ui:
                n = len(statuses)
                if n == 1:
                    status_val, conf = statuses[0]
                    st.info(f"**Person 1**: {status_val} — {int(conf * 100)}%")
                else:
                    st.info(
                        f"**Per person**: {summary.get('normal', 0)} Normal, {summary.get('suspicious', 0)} Suspicious "
                        f"(see table below)."
                    )
                    for i, (status_val, conf) in enumerate(statuses):
                        st.caption(f"Person {i + 1}: {status_val} ({int(conf * 100)}%)")

elif mode == "Video":
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=False,
    )

    if uploaded_video is not None:
        with st.status("Preparing video...", expanded=True) as vid_status:
            st.write("Saving uploaded video...")
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_file.write(uploaded_video.read())
            tmp_file.flush()
            vid_status.update(label="Processing video...", state="running")

        progress_bar = st.progress(0, text="Processing frames...")
        status_text = st.empty()

        def update_progress(current: int, total: int, message: str):
            if total > 0:
                progress_bar.progress(min(current / total, 1.0), text=message)
            status_text.caption(message)

        frames_data = process_video_sampled(
            tmp_file.name,
            max_frames=20,
            multi_person=multi_person,
            progress_callback=update_progress,
            yolo_conf=yolo_conf,
            yolo_imgsz=min(yolo_imgsz, 960),
            max_people=max_people,
            yolo_iou=yolo_iou,
            kpt_floor=kpt_floor,
            object_detect_conf=object_detect_conf,
            object_detect_hires=object_detect_hires,
            object_tta=object_tta,
            exam_overlay=exam_overlay,
        )

        progress_bar.empty()
        status_text.empty()
        if frames_data:
            st.success(f"Processed {len(frames_data)} frames.")

        if not frames_data:
            st.error("Could not process the uploaded video.")
        else:
            last_summary = frames_data[-1][2] if frames_data[-1][2] else {}
            _render_activity_dashboard(last_summary, compact=minimal_ui)
            st.subheader("Frames") if minimal_ui else st.subheader("Sampled frames")
            cols = st.columns(2)
            for i, (frame_bgr, statuses, _) in enumerate(frames_data):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                col = cols[i % 2]
                caption = f"Frame {i}"
                if statuses:
                    norm = sum(1 for s, _ in statuses if s == "Normal")
                    susp = len(statuses) - norm
                    caption += f" — N:{norm} S:{susp}"
                col.image(rgb, caption=caption, use_container_width=True)

elif mode == "Live (webcam)":
    st.session_state["live_params"] = {
        "multi_person": multi_person,
        "yolo_conf": yolo_conf,
        "yolo_imgsz": yolo_imgsz,
        "max_people": max_people,
        "yolo_iou": yolo_iou,
        "kpt_floor": kpt_floor,
        "object_detect_conf": object_detect_conf,
        "object_detect_hires": object_detect_hires,
        "object_tta": object_tta,
        "exam_overlay": exam_overlay,
        "minimal_ui": minimal_ui,
    }
    if minimal_ui:
        st.caption("Live preview · stop when done to free the camera")
    else:
        st.subheader("Live camera (real-time)")
        st.write(
            "Same pipeline as **uploaded video**: pose + objects each frame, **lateral head-motion** over time. "
            "Inference capped at **960px** here. **Stop** releases the camera."
        )
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("Start live", type="primary", disabled=st.session_state.get("live_on", False)):
            _stop_live_camera()
            cap = cv2.VideoCapture(live_cam_idx)
            if not cap.isOpened():
                st.error(f"Could not open camera index {live_cam_idx}.")
            else:
                # Ask the webcam for HD so Streamlit preview can use more page width.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, _LIVE_CAPTURE_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _LIVE_CAPTURE_H)
                st.session_state["live_cap"] = cap
                st.session_state["live_tracker"] = LateralMotionTracker()
                st.session_state["live_on"] = True
                st.rerun()
    with bc2:
        if st.button("Stop", disabled=not st.session_state.get("live_on", False)):
            _stop_live_camera()
            st.rerun()

    fragment = getattr(st, "fragment", None)
    if fragment is None:
        st.warning(
            "Live preview requires **Streamlit ≥ 1.33** (`pip install -U 'streamlit>=1.33'`). "
            "Alternatively run **`python main.py`** → **1** for OpenCV fullscreen webcam."
        )
    elif st.session_state.get("live_on"):
        fragment(run_every=_LIVE_REFRESH)(_live_preview_fragment)()
    else:
        st.info("Press **Start live** below." if minimal_ui else "Press **Start live** to open the camera stream.")
