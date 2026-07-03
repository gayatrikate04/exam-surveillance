"""
Shared image/video processing helpers (no Streamlit).
Used by streamlit_app.py and api_server.py.
"""
import gc
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from detector import LateralMotionTracker, process_frame, process_frame_multi

_log = logging.getLogger(__name__)

Status = Tuple[str, float]


def process_image(
    image_bgr: np.ndarray,
    multi_person: bool = False,
    yolo_conf: float = 0.18,
    yolo_imgsz: int = 1280,
    max_people: int = 200,
    yolo_iou: float = 0.55,
    kpt_floor: float = 0.12,
    object_detect_conf: float = 0.20,
    object_detect_hires: bool = True,
    object_tta: bool = False,
    exam_overlay: str = "full",
) -> Tuple[np.ndarray, List[Status], Dict[str, Any]]:
    aug = True if object_tta else None
    match_pose = not object_detect_hires
    if multi_person:
        return process_frame_multi(
            image_bgr,
            None,
            max_persons=max_people,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
            iou=yolo_iou,
            kpt_conf_floor=kpt_floor,
            object_detect_conf=object_detect_conf,
            object_detect_match_pose_imgsz=match_pose,
            object_detect_augment=aug,
            exam_overlay=exam_overlay,
        )
    return process_frame(
        image_bgr,
        None,
        conf=max(0.2, yolo_conf),
        imgsz=min(yolo_imgsz, 960),
        object_detect_conf=object_detect_conf,
        object_detect_match_pose_imgsz=match_pose,
        object_detect_augment=aug,
        exam_overlay=exam_overlay,
    )


def process_video_sampled(
    video_path: str,
    max_frames: int = 20,
    multi_person: bool = False,
    progress_callback=None,
    yolo_conf: float = 0.18,
    yolo_imgsz: int = 960,
    max_people: int = 200,
    yolo_iou: float = 0.55,
    kpt_floor: float = 0.12,
    object_detect_conf: float = 0.20,
    object_detect_hires: bool = True,
    object_tta: bool = False,
    motion_tracker: Optional[LateralMotionTracker] = None,
    exam_overlay: str = "full",
) -> List[Tuple[np.ndarray, List[Status], Dict[str, Any]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        step = 1
    else:
        step = max(total_frames // max_frames, 1)

    out: List[Tuple[np.ndarray, List[Status], Dict[str, Any]]] = []
    tr = motion_tracker if motion_tracker is not None else LateralMotionTracker()

    consecutive_read_failures = 0
    max_consecutive_read_failures = 12

    try:
        idx = 0
        while True:
            try:
                ret, frame = cap.read()
            except cv2.error as exc:
                _log.warning("VideoCapture.read raised cv2.error (%s); aborting read loop", exc)
                break
            if not ret or frame is None or frame.size == 0:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_consecutive_read_failures:
                    break
                idx += 1
                continue
            consecutive_read_failures = 0

            if idx % step == 0:
                if progress_callback:
                    progress_callback(len(out), max_frames, f"Processing frame {len(out) + 1}/{max_frames}...")
                aug = True if object_tta else None
                match_pose = not object_detect_hires
                try:
                    if multi_person:
                        annotated, statuses, summ = process_frame_multi(
                            frame,
                            None,
                            max_persons=max_people,
                            conf=yolo_conf,
                            imgsz=yolo_imgsz,
                            iou=yolo_iou,
                            kpt_conf_floor=kpt_floor,
                            object_detect_conf=object_detect_conf,
                            object_detect_match_pose_imgsz=match_pose,
                            object_detect_augment=aug,
                            motion_tracker=tr,
                            exam_overlay=exam_overlay,
                        )
                    else:
                        annotated, statuses, summ = process_frame(
                            frame,
                            None,
                            conf=max(0.2, yolo_conf),
                            imgsz=min(yolo_imgsz, 960),
                            object_detect_conf=object_detect_conf,
                            object_detect_match_pose_imgsz=match_pose,
                            object_detect_augment=aug,
                            motion_tracker=tr,
                            exam_overlay=exam_overlay,
                        )
                    out.append((annotated, statuses, summ))
                except Exception as exc:
                    # One bad frame should not kill the whole video — skip and keep going.
                    _log.warning("Frame %d failed (%s); skipping", idx, exc)

            idx += 1
            if len(out) >= max_frames:
                break
    finally:
        try:
            cap.release()
        except Exception:
            pass

    if progress_callback:
        progress_callback(max_frames, max_frames, "Done.")
    return out


def iter_video_frames_sampled(
    video_path: str,
    max_frames: int = 20,
    multi_person: bool = False,
    yolo_conf: float = 0.18,
    yolo_imgsz: int = 960,
    max_people: int = 200,
    yolo_iou: float = 0.55,
    kpt_floor: float = 0.12,
    object_detect_conf: float = 0.20,
    object_detect_hires: bool = True,
    object_tta: bool = False,
    motion_tracker: Optional[LateralMotionTracker] = None,
    exam_overlay: str = "full",
    gc_every: int = 4,
) -> "Iterator[Tuple[np.ndarray, List[Status], Dict[str, Any]]]":
    """
    Memory-bounded streaming variant of process_video_sampled.

    Yields one (annotated, statuses, summary) at a time. The caller is expected
    to encode and release the annotated numpy buffer between yields so peak RAM
    stays at ~one frame instead of N. Used by the streaming /api/process/video
    endpoint to prevent uvicorn from being OOM-killed on long videos.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = 1 if total_frames <= 0 else max(total_frames // max_frames, 1)

    tr = motion_tracker if motion_tracker is not None else LateralMotionTracker()
    aug = True if object_tta else None
    match_pose = not object_detect_hires

    consecutive_read_failures = 0
    max_consecutive_read_failures = 12
    emitted = 0

    try:
        idx = 0
        while emitted < max_frames:
            try:
                ret, frame = cap.read()
            except cv2.error as exc:
                _log.warning("VideoCapture.read raised cv2.error (%s); aborting read loop", exc)
                break
            if not ret or frame is None or frame.size == 0:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_consecutive_read_failures:
                    break
                idx += 1
                continue
            consecutive_read_failures = 0

            if idx % step == 0:
                try:
                    if multi_person:
                        annotated, statuses, summ = process_frame_multi(
                            frame, None,
                            max_persons=max_people, conf=yolo_conf, imgsz=yolo_imgsz,
                            iou=yolo_iou, kpt_conf_floor=kpt_floor,
                            object_detect_conf=object_detect_conf,
                            object_detect_match_pose_imgsz=match_pose,
                            object_detect_augment=aug, motion_tracker=tr,
                            exam_overlay=exam_overlay,
                        )
                    else:
                        annotated, statuses, summ = process_frame(
                            frame, None,
                            conf=max(0.2, yolo_conf), imgsz=min(yolo_imgsz, 960),
                            object_detect_conf=object_detect_conf,
                            object_detect_match_pose_imgsz=match_pose,
                            object_detect_augment=aug, motion_tracker=tr,
                            exam_overlay=exam_overlay,
                        )
                    yield annotated, statuses, summ
                    # Caller has consumed it; drop our refs and nudge GC so the
                    # next frame does not pile on top of the last one.
                    del annotated, statuses, summ
                    emitted += 1
                    if gc_every > 0 and emitted % gc_every == 0:
                        gc.collect()
                except Exception as exc:
                    _log.warning("Frame %d failed (%s); skipping", idx, exc)
            idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass


def list_available_cameras(max_index: int = 10) -> List[int]:
    found: List[int] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                found.append(i)
        finally:
            cap.release()
    return found


def annotated_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return b""
    return buf.tobytes()
