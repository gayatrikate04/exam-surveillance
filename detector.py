from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from weapon_detection import detect_and_draw_weapons

import cv2
import numpy as np

from utils import calculate_angle, draw_status_text

try:
    from ultralytics import YOLO
except ImportError as e:
    YOLO = None  # type: ignore

_log = logging.getLogger(__name__)
# Ultralytics models are not thread-safe; serialize predict calls so concurrent
# /api/live/frame or /api/process/video requests do not race on shared state.
_yolo_lock = threading.Lock()


Status = Tuple[str, float]
ActivitySummary = Dict[str, Any]

# COCO class names (ultralytics yolov8n.pt) used for exam object monitoring
EXAM_OBJECT_NAMES = frozenset(
    {"cell phone", "book", "laptop", "mouse", "keyboard", "tv", "remote"}
)
PHONE_NAMES = frozenset({"cell phone"})

# COCO 17 keypoints (YOLOv8 pose): 0=nose, 5=L_shoulder, 6=R_shoulder, ...
YOLO_POSE_SKELETON: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (5, 11),
    (6, 8),
    (8, 10),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

_yolo_model: Optional["YOLO"] = None
_yolo_detect: Optional["YOLO"] = None
_DEFAULT_MODEL = "yolov8n-pose.pt"
# Larger detect model improves phone/book AP vs yolov8n (slower; override with YOLO_DETECT_MODEL).
_DEFAULT_DETECT_MODEL = "yolov8s.pt"

# Round to multiple of 32 (YOLO stride)
_IMGSZ_CAP = 1280


def _round_stride32(n: int) -> int:
    return max(32, (int(n) // 32) * 32)


def _pick_detect_imgsz(
    image_h: int,
    image_w: int,
    pose_imgsz: int,
    match_pose_only: bool = False,
    override: Optional[int] = None,
) -> int:
    """
    Object detection often needs >= pose imgsz (small phones in wide shots).
    `match_pose_only=True` disables the extra upscaling pass.
    """
    cap = int(os.environ.get("YOLO_DETECT_IMGSZ_CAP", "1920"))
    if override is not None and override > 0:
        return _round_stride32(min(int(override), cap))
    env = os.environ.get("YOLO_DETECT_IMGSZ")
    if env:
        return _round_stride32(min(int(env), cap))
    if match_pose_only:
        return _round_stride32(pose_imgsz)
    floor = int(os.environ.get("YOLO_DETECT_IMGSZ_FLOOR", "1280"))
    long_edge = max(image_h, image_w)
    from_native = _round_stride32(min(long_edge, cap))
    target = max(pose_imgsz, floor, from_native)
    return _round_stride32(min(target, cap))


def _object_detect_conf(pose_conf_thres: float, override: Optional[float] = None) -> float:
    if override is not None:
        return float(max(0.02, min(0.95, override)))
    env = os.environ.get("YOLO_DETECT_CONF")
    if env:
        return float(max(0.02, min(0.95, float(env))))
    # Slightly relaxed vs pose so small objects are not dropped when pose conf is tuned for crowds.
    return max(0.16, min(0.45, float(pose_conf_thres) * 0.82))


def _get_yolo() -> "YOLO":
    global _yolo_model
    if YOLO is None:
        raise ImportError("Install ultralytics: pip install ultralytics")
    if _yolo_model is None:
        weights = os.environ.get("YOLO_POSE_MODEL", _DEFAULT_MODEL)
        _yolo_model = YOLO(weights)
    return _yolo_model


def _get_yolo_detect() -> "YOLO":
    global _yolo_detect
    if YOLO is None:
        raise ImportError("Install ultralytics: pip install ultralytics")
    if _yolo_detect is None:
        weights = os.environ.get("YOLO_DETECT_MODEL", _DEFAULT_DETECT_MODEL)
        _yolo_detect = YOLO(weights)
    return _yolo_detect


def _exam_class_ids(model: "YOLO") -> List[int]:
    names = getattr(model, "names", None)
    if not isinstance(names, dict):
        return []
    return [int(i) for i, nm in names.items() if str(nm) in EXAM_OBJECT_NAMES]


def _dedupe_pose_detections(
    xyxy: np.ndarray,
    conf: Optional[np.ndarray],
    max_keep: int,
    iou_suppress: float,
) -> List[int]:
    """Greedy NMS on pose boxes: one person should not produce two overlapping tracks."""
    n = int(len(xyxy))
    if n <= 1:
        return list(range(min(n, max_keep)))
    n = min(n, max_keep)
    scores = conf[:n].astype(np.float64) if conf is not None else np.ones(n, dtype=np.float64)
    order = np.argsort(-scores)
    picked: List[int] = []
    for idx in order:
        i = int(idx)
        if i >= n:
            continue
        bi = xyxy[i]
        ok = True
        for j in picked:
            if _iou_xyxy(bi, xyxy[j]) >= iou_suppress:
                ok = False
                break
        if ok:
            picked.append(i)
    return sorted(picked)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / (union + 1e-6))


def _box_from_keypoints(xy: np.ndarray, kcf: np.ndarray, min_c: float) -> Optional[np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for i in range(17):
        if float(kcf[i]) >= min_c:
            xs.append(float(xy[i, 0]))
            ys.append(float(xy[i, 1]))
    if len(xs) < 3:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = max(8.0, (x2 - x1) * 0.08)
    pad_y = max(8.0, (y2 - y1) * 0.08)
    return np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)


def _empty_summary() -> ActivitySummary:
    return {
        "people": 0,
        "normal": 0,
        "suspicious": 0,
        "objects": {},
        "objects_list": [],
        "phone_near_person": 0,
        "back_turned": 0,
        "overall_risk": "low",
    }


def _pick_imgsz(image_h: int, image_w: int, override: Optional[int]) -> int:
    if override is not None and override > 0:
        return int(override)
    env = os.environ.get("YOLO_IMGSZ")
    if env:
        return max(32, int(env))
    m = max(image_h, image_w)
    if m <= 900:
        target = 640
    elif m <= 1400:
        target = 960
    else:
        target = 1280
    return min(target, _IMGSZ_CAP)


def _overlay_visual_scale(h: int, w: int) -> Dict[str, Any]:
    """
    Scale fonts and strokes with image size so HD exports keep labels readable.
    Baseline ~720px long edge -> scale 1.0; capped so 8K does not explode.
    """
    m = float(max(h, w))
    base = float(os.environ.get("HD_LABEL_BASE_PX", "640"))
    boost = float(os.environ.get("HD_LABEL_BOOST", "1.15"))
    s = min(4.0, max(1.0, (m / base) * boost))
    return {
        "s": s,
        "font_person": 0.55 * s,
        "thick_person": max(2, int(round(2 * s))),
        "font_object": 0.5 * s,
        "thick_object": max(2, int(round(2 * s))),
        "font_banner1": 0.58 * s,
        "font_banner2": 0.48 * s,
        "thick_banner": max(2, int(round(2 * s))),
        "line_pose": max(2, int(round(2 * s))),
        "kpt_radius": max(3, int(round(3 * s))),
        "box_margin": int(max(10, round(10 * s))),
        "box_outline": max(2, int(round(2 * s))),
    }


class _LM:
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float, y: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


def _landmarks_from_coco(
    xy: np.ndarray,
    kpt_conf: np.ndarray,
    img_w: int,
    img_h: int,
    min_kpt_conf: float = 0.15,
) -> List[Optional[_LM]]:
    """Map COCO nose/L_shoulder/R_shoulder into indices 0,11,12 for analyze_posture."""
    landmarks: List[Optional[_LM]] = [None] * 33
    mapping = [(0, 0), (11, 5), (12, 6)]
    for mp_i, coco_i in mapping:
        x_pix = float(xy[coco_i, 0])
        y_pix = float(xy[coco_i, 1])
        c = float(kpt_conf[coco_i])
        if c < min_kpt_conf:
            continue
        landmarks[mp_i] = _LM(x_pix / max(img_w, 1), y_pix / max(img_h, 1), c)
    return landmarks


def _head_lateral_offset_normalized(
    xy: np.ndarray,
    kpt_conf: np.ndarray,
    img_w: int,
    img_h: int,
    min_kpt_conf: float = 0.15,
) -> Optional[float]:
    """
    Horizontal head position relative to shoulder midline, normalized to image width.
    Negative = head left of torso center, positive = right (camera view).
    """
    nose_i, ls_i, rs_i = 0, 5, 6
    for idx in (nose_i, ls_i, rs_i):
        if float(kpt_conf[idx]) < min_kpt_conf:
            return None
    nx = float(xy[nose_i, 0]) / max(img_w, 1)
    ls_x = float(xy[ls_i, 0]) / max(img_w, 1)
    rs_x = float(xy[rs_i, 0]) / max(img_w, 1)
    mid_x = 0.5 * (ls_x + rs_x)
    return nx - mid_x


def _is_back_facing_candidate(
    kpt_conf: np.ndarray,
    min_kpt_conf: float = 0.15,
) -> bool:
    """
    Heuristic: back-facing likely when shoulders are visible but frontal face keypoints
    (nose/eyes) are consistently not visible.
    """
    shoulder_ok = float(kpt_conf[5]) >= min_kpt_conf and float(kpt_conf[6]) >= min_kpt_conf
    face_seen = sum(1 for i in (0, 1, 2) if float(kpt_conf[i]) >= min_kpt_conf)
    return bool(shoulder_ok and face_seen == 0)


def _derivative_reversals(values: Sequence[float], min_step: float = 0.0) -> int:
    """Count direction changes in head offset; ignore steps smaller than min_step (keypoint noise)."""
    if len(values) < 3:
        return 0
    d = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if min_step > 0:
        d = [di if abs(di) >= min_step else 0.0 for di in d]
    rev = 0
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            rev += 1
    return rev


def _median_smooth3(values: Sequence[float]) -> List[float]:
    """Light smoothing so live keypoint jitter does not fake left–right reversals."""
    v = [float(x) for x in values]
    n = len(v)
    if n < 3:
        return v
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        out.append(float(np.median(v[lo:hi])))
    return out


def _has_both_side_excursions(vals: Sequence[float], eps: float) -> bool:
    """
    True only if offset goes meaningfully to the left and right of torso midline (vs sustained one-way look).
    Units: normalized horizontal nose–mid-shoulder offset (~0.04–0.12 typical turn).
    """
    if not vals or eps <= 0:
        return False
    hi = max(vals)
    lo = min(vals)
    return hi >= eps and lo <= -eps


class LateralMotionTracker:
    """
    Marks posture as Suspicious only when a person's head (vs shoulders) moves
    left/right over recent frames—typical of scanning neighbors, not of holding
    a normal exam pose. Requires video; single images cannot use this signal.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        min_frames: Optional[int] = None,
        swing_threshold: Optional[float] = None,
        min_reversals: Optional[int] = None,
        iou_match_min: Optional[float] = None,
    ) -> None:
        wn = window if window is not None else int(os.environ.get("LATERAL_WINDOW", "28"))
        self.window = max(5, wn)
        self.min_frames = min_frames if min_frames is not None else int(os.environ.get("LATERAL_MIN_FRAMES", "12"))
        self.swing_threshold = (
            swing_threshold
            if swing_threshold is not None
            else float(os.environ.get("LATERAL_SWING_THRESHOLD", "0.042"))
        )
        self.min_reversals = (
            min_reversals if min_reversals is not None else int(os.environ.get("LATERAL_MIN_REVERSALS", "2"))
        )
        self.iou_match_min = (
            iou_match_min if iou_match_min is not None else float(os.environ.get("LATERAL_IOU_MATCH", "0.22"))
        )
        self.min_excursion = float(os.environ.get("LATERAL_MIN_EXCURSION", "0.018"))
        self.reversal_min_step = float(os.environ.get("LATERAL_REVERSAL_MIN_STEP", "0.005"))
        self.back_min_frames = int(os.environ.get("BACK_TURN_MIN_FRAMES", "6"))
        self.back_decay = int(os.environ.get("BACK_TURN_DECAY", "1"))
        self._next_id = 0
        self._tracks: Dict[int, Deque[float]] = {}
        self._back_streak: Dict[int, int] = {}
        self._prev_list: List[Tuple[int, np.ndarray]] = []

    def reset(self) -> None:
        self._next_id = 0
        self._tracks.clear()
        self._back_streak.clear()
        self._prev_list = []

    def _greedy_match(self, curr_boxes: List[np.ndarray]) -> List[int]:
        """Returns track id for each current index."""
        n = len(curr_boxes)
        if not self._prev_list:
            return [self._new_id() for _ in range(n)]

        pairs: List[Tuple[float, int, int]] = []
        for j in range(n):
            cb = curr_boxes[j]
            for tid, pb in self._prev_list:
                pairs.append((_iou_xyxy(cb, pb), j, tid))
        pairs.sort(key=lambda x: x[0], reverse=True)

        assigned_j: set = set()
        assigned_t: set = set()
        match_tid_for_j: Dict[int, int] = {}
        for iou_v, j, tid in pairs:
            if j in assigned_j or tid in assigned_t:
                continue
            if iou_v < self.iou_match_min:
                break
            assigned_j.add(j)
            assigned_t.add(tid)
            match_tid_for_j[j] = tid

        out: List[int] = []
        for j in range(n):
            if j in match_tid_for_j:
                out.append(match_tid_for_j[j])
            else:
                out.append(self._new_id())
        return out

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def step(
        self,
        person_boxes: List[np.ndarray],
        head_offsets: List[Optional[float]],
        back_candidates: Optional[List[bool]] = None,
    ) -> Tuple[List[Status], List[bool]]:
        """Update tracks and return statuses plus back-facing alert flags per person."""
        n = len(person_boxes)
        tids = self._greedy_match(person_boxes)
        current_tids: set = set()

        for j in range(n):
            tid = tids[j]
            current_tids.add(tid)
            if tid not in self._tracks:
                self._tracks[tid] = deque(maxlen=self.window)
            if tid not in self._back_streak:
                self._back_streak[tid] = 0
            off = head_offsets[j]
            if off is not None:
                self._tracks[tid].append(float(off))
            is_back = bool(back_candidates[j]) if back_candidates is not None and j < len(back_candidates) else False
            if is_back:
                self._back_streak[tid] = self._back_streak.get(tid, 0) + 1
            else:
                self._back_streak[tid] = max(0, self._back_streak.get(tid, 0) - self.back_decay)

        for tid in list(self._tracks.keys()):
            if tid not in current_tids:
                del self._tracks[tid]
        for tid in list(self._back_streak.keys()):
            if tid not in current_tids:
                del self._back_streak[tid]

        self._prev_list = [(tids[j], person_boxes[j]) for j in range(n)]

        out: List[Status] = []
        back_alerts: List[bool] = []
        for j in range(n):
            tid = tids[j]
            hist = self._tracks.get(tid, deque())
            back_streak = self._back_streak.get(tid, 0)
            back_alert = back_streak >= self.back_min_frames
            vals = _median_smooth3([float(x) for x in hist])
            if len(vals) < self.min_frames:
                if back_alert:
                    conf_b = min(0.95, 0.86 + 0.02 * max(0, back_streak - self.back_min_frames))
                    out.append(("Suspicious", conf_b))
                else:
                    out.append(("Normal", 0.55))
                back_alerts.append(back_alert)
                continue
            ptp = max(vals) - min(vals)
            rev = _derivative_reversals(vals, min_step=self.reversal_min_step)
            # Need real back-and-forth across midline (both-side excursions), not sustained gaze or jitter.
            scanning = (
                rev >= self.min_reversals
                and ptp >= self.swing_threshold
                and _has_both_side_excursions(vals, self.min_excursion)
            )
            if scanning:
                conf = min(0.95, 0.52 + 0.38 * min(1.5, ptp / max(self.swing_threshold, 1e-6)))
                if back_alert:
                    conf = max(conf, min(0.95, 0.86 + 0.02 * max(0, back_streak - self.back_min_frames)))
                out.append(("Suspicious", conf))
            else:
                if back_alert:
                    conf_b = min(0.95, 0.86 + 0.02 * max(0, back_streak - self.back_min_frames))
                    out.append(("Suspicious", conf_b))
                else:
                    out.append(("Normal", max(0.55, 0.78 - 0.25 * min(1.0, ptp / max(self.swing_threshold, 1e-6)))))
            back_alerts.append(back_alert)
        return out, back_alerts


def _safe_get_landmark(landmarks: Sequence, index: int):
    try:
        return landmarks[index]
    except IndexError:
        return None


def analyze_posture(landmarks: Sequence, image_shape: Tuple[int, int, int]) -> Status:
    nose = _safe_get_landmark(landmarks, 0)
    left_shoulder = _safe_get_landmark(landmarks, 11)
    right_shoulder = _safe_get_landmark(landmarks, 12)

    if not (nose and left_shoulder and right_shoulder):
        return "Normal", 0.5

    visibilities = [
        getattr(nose, "visibility", 1.0),
        getattr(left_shoulder, "visibility", 1.0),
        getattr(right_shoulder, "visibility", 1.0),
    ]
    if min(visibilities) < 0.4:
        return "Normal", 0.5

    nose_x = nose.x
    nose_y = nose.y
    ls_x, ls_y = left_shoulder.x, left_shoulder.y
    rs_x, rs_y = right_shoulder.x, right_shoulder.y

    shoulders_mid_x = (ls_x + rs_x) / 2.0
    head_offset = nose_x - shoulders_mid_x
    head_turn_threshold = 0.08
    head_turn_suspicious = abs(head_offset) > head_turn_threshold

    shoulders_mid_y = (ls_y + rs_y) / 2.0
    vertical_offset = nose_y - shoulders_mid_y
    lean_threshold = 0.08
    lean_suspicious = vertical_offset > lean_threshold

    angle_rad = np.arctan2(rs_y - ls_y, rs_x - ls_x + 1e-6)
    shoulder_tilt_deg = abs(np.degrees(angle_rad))
    shoulder_tilt_threshold = 18.0
    ref_point = (rs_x, ls_y)
    shoulder_angle = calculate_angle((ls_x, ls_y), (rs_x, rs_y), ref_point)
    if abs(shoulder_angle - shoulder_tilt_deg) > 10.0:
        shoulder_tilt_deg = min(shoulder_tilt_deg, shoulder_angle)
    shoulder_tilt_suspicious = shoulder_tilt_deg > shoulder_tilt_threshold

    rules_triggered = [
        head_turn_suspicious,
        lean_suspicious,
        shoulder_tilt_suspicious,
    ]
    num_triggered = sum(1 for r in rules_triggered if r)

    if num_triggered > 0:
        status = "Suspicious"
        confidence = min(1.0, 0.4 + 0.2 * num_triggered)
    else:
        status = "Normal"
        confidence = 0.75

    return status, confidence


def _draw_yolo_person(
    image: np.ndarray,
    xy: np.ndarray,
    kpt_conf: np.ndarray,
    status: str,
    confidence: float,
    min_kpt_conf: float = 0.15,
    vis: Optional[Dict[str, Any]] = None,
    compact_label: bool = False,
) -> None:
    h, w = image.shape[0], image.shape[1]
    if vis is None:
        vis = _overlay_visual_scale(h, w)
    line_w, kr, bm, bo = vis["line_pose"], vis["kpt_radius"], vis["box_margin"], vis["box_outline"]
    fsp, tsp = vis["font_person"], vis["thick_person"]

    pts: List[Optional[Tuple[int, int]]] = []
    for i in range(17):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        c = float(kpt_conf[i])
        if c < min_kpt_conf:
            pts.append(None)
        else:
            xi = int(np.clip(x, 0, image.shape[1] - 1))
            yi = int(np.clip(y, 0, image.shape[0] - 1))
            pts.append((xi, yi))

    line_col = (245, 66, 230)
    kpt_col = (245, 117, 66)
    for a, b in YOLO_POSE_SKELETON:
        if pts[a] and pts[b]:
            cv2.line(image, pts[a], pts[b], line_col, line_w, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(image, p, kr, kpt_col, -1, cv2.LINE_AA)

    valid = [p for p in pts if p]
    if not valid:
        return
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    x_min = max(0, min(xs) - bm)
    y_min = max(0, min(ys) - bm)
    x_max = min(image.shape[1] - 1, max(xs) + bm)
    y_max = min(image.shape[0] - 1, max(ys) + bm)
    is_suspicious = status == "Suspicious"
    box_color = (0, 0, 255) if is_suspicious else (0, 200, 0)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, bo, cv2.LINE_AA)
    ty = max(y_min - bm // 2, int(22 * vis["s"]))
    label = status if compact_label else f"{status} ({int(confidence * 100)}%)"
    draw_status_text(
        image,
        label,
        (x_min + max(4, bm // 4), ty),
        is_suspicious=is_suspicious,
        font_scale=fsp,
        thickness=tsp,
    )


def _run_yolo(
    frame_bgr: np.ndarray,
    max_det: int,
    imgsz: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
):
    model = _get_yolo()
    try:
        with _yolo_lock:
            return model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                verbose=False,
            )
    except Exception as exc:
        _log.warning("YOLO pose predict failed (%s); returning empty result", exc)
        return []


def _detect_exam_objects(
    frame_bgr: np.ndarray,
    imgsz: int,
    conf: float = 0.22,
    iou: Optional[float] = None,
    augment: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if os.environ.get("YOLO_DETECT_OBJECTS", "1") != "1":
        return [], {}
    try:
        dmodel = _get_yolo_detect()
    except Exception:
        return [], {}
    cids = _exam_class_ids(dmodel)
    if not cids:
        return [], {}
    iou_d = float(iou) if iou is not None else float(os.environ.get("YOLO_DETECT_IOU", "0.50"))
    if augment is None:
        use_aug = os.environ.get("YOLO_DETECT_AUGMENT", "0").strip().lower() in ("1", "true", "yes")
    else:
        use_aug = augment
    agnostic = os.environ.get("YOLO_DETECT_AGNOSTIC_NMS", "0").strip().lower() in ("1", "true", "yes")
    try:
        with _yolo_lock:
            det = dmodel.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=conf,
                iou=iou_d,
                max_det=160,
                classes=cids,
                augment=use_aug,
                agnostic_nms=agnostic,
                verbose=False,
            )
    except Exception as exc:
        _log.warning("YOLO detect predict failed (%s); skipping object detection", exc)
        return [], {}
    if not det or det[0].boxes is None or len(det[0].boxes) == 0:
        return [], {}
    boxes = det[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    cf = boxes.conf.cpu().numpy()
    names_map = det[0].names
    items: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for i in range(len(xyxy)):
        nm = str(names_map.get(int(cls[i]), cls[i]))
        items.append({"xyxy": xyxy[i], "name": nm, "conf": float(cf[i])})
        counts[nm] = counts.get(nm, 0) + 1
    return items, counts


def _draw_exam_objects(image: np.ndarray, items: List[Dict[str, Any]], vis: Optional[Dict[str, Any]] = None) -> None:
    h, w = image.shape[0], image.shape[1]
    if vis is None:
        vis = _overlay_visual_scale(h, w)
    fso, tso, bo = vis["font_object"], vis["thick_object"], vis["box_outline"]
    pad = max(8, int(8 * vis["s"]))
    for it in items:
        x1, y1, x2, y2 = [int(round(v)) for v in it["xyxy"]]
        nm = it["name"]
        cf = it["conf"]
        col = (200, 220, 100)
        if nm == "cell phone":
            col = (0, 140, 255)
        elif nm == "book":
            col = (80, 180, 255)
        elif nm == "laptop":
            col = (255, 100, 100)
        x2, y2 = min(x2, image.shape[1] - 1), min(y2, image.shape[0] - 1)
        x1, y1 = max(0, x1), max(0, y1)
        cv2.rectangle(image, (x1, y1), (x2, y2), col, bo, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{nm} {int(cf * 100)}%",
            (x1, max(y1 - pad, int(16 * vis["s"]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            fso,
            col,
            tso,
            cv2.LINE_AA,
        )


def _phone_associates_student(phone_xyxy: np.ndarray, person_xyxy: np.ndarray, iou_floor: float = 0.06) -> bool:
    if _iou_xyxy(phone_xyxy, person_xyxy) >= iou_floor:
        return True
    cx = 0.5 * (float(phone_xyxy[0]) + float(phone_xyxy[2]))
    cy = 0.5 * (float(phone_xyxy[1]) + float(phone_xyxy[3]))
    px1, py1, px2, py2 = map(float, person_xyxy)
    return px1 <= cx <= px2 and py1 <= cy <= py2


def _draw_exam_banner(
    image: np.ndarray,
    summary: ActivitySummary,
    infer_sz: int,
    conf_thres: float,
    y_line1: int = 5,
    vis: Optional[Dict[str, Any]] = None,
    style: str = "full",
) -> None:
    h, w = image.shape[:2]
    if vis is None:
        vis = _overlay_visual_scale(h, w)
    fb1, fb2, tb = vis["font_banner1"], vis["font_banner2"], vis["thick_banner"]
    gap = int(10 * vis["s"])
    n = summary["people"]
    risk = str(summary.get("overall_risk", "low"))
    top_col = (120, 255, 120) if risk == "low" else (120, 200, 255) if risk == "medium" else (120, 120, 255)

    if style == "minimal":
        phone = int(summary.get("phone_near_person", 0))
        back = int(summary.get("back_turned", 0))
        line = (
            f"{n} person{'s' if n != 1 else ''}  ·  {summary['normal']} ok  ·  {summary['suspicious']} alert ·  "
            f"{risk.upper()}"
        )
        if back:
            line += f"  ·  BACK×{back}"
        if phone:
            line += f"  ·  PHONE×{phone}"
        (tw1, th1), bl1 = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fb1, tb)
        th = th1 + bl1 + gap * 2
        box_w = min(w - 10, tw1 + gap * 3)
        cv2.rectangle(image, (5, y_line1), (5 + box_w, y_line1 + th), (20, 20, 20), -1, cv2.LINE_AA)
        cv2.putText(image, line, (gap, y_line1 + gap + th1), cv2.FONT_HERSHEY_SIMPLEX, fb1, top_col, tb, cv2.LINE_AA)
        return

    bn = f"Exam monitor | People: {n}  |  Normal: {summary['normal']}  Suspicious: {summary['suspicious']}"
    if summary.get("phone_near_person", 0) > 0:
        bn += f"  |  Phone near student: {summary['phone_near_person']}"
    (tw1, th1), bl1 = cv2.getTextSize(bn, cv2.FONT_HERSHEY_SIMPLEX, fb1, tb)
    obj_parts = []
    for k, v in sorted(summary.get("objects", {}).items()):
        if v:
            obj_parts.append(f"{k}:{v}")
    line2 = "Objects: " + (", ".join(obj_parts) if obj_parts else "none")
    det_px = summary.get("object_detect_imgsz", infer_sz)
    pm = summary.get("posture_mode", "")
    pm_short = {"lateral_motion": "lateralΔ", "snapshot_neutral": "photo", "static_rules": "legacy"}.get(
        str(pm), str(pm)[:8]
    )
    line2 += (
        f"  |  risk={summary.get('overall_risk', 'low')}  |  {pm_short}  |  pose@{infer_sz}px det@{det_px}px "
        f"conf>={conf_thres:.2f}"
    )
    (tw2, th2), bl2 = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, fb2, tb)
    tw = max(tw1, tw2)
    th = th1 + th2 + bl1 + bl2 + gap * 2
    box_w = min(w - 10, tw + gap * 3)
    cv2.rectangle(image, (5, y_line1), (5 + box_w, y_line1 + th), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(image, bn, (gap, y_line1 + gap + th1), cv2.FONT_HERSHEY_SIMPLEX, fb1, top_col, tb, cv2.LINE_AA)
    cv2.putText(
        image,
        line2,
        (gap, y_line1 + gap + th1 + th2 + bl1 + gap // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        fb2,
        (230, 230, 230),
        tb,
        cv2.LINE_AA,
    )


def process_frame(
    frame_bgr: np.ndarray,
    pose=None,
    conf: Optional[float] = None,
    imgsz: Optional[int] = None,
    object_detect_conf: Optional[float] = None,
    object_detect_imgsz: Optional[int] = None,
    object_detect_match_pose_imgsz: bool = False,
    object_detect_augment: Optional[bool] = None,
    motion_tracker: Optional["LateralMotionTracker"] = None,
    exam_overlay: str = "full",
) -> Tuple[np.ndarray, List[Status], ActivitySummary]:
    """Single best detection (highest confidence). `pose` is unused (YOLOv8)."""
    c = conf if conf is not None else float(os.environ.get("YOLO_CONF", "0.25"))
    return process_frame_multi(
        frame_bgr,
        pose,
        max_persons=1,
        conf=c,
        imgsz=imgsz,
        iou=0.7,
        object_detect_conf=object_detect_conf,
        object_detect_imgsz=object_detect_imgsz,
        object_detect_match_pose_imgsz=object_detect_match_pose_imgsz,
        object_detect_augment=object_detect_augment,
        motion_tracker=motion_tracker,
        exam_overlay=exam_overlay,
    )


def process_frame_multi(
    frame_bgr: np.ndarray,
    pose=None,
    max_persons: int = 200,
    conf: Optional[float] = None,
    imgsz: Optional[int] = None,
    iou: Optional[float] = None,
    kpt_conf_floor: Optional[float] = None,
    show_people_count: bool = True,
    object_detect_conf: Optional[float] = None,
    object_detect_imgsz: Optional[int] = None,
    object_detect_match_pose_imgsz: bool = False,
    object_detect_augment: Optional[bool] = None,
    motion_tracker: Optional["LateralMotionTracker"] = None,
    exam_overlay: str = "full",
) -> Tuple[np.ndarray, List[Status], ActivitySummary]:
    """
    YOLOv8-pose + optional YOLO object detection for exam monitoring.
    Returns annotated image, per-person statuses, and activity summary dict.

    Posture Suspicious (default): only when `motion_tracker` is set (video)—repeated
    left/right head motion vs shoulders. Single images treat pose as neutral unless
    `POSTURE_LEGACY_STATIC=1` (old head-turn/lean rules, noisy in exam halls).

    Object detection uses a higher default inference size than pose when
    `object_detect_match_pose_imgsz` is False (better on small phones/books).

    Env: YOLO_DETECT_CONF, YOLO_DETECT_IMGSZ*, YOLO_DETECT_IOU, YOLO_DETECT_AUGMENT,
    YOLO_DETECT_AGNOSTIC_NMS (*_FLOOR, *_CAP).
    Lateral motion: LATERAL_WINDOW, LATERAL_MIN_FRAMES, LATERAL_SWING_THRESHOLD,
    LATERAL_MIN_REVERSALS, LATERAL_IOU_MATCH, LATERAL_MIN_EXCURSION (both sides of midline),
    LATERAL_REVERSAL_MIN_STEP (ignore tiny frame-to-frame jitter).
    YOLO_POSE_DEDUPE_IOU: suppress overlapping duplicate pose boxes (0 disables).

    exam_overlay: "full" (default) = two-line debug banner; "minimal" = one short status line;
    "none" = no banner (compact person labels).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, [], _empty_summary()

    h, w = frame_bgr.shape[0], frame_bgr.shape[1]
    image = frame_bgr.copy()
    vis = _overlay_visual_scale(h, w)
    statuses: List[Status] = []
    summary = _empty_summary()
    # Independent of pose detection: weapon detection must still run when no
    # person/keypoints are found.
    image, weapon_detections, weapon_status = detect_and_draw_weapons(image)
    summary["weapons"] = weapon_detections
    summary["weapon_count"] = len(weapon_detections)
    summary["weapon_detection_status"] = weapon_status
    summary["weapon_alert"] = bool(weapon_detections)

    ov_raw = str(exam_overlay or "full").strip().lower()
    ov = ov_raw if ov_raw in ("full", "minimal", "none") else "full"
    compact_label = ov != "full"

    conf_thres = conf if conf is not None else float(os.environ.get("YOLO_CONF", "0.18"))
    iou_thres = iou if iou is not None else float(os.environ.get("YOLO_IOU", "0.55"))
    infer_sz = _pick_imgsz(h, w, imgsz)
    kpt_min = kpt_conf_floor if kpt_conf_floor is not None else float(os.environ.get("YOLO_KPT_CONF", "0.15"))
    use_legacy_static = os.environ.get("POSTURE_LEGACY_STATIC", "0").strip().lower() in ("1", "true", "yes")
    if motion_tracker is not None:
        summary["posture_mode"] = "lateral_motion"
    elif use_legacy_static:
        summary["posture_mode"] = "static_rules"
    else:
        summary["posture_mode"] = "snapshot_neutral"

    results = _run_yolo(
        frame_bgr,
        max_det=max(1, max_persons),
        imgsz=infer_sz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )
    if (
        not results
        or len(results) == 0
        or results[0].keypoints is None
        or getattr(results[0].keypoints, "xy", None) is None
    ):
        fs, thk = 0.85 * vis["s"], max(2, int(round(2.5 * vis["s"])))
        cv2.putText(
            image,
            "No pose detected",
            (max(w // 6, 10), h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (0, 0, 255),
            thk,
            cv2.LINE_AA,
        )
        return image, statuses, summary

    try:
        kpts = results[0].keypoints.xy.cpu().numpy()
        kconf = results[0].keypoints.conf
        if kconf is not None:
            kconf = kconf.cpu().numpy()
        else:
            kconf = np.ones((kpts.shape[0], 17), dtype=np.float32)
    except Exception as exc:
        _log.warning("Failed to read YOLO keypoints (%s); returning empty", exc)
        return image, statuses, summary

    if kpts.size == 0 or kpts.shape[0] == 0:
        fs, thk = 0.85 * vis["s"], max(2, int(round(2.5 * vis["s"])))
        cv2.putText(
            image,
            "No pose detected",
            (max(w // 6, 10), h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (0, 0, 255),
            thk,
            cv2.LINE_AA,
        )
        return image, statuses, summary

    n_cap = min(int(kpts.shape[0]), max(1, max_persons))
    pose_boxes_np: Optional[np.ndarray] = None
    if results[0].boxes is not None and results[0].boxes.xyxy is not None:
        pb = results[0].boxes.xyxy.cpu().numpy()
        bc = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None
        if len(pb) >= n_cap:
            dedupe_iou = float(os.environ.get("YOLO_POSE_DEDUPE_IOU", "0.55"))
            if dedupe_iou > 0 and n_cap > 1:
                keep = _dedupe_pose_detections(
                    pb[:n_cap],
                    bc[:n_cap] if bc is not None else None,
                    n_cap,
                    dedupe_iou,
                )
                kpts = kpts[keep]
                kconf = kconf[keep]
                pose_boxes_np = pb[keep].astype(np.float32)
            else:
                pose_boxes_np = pb[:n_cap].astype(np.float32)

    n = min(int(kpts.shape[0]), max(1, max_persons))
    kpts = kpts[:n]
    kconf = kconf[:n]
    if pose_boxes_np is not None:
        pose_boxes_np = pose_boxes_np[:n]

    people: List[Dict[str, Any]] = []
    for i in range(n):
        xy = kpts[i]
        cf = kconf[i]
        if pose_boxes_np is not None:
            pbox = np.asarray(pose_boxes_np[i], dtype=np.float32)
        else:
            b = _box_from_keypoints(xy, cf, kpt_min)
            pbox = b if b is not None else np.array([0, 0, float(w), float(h)], dtype=np.float32)
        head_off = _head_lateral_offset_normalized(xy, cf, w, h, min_kpt_conf=kpt_min)
        if use_legacy_static:
            landmarks = _landmarks_from_coco(xy, cf, w, h, min_kpt_conf=kpt_min)
            st, co = analyze_posture(landmarks, (h, w, 3))
        else:
            st, co = ("Normal", 0.62)
        people.append(
            {
                "xy": xy,
                "cf": cf,
                "status": st,
                "conf": co,
                "pbox": pbox,
                "phone_flag": False,
                "back_flag": False,
                "head_offset": head_off,
            }
        )

    if motion_tracker is not None and not use_legacy_static:
        boxes_list = [np.asarray(p["pbox"], dtype=np.float32) for p in people]
        offs = [p.get("head_offset") for p in people]
        back_candidates = [_is_back_facing_candidate(np.asarray(p["cf"], dtype=np.float32), min_kpt_conf=kpt_min) for p in people]
        motion_st, back_alerts = motion_tracker.step(boxes_list, offs, back_candidates=back_candidates)
        for idx, (st, co) in enumerate(motion_st):
            people[idx]["status"] = st
            people[idx]["conf"] = co
            if idx < len(back_alerts):
                people[idx]["back_flag"] = bool(back_alerts[idx])

    det_imgsz = _pick_detect_imgsz(
        h,
        w,
        infer_sz,
        match_pose_only=object_detect_match_pose_imgsz,
        override=object_detect_imgsz,
    )
    det_conf = _object_detect_conf(conf_thres, object_detect_conf)
    obj_items, obj_counts = _detect_exam_objects(
        frame_bgr,
        det_imgsz,
        conf=det_conf,
        augment=object_detect_augment,
    )
    summary["object_detect_imgsz"] = det_imgsz
    summary["objects"] = dict(obj_counts)
    summary["objects_list"] = list(obj_items)
    phone_near = 0
    for obj in obj_items:
        if obj["name"] not in PHONE_NAMES:
            continue
        obox = np.asarray(obj["xyxy"], dtype=np.float32)
        for j, p in enumerate(people):
            if _phone_associates_student(obox, p["pbox"]):
                people[j]["status"] = "Suspicious"
                people[j]["conf"] = max(float(people[j]["conf"]), 0.92)
                if not people[j]["phone_flag"]:
                    phone_near += 1
                people[j]["phone_flag"] = True
    summary["phone_near_person"] = phone_near
    summary["back_turned"] = sum(1 for p in people if p.get("back_flag"))

    for p in people:
        statuses.append((p["status"], float(p["conf"])))
        _draw_yolo_person(
            image,
            p["xy"],
            p["cf"],
            p["status"],
            p["conf"],
            min_kpt_conf=kpt_min,
            vis=vis,
            compact_label=compact_label,
        )

    _draw_exam_objects(image, obj_items, vis=vis)

    summary["people"] = n
    summary["normal"] = sum(1 for p in people if p["status"] == "Normal")
    summary["suspicious"] = sum(1 for p in people if p["status"] == "Suspicious")
    if n <= 0:
        summary["overall_risk"] = "low"
    elif summary["phone_near_person"] > 0 or summary["suspicious"] > max(2, int(0.45 * n)):
        summary["overall_risk"] = "high"
    elif summary["suspicious"] > 0:
        summary["overall_risk"] = "medium"
    else:
        summary["overall_risk"] = "low"

    if show_people_count and n > 0 and ov != "none":
        _draw_exam_banner(
            image,
            summary,
            infer_sz,
            conf_thres,
            y_line1=5,
            vis=vis,
            style="minimal" if ov == "minimal" else "full",
        )

    if not statuses:
        fs, thk = 0.85 * vis["s"], max(2, int(round(2.5 * vis["s"])))
        cv2.putText(
            image,
            "No pose detected",
            (max(w // 6, 10), h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (0, 0, 255),
            thk,
            cv2.LINE_AA,
        )

    return image, statuses, summary
