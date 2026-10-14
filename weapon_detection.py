"""Standalone weapon detection helper for the classroom-monitoring prototype.

Uses a separate Ultralytics YOLO checkpoint. Detection results are unverified alerts,
not proof of an incident. Configure with WEAPON_MODEL_PATH and WEAPON_CONF.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    torch = None
    YOLO = None

_log = logging.getLogger(__name__)
_model = None
_model_lock = threading.Lock()
_predict_lock = threading.Lock()


def _get_model():
    global _model
    if YOLO is None:
        raise RuntimeError("Weapon detector dependencies missing; install ultralytics and torch.")
    if _model is None:
        model_path = os.environ.get(
            "WEAPON_MODEL_PATH", "models/weapon_detector_candidate2.pt"
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Weapon model not found: {model_path}. Set WEAPON_MODEL_PATH."
            )
        with _model_lock:
            if _model is None:
                _model = YOLO(model_path)
                _log.info("Loaded weapon model %s; classes=%s", model_path, _model.names)
    return _model


def detect_and_draw_weapons(
    image_bgr: np.ndarray,
    confidence: float | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
    """Return annotated image, normalized detections, and status ('ok' or 'error')."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr, [], "error"

    threshold = confidence
    if threshold is None:
        threshold = float(os.environ.get("WEAPON_CONF", "0.35"))
    threshold = max(0.05, min(0.95, float(threshold)))

    try:
        model = _get_model()
        device = 0 if torch is not None and torch.cuda.is_available() else "cpu"
        with _predict_lock:
            results = model.predict(
                source=image_bgr,
                conf=threshold,
                imgsz=int(os.environ.get("WEAPON_IMGSZ", "640")),
                device=device,
                verbose=False,
            )
        if not results:
            return image_bgr, [], "ok"

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return image_bgr, [], "ok"

        names = result.names if isinstance(result.names, dict) else model.names
        annotated = image_bgr.copy()
        detections: List[Dict[str, Any]] = []
        height, width = annotated.shape[:2]

        for box in boxes:
            class_id = int(box.cls.item())
            label_raw = str(names.get(class_id, class_id))
            label_lower = label_raw.lower()
            if "knife" in label_lower:
                label = "knife"
            elif any(word in label_lower for word in ("handgun", "gun", "firearm", "pistol")):
                label = "handgun"
            else:
                # Keep unexpected model classes visible for debugging, but don't
                # silently relabel them as a weapon category.
                label = label_raw.strip().lower()

            score = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            detections.append({
                "label": label,
                "raw_label": label_raw,
                "confidence": round(score, 4),
                "xyxy": [x1, y1, x2, y2],
            })

            # Red outline and explicit "UNVERIFIED" label to avoid presenting a
            # model prediction as a confirmed weapon incident.
            caption = f"UNVERIFIED {label} {score:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
            text_y = max(24, y1 - 8)
            cv2.putText(
                annotated, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
            )

        return annotated, detections, "ok"
    except Exception:
        _log.exception("Weapon detector failed")
        return image_bgr, [], "error"
