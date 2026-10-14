"""
FastAPI server for Next.js frontend: image / video / live-frame inference.

Run from the project root:
  python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, AsyncIterator, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app_pipeline import (
    annotated_png_bytes,
    iter_video_frames_sampled,
    list_available_cameras,
    process_image,
)
from detector import LateralMotionTracker, process_frame, process_frame_multi

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

app = FastAPI(title="Classroom Monitor API", version="1.0.0")

_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:3001,http://127.0.0.1:3001",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

_live_trackers: Dict[str, LateralMotionTracker] = {}

# Prototype-only in-memory incident storage. Incidents disappear when the
# server restarts. Add authentication and persistent storage before deployment.
_incident_lock = Lock()
_incidents: List[Dict[str, Any]] = []
_incident_seen: Dict[str, float] = {}


def _record_weapon_incidents(
    summary: Dict[str, Any],
    source: str,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Record weapon detections for human review, suppressing short repeats."""
    detections = summary.get("weapons", [])
    if not isinstance(detections, list):
        return []

    created: List[Dict[str, Any]] = []
    now = time.time()

    with _incident_lock:
        # Prune deduplication keys older than 30 seconds.
        for key, stamp in list(_incident_seen.items()):
            if now - stamp > 30:
                _incident_seen.pop(key, None)

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            label = str(detection.get("label", "unknown"))
            try:
                confidence = float(detection.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0.0

            # Suppress repeated detections of the same category from the same
            # source/session for 10 seconds.
            dedupe_key = f"{source}|{session_id or '-'}|{label}"
            if now - _incident_seen.get(dedupe_key, 0) < 10:
                continue
            _incident_seen[dedupe_key] = now

            incident = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "session_id": session_id,
                "type": label,
                "confidence": confidence,
                "bbox": detection.get("xyxy"),
                "status": "needs_review",
                "review_note": "",
            }
            _incidents.append(incident)
            created.append(incident)

        # Keep the prototype bounded to the latest 500 incidents.
        if len(_incidents) > 500:
            del _incidents[:-500]

    return created


# Serialize heavy video processing. Increase MAX_CONCURRENT_VIDEOS only if the
# machine has enough memory for multiple concurrent inference jobs.
_video_semaphore = asyncio.Semaphore(
    max(1, int(os.environ.get("MAX_CONCURRENT_VIDEOS", "1")))
)


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and nested values for JSON responses."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        try:
            return _json_safe(obj.item())
        except (TypeError, ValueError):
            return str(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return base64.standard_b64encode(obj).decode("ascii")
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except (TypeError, ValueError):
            pass
    return str(obj)


def _api_json(payload: Any) -> JSONResponse:
    """Return JSON without Pydantic re-serialization of numpy values."""
    return JSONResponse(content=_json_safe(payload))


def _statuses_json(statuses: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for status, confidence in statuses or []:
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        result.append({"label": str(status), "confidence": score})
    return result


def _summary_to_json(summary: Dict[str, Any]) -> Dict[str, Any]:
    result = _json_safe(summary)
    if not isinstance(result, dict):
        return {}
    # Frontend only needs counts; omit large raw object lists from API payloads.
    result.pop("objects_list", None)
    return result


def _bgr_to_png_b64(bgr: np.ndarray) -> str:
    raw = annotated_png_bytes(bgr)
    return base64.standard_b64encode(raw).decode("ascii")


def _bgr_to_jpeg_b64(bgr: np.ndarray, quality: int = 82) -> str:
    ok, buffer = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return _bgr_to_png_b64(bgr)
    return base64.standard_b64encode(buffer.tobytes()).decode("ascii")


def _decode_image_upload(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return bgr


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/admin/incidents")
def get_incidents(status: Optional[str] = None) -> Dict[str, Any]:
    with _incident_lock:
        rows = list(reversed(_incidents))

    if status:
        rows = [row for row in rows if row["status"] == status]

    return {"incidents": rows, "count": len(rows)}


@app.patch("/api/admin/incidents/{incident_id}")
async def review_incident(
    incident_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    new_status = str(payload.get("status", "")).strip()
    allowed_statuses = {"reviewed", "false_positive", "escalated"}
    if new_status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="status must be reviewed, false_positive, or escalated",
        )

    note = str(payload.get("review_note", ""))[:1000]

    with _incident_lock:
        for row in _incidents:
            if row["id"] == incident_id:
                row["status"] = new_status
                row["review_note"] = note
                return {"incident": row}

    raise HTTPException(status_code=404, detail="Incident not found")


@app.get("/api/cameras")
def get_cameras(max_index: int = 10) -> Dict[str, List[int]]:
    return {"indices": list_available_cameras(max_index)}


@app.post("/api/process/image")
async def process_image_endpoint(
    file: UploadFile = File(...),
    multi_person: bool = Form(True),
    yolo_conf: float = Form(0.18),
    yolo_imgsz: int = Form(1280),
    max_people: int = Form(200),
    yolo_iou: float = Form(0.55),
    kpt_floor: float = Form(0.12),
    object_detect_conf: float = Form(0.2),
    object_detect_hires: bool = Form(True),
    object_tta: bool = Form(False),
    exam_overlay: str = Form("full"),
) -> Dict[str, Any]:
    data = await file.read()
    bgr = _decode_image_upload(data)

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

    _record_weapon_incidents(summary, source="image")
    original_base64 = _bgr_to_png_b64(bgr)
    annotated_base64 = _bgr_to_png_b64(annotated)

    return _api_json(
        {
            "original_png_base64": original_base64,
            "annotated_png_base64": annotated_base64,
            "summary": _summary_to_json(summary),
            "statuses": _statuses_json(statuses),
        }
    )


@app.post("/api/process/video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    multi_person: bool = Form(True),
    yolo_conf: float = Form(0.18),
    yolo_imgsz: int = Form(960),
    max_people: int = Form(200),
    yolo_iou: float = Form(0.55),
    kpt_floor: float = Form(0.12),
    object_detect_conf: float = Form(0.2),
    object_detect_hires: bool = Form(True),
    object_tta: bool = Form(False),
    exam_overlay: str = Form("full"),
    max_frames: int = Form(20),
) -> StreamingResponse:
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    max_upload_mb = int(os.environ.get("MAX_VIDEO_UPLOAD_MB", "300"))
    max_upload_bytes = max_upload_mb * 1024 * 1024
    total = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        path = temporary_file.name
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_upload_bytes:
                temporary_file.close()
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Video larger than {max_upload_mb} MB. "
                        "Set MAX_VIDEO_UPLOAD_MB to raise the cap."
                    ),
                )
            temporary_file.write(chunk)

    _log.info(
        "video upload received: %.1f MB -> %s",
        total / (1024 * 1024),
        path,
    )

    max_frames_capped = max(
        1,
        min(
            int(max_frames),
            int(os.environ.get("MAX_VIDEO_FRAMES", "60")),
        ),
    )
    imgsz_capped = max(320, min(int(yolo_imgsz), 960))
    people_capped = max(
        1,
        min(int(max_people), int(os.environ.get("MAX_PEOPLE", "200"))),
    )

    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue(maxsize=2)
    error_holder: Dict[str, Any] = {}
    last_summary_holder: Dict[str, Any] = {"v": {}}
    count_holder: Dict[str, int] = {"n": 0}
    loop = asyncio.get_running_loop()

    def _producer() -> None:
        try:
            frames = iter_video_frames_sampled(
                path,
                max_frames=max_frames_capped,
                multi_person=multi_person,
                yolo_conf=yolo_conf,
                yolo_imgsz=imgsz_capped,
                max_people=people_capped,
                yolo_iou=yolo_iou,
                kpt_floor=kpt_floor,
                object_detect_conf=object_detect_conf,
                object_detect_hires=object_detect_hires,
                object_tta=object_tta,
                exam_overlay=exam_overlay,
            )

            for index, (frame_bgr, statuses, summary) in enumerate(frames):
                _record_weapon_incidents(
                    summary,
                    source="video",
                    session_id=path,
                )

                frame_object = {
                    "index": index,
                    "annotated_png_base64": _bgr_to_jpeg_b64(frame_bgr),
                    "media_type": "image/jpeg",
                    "summary": _summary_to_json(summary),
                    "statuses": _statuses_json(statuses),
                }
                last_summary_holder["v"] = frame_object["summary"]
                count_holder["n"] = index + 1

                asyncio.run_coroutine_threadsafe(
                    queue.put(json.dumps(frame_object)),
                    loop,
                ).result()

                del frame_bgr, statuses, summary, frame_object
                if (index + 1) % 4 == 0:
                    gc.collect()

        except Exception as exc:
            _log.exception("Video processing failed mid-stream")
            error_holder["exc"] = exc
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    async def _stream_body() -> AsyncIterator[bytes]:
        async with _video_semaphore:
            try:
                producer_task = asyncio.create_task(asyncio.to_thread(_producer))
                yield b'{"frames":['
                first = True

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    separator = b"" if first else b","
                    first = False
                    yield separator + item.encode("utf-8")

                await producer_task
                tail: Dict[str, Any] = {
                    "frame_count": count_holder["n"],
                    "last_summary": last_summary_holder["v"],
                }
                if "exc" in error_holder:
                    tail["error"] = (
                        f"Video processing failed: {error_holder['exc']}"
                    )

                yield b"]," + json.dumps(tail)[1:].encode("utf-8")
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return StreamingResponse(_stream_body(), media_type="application/json")


@app.post("/api/live/frame")
async def live_frame_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(""),
    multi_person: bool = Form(True),
    yolo_conf: float = Form(0.22),
    yolo_imgsz: int = Form(960),
    max_people: int = Form(200),
    yolo_iou: float = Form(0.55),
    kpt_floor: float = Form(0.12),
    object_detect_conf: float = Form(0.2),
    object_detect_hires: bool = Form(True),
    object_tta: bool = Form(False),
    exam_overlay: str = Form("minimal"),
) -> Dict[str, Any]:
    session = session_id.strip() or str(uuid.uuid4())
    if session not in _live_trackers:
        _live_trackers[session] = LateralMotionTracker()

    data = await file.read()
    bgr = _decode_image_upload(data)
    yolo_imgsz_capped = min(int(yolo_imgsz), 960)
    augment = True if object_tta else None
    match_pose = not object_detect_hires
    tracker = _live_trackers[session]

    try:
        if multi_person:
            annotated, statuses, summary = process_frame_multi(
                bgr,
                None,
                max_persons=max_people,
                conf=float(yolo_conf),
                imgsz=yolo_imgsz_capped,
                iou=float(yolo_iou),
                kpt_conf_floor=float(kpt_floor),
                object_detect_conf=float(object_detect_conf),
                object_detect_match_pose_imgsz=match_pose,
                object_detect_augment=augment,
                motion_tracker=tracker,
                exam_overlay=exam_overlay,
            )
        else:
            annotated, statuses, summary = process_frame(
                bgr,
                None,
                conf=max(0.2, float(yolo_conf)),
                imgsz=yolo_imgsz_capped,
                object_detect_conf=float(object_detect_conf),
                object_detect_match_pose_imgsz=match_pose,
                object_detect_augment=augment,
                motion_tracker=tracker,
                exam_overlay=exam_overlay,
            )
    except Exception as exc:
        # Keep a live polling client alive if a single frame fails.
        _log.warning(
            "Live frame inference failed (%s); returning raw frame",
            exc,
        )
        annotated = bgr
        statuses = []
        summary = {
            "people": 0,
            "normal": 0,
            "suspicious": 0,
            "objects": {},
            "phone_near_person": 0,
            "back_turned": 0,
            "overall_risk": "low",
            "error": "inference_failed",
            "weapons": [],
            "weapon_count": 0,
            "weapon_detection_status": "error",
            "weapon_alert": False,
        }

    _record_weapon_incidents(summary, source="live", session_id=session)

    height, width = annotated.shape[:2]
    minimum_width = int(os.environ.get("LIVE_DISPLAY_MIN_W", "1280"))
    if width < minimum_width:
        scale = minimum_width / max(1, width)
        annotated = cv2.resize(
            annotated,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )

    return _api_json(
        {
            "session_id": session,
            "annotated_png_base64": _bgr_to_png_b64(annotated),
            "summary": _summary_to_json(summary),
            "statuses": _statuses_json(statuses),
        }
    )


@app.delete("/api/live/session/{session_id}")
def live_session_delete(session_id: str) -> Dict[str, str]:
    _live_trackers.pop(session_id, None)
    return {"ok": "true", "session_id": session_id}



