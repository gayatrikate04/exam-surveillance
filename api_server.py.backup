"""
FastAPI server for Next.js frontend: image / video / live-frame inference.

Run from repo root (same directory as detector weights):
  pip install fastapi uvicorn python-multipart
  uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import os
import tempfile
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

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

app = FastAPI(title="Classroom Monitor API", version="1.0.0")

_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:3001,http://127.0.0.1:3001",
    ).split(",")
    if o.strip()
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

# Serialize heavy video processing — running two of these in parallel was OOM'ing
# the uvicorn worker and producing the ECONNREFUSED / socket-hang-up cascade on
# the Next.js proxy. Override with MAX_CONCURRENT_VIDEOS if you have headroom.
_video_semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_VIDEOS", "1")))


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types (and nested dicts/lists) for JSON responses."""
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
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
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
    """Return JSON without Pydantic re-serialization (avoids numpy edge cases)."""
    return JSONResponse(content=_json_safe(payload))


def _statuses_json(statuses: Any) -> List[Dict[str, Any]]:
    return [{"label": str(s), "confidence": float(c)} for s, c in statuses]


def _summary_to_json(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = _json_safe(summary)
    if not isinstance(out, dict):
        return {}
    # Frontend only needs counts; drop heavy raw detections from API payloads.
    out.pop("objects_list", None)
    return out


def _bgr_to_png_b64(bgr: np.ndarray) -> str:
    raw = annotated_png_bytes(bgr)
    return base64.standard_b64encode(raw).decode("ascii")


def _bgr_to_jpeg_b64(bgr: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return _bgr_to_png_b64(bgr)
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _decode_image_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return bgr


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


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
    orig_b64 = _bgr_to_png_b64(bgr)
    ann_b64 = _bgr_to_png_b64(annotated)
    return _api_json(
        {
            "original_png_base64": orig_b64,
            "annotated_png_base64": ann_b64,
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
    # Stream upload to disk in chunks instead of reading the whole video into RAM.
    # A 200MB MP4 + the full numpy frame buffers + YOLO weights was OOM'ing the
    # uvicorn worker on /api/process/video, which is what produced ECONNREFUSED.
    max_upload_mb = int(os.environ.get("MAX_VIDEO_UPLOAD_MB", "300"))
    max_upload_bytes = max_upload_mb * 1024 * 1024
    total = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            total += len(chunk)
            if total > max_upload_bytes:
                tmp.close()
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"Video larger than {max_upload_mb} MB. Set MAX_VIDEO_UPLOAD_MB to raise the cap.",
                )
            tmp.write(chunk)
    _log.info("video upload received: %.1f MB -> %s", total / (1024 * 1024), path)

    # Clamp client-supplied knobs so a runaway request can't OOM the worker.
    max_frames_capped = max(1, min(int(max_frames), int(os.environ.get("MAX_VIDEO_FRAMES", "60"))))
    imgsz_capped = max(320, min(int(yolo_imgsz), 960))
    people_capped = max(1, min(int(max_people), int(os.environ.get("MAX_PEOPLE", "200"))))

    # Stream the response: process one frame, JSON-encode it, write it to the
    # wire, drop the numpy buffer, and only then process the next. Peak RAM
    # stays at ~1 frame instead of N — this is what stops uvicorn from being
    # OOM-killed mid-request (the cause of the ECONNREFUSED/socket-hang-up
    # cascade you were seeing). The output is still one JSON document so the
    # existing client `await res.json()` keeps working unchanged.
    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue(maxsize=2)
    err_holder: Dict[str, Any] = {}

    def _producer() -> None:
        try:
            for i, (frame_bgr, statuses, summ) in enumerate(
                iter_video_frames_sampled(
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
            ):
                obj = {
                    "index": i,
                    "annotated_png_base64": _bgr_to_jpeg_b64(frame_bgr),
                    "media_type": "image/jpeg",
                    "summary": _summary_to_json(summ),
                    "statuses": _statuses_json(statuses),
                }
                last_summary_holder["v"] = obj["summary"]
                count_holder["n"] = i + 1
                asyncio.run_coroutine_threadsafe(queue.put(json.dumps(obj)), loop).result()
                # Caller owns the frame now; let GC reclaim the BGR buffer.
                del frame_bgr, statuses, summ, obj
                gc.collect() if (i + 1) % 4 == 0 else None
        except Exception as exc:
            _log.exception("Video processing failed mid-stream")
            err_holder["exc"] = exc
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    last_summary_holder: Dict[str, Any] = {"v": {}}
    count_holder: Dict[str, int] = {"n": 0}
    loop = asyncio.get_running_loop()

    async def _stream_body() -> "AsyncIterator[bytes]":
        async with _video_semaphore:
            try:
                producer_task = asyncio.create_task(asyncio.to_thread(_producer))
                # JSON prelude: opens the array of frames.
                yield b'{"frames":['
                first = True
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    sep = b"" if first else b","
                    first = False
                    yield sep + item.encode("utf-8")
                await producer_task
                # JSON tail: closes the array and appends totals/last summary.
                tail = {
                    "frame_count": count_holder["n"],
                    "last_summary": last_summary_holder["v"],
                }
                if "exc" in err_holder:
                    tail["error"] = f"Video processing failed: {err_holder['exc']}"
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
    sid = session_id.strip() or str(uuid.uuid4())
    if sid not in _live_trackers:
        _live_trackers[sid] = LateralMotionTracker()

    data = await file.read()
    bgr = _decode_image_upload(data)
    y_imgsz = min(int(yolo_imgsz), 960)
    aug = True if object_tta else None
    match_pose = not object_detect_hires
    tracker = _live_trackers[sid]

    try:
        if multi_person:
            annotated, statuses, summary = process_frame_multi(
                bgr,
                None,
                max_persons=max_people,
                conf=float(yolo_conf),
                imgsz=y_imgsz,
                iou=float(yolo_iou),
                kpt_conf_floor=float(kpt_floor),
                object_detect_conf=float(object_detect_conf),
                object_detect_match_pose_imgsz=match_pose,
                object_detect_augment=aug,
                motion_tracker=tracker,
                exam_overlay=exam_overlay,
            )
        else:
            annotated, statuses, summary = process_frame(
                bgr,
                None,
                conf=max(0.2, float(yolo_conf)),
                imgsz=y_imgsz,
                object_detect_conf=float(object_detect_conf),
                object_detect_match_pose_imgsz=match_pose,
                object_detect_augment=aug,
                motion_tracker=tracker,
                exam_overlay=exam_overlay,
            )
    except Exception as exc:
        # Never 500 a live stream on a single bad frame — return the raw frame so
        # the client keeps polling instead of tearing down its connection.
        _log.warning("live frame inference failed (%s); returning raw frame", exc)
        annotated = bgr
        statuses = []
        summary = {
            "people": 0, "normal": 0, "suspicious": 0, "objects": {},
            "phone_near_person": 0, "back_turned": 0, "overall_risk": "low",
            "error": "inference_failed",
        }

    h, w = annotated.shape[:2]
    min_w = int(os.environ.get("LIVE_DISPLAY_MIN_W", "1280"))
    if w < min_w:
        scale = min_w / max(1, w)
        annotated = cv2.resize(
            annotated,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )

    return _api_json(
        {
            "session_id": sid,
            "annotated_png_base64": _bgr_to_png_b64(annotated),
            "summary": _summary_to_json(summary),
            "statuses": _statuses_json(statuses),
        }
    )


@app.delete("/api/live/session/{session_id}")
def live_session_delete(session_id: str) -> Dict[str, str]:
    _live_trackers.pop(session_id, None)
    return {"ok": "true", "session_id": session_id}
