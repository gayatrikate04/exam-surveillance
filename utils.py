from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """
    Calculate the angle (in degrees) formed at point B by points A-B-C.
    """
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    c_arr = np.array(c, dtype=np.float32)

    ba = a_arr - b_arr
    bc = c_arr - b_arr

    # Protect against zero-length vectors
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 0.0

    cos_angle = np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def normalized_to_pixel_coordinates(
    x: float, y: float, image_shape: Tuple[int, int, int]
) -> Optional[Tuple[int, int]]:
    """
    Convert normalized landmark coordinates (0-1) to pixel coordinates.
    """
    height, width, _ = image_shape
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None

    px = min(int(x * width), width - 1)
    py = min(int(y * height), height - 1)
    return px, py


def get_landmark_points(
    landmarks: Sequence, image_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Convert a sequence of MediaPipe landmarks to an array of pixel coordinates.
    Landmarks with low visibility are still included but may be less reliable.
    """
    points = []
    for lm in landmarks:
        coord = normalized_to_pixel_coordinates(lm.x, lm.y, image_shape)
        if coord is not None:
            points.append(coord)

    if not points:
        return np.empty((0, 2), dtype=np.int32)

    return np.asarray(points, dtype=np.int32)


def draw_status_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    is_suspicious: bool,
    font_scale: float = 0.8,
    thickness: int = 2,
) -> None:
    """
    Draw status text on the frame with color based on suspicion.
    Increase font_scale / thickness on high-resolution images for readable exports.
    """
    color = (0, 0, 255) if is_suspicious else (0, 200, 0)  # BGR
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )

