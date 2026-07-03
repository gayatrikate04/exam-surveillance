import os
from typing import Any, Callable

import cv2

from detector import LateralMotionTracker, process_frame_multi


def _run_on_capture(
    capture: cv2.VideoCapture,
    frame_processor: Callable[[cv2.Mat], Any],
    window_title: str,
) -> None:
    if not capture.isOpened():
        print("Error: Unable to open video source.")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        annotated_frame = frame_processor(frame)
        cv2.imshow(window_title, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    capture.release()
    cv2.destroyAllWindows()


def _make_frame_processor(multi_person: bool) -> Tuple[LateralMotionTracker, Callable[[cv2.Mat], cv2.Mat]]:
    tracker = LateralMotionTracker()
    conf = float(os.environ.get("YOLO_CONF", "0.22"))

    def _proc(frame: cv2.Mat) -> cv2.Mat:
        annotated, _, __ = process_frame_multi(
            frame,
            None,
            max_persons=200 if multi_person else 1,
            conf=conf,
            motion_tracker=tracker,
        )
        return annotated

    return tracker, _proc


def run_webcam() -> None:
    print("Starting webcam (YOLOv8-Pose + lateral-motion posture on video, press 'q' or ESC)...")
    cap = cv2.VideoCapture(0)
    _, proc = _make_frame_processor(multi_person=True)
    _run_on_capture(cap, proc, "Cheating Detection - Webcam")


def run_video_file() -> None:
    path = input("Enter path to video file: ").strip().strip('"')
    if not path:
        print("No path provided.")
        return

    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return

    cap = cv2.VideoCapture(path)
    print("Processing video file (YOLOv8-Pose + lateral-motion, press 'q' or ESC)...")
    _, proc = _make_frame_processor(multi_person=True)
    _run_on_capture(cap, proc, "Cheating Detection - Video")


def run_image_file() -> None:
    path = input("Enter path to image file: ").strip().strip('"')
    if not path:
        print("No path provided.")
        return

    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return

    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to read image: {path}")
        return

    annotated_image, _, __ = process_frame_multi(
        image,
        None,
        max_persons=200,
        conf=float(os.environ.get("YOLO_CONF", "0.22")),
        motion_tracker=None,
    )

    cv2.imshow("Cheating Detection - Image", annotated_image)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    print("Select input source (YOLOv8-Pose):")
    print("1 - Live Webcam")
    print("2 - Video File")
    print("3 - Image File")

    choice = input("Enter choice (1/2/3): ").strip()

    try:
        if choice == "1":
            run_webcam()
        elif choice == "2":
            run_video_file()
        elif choice == "3":
            run_image_file()
        else:
            print("Invalid choice. Please run the program again and select 1, 2, or 3.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
