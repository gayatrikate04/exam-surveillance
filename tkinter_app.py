import os
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import cv2
from PIL import Image, ImageTk

from detector import LateralMotionTracker, process_frame_multi


CAM_PROBE_MAX = 10


def list_available_cameras(max_index: int = CAM_PROBE_MAX) -> List[int]:
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


class TkDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Classroom Monitor - Tkinter Live Detection")
        self.root.geometry("1280x800")

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.photo = None
        self.motion_tracker: Optional[LateralMotionTracker] = None
        self.prev_back_alert_on = False

        self.conf = tk.DoubleVar(value=float(os.environ.get("YOLO_CONF", "0.18")))
        self.max_people = tk.IntVar(value=200)
        self.camera_var = tk.StringVar(value="0")

        self._build_ui()
        self._refresh_camera_list()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Camera").pack(side=tk.LEFT)
        self.camera_combo = ttk.Combobox(top, textvariable=self.camera_var, width=10, state="readonly")
        self.camera_combo.pack(side=tk.LEFT, padx=(8, 16))

        ttk.Button(top, text="Refresh Cameras", command=self._refresh_camera_list).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Start", command=self.start).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="Stop", command=self.stop).pack(side=tk.LEFT)

        ttk.Label(top, text="Confidence").pack(side=tk.LEFT, padx=(24, 6))
        ttk.Scale(top, from_=0.05, to=0.45, variable=self.conf, orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT)

        ttk.Label(top, text="Max people").pack(side=tk.LEFT, padx=(24, 6))
        ttk.Spinbox(top, from_=1, to=250, textvariable=self.max_people, width=6).pack(side=tk.LEFT)

        self.status_label = ttk.Label(self.root, text="Idle", padding=(10, 6))
        self.status_label.pack(fill=tk.X)

        self.alert_label = tk.Label(
            self.root,
            text="No active alert",
            bg="#e8f5e9",
            fg="#1b5e20",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=6,
        )
        self.alert_label.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _refresh_camera_list(self) -> None:
        cams = list_available_cameras()
        values = [str(i) for i in cams] if cams else ["0"]
        self.camera_combo["values"] = values
        if self.camera_var.get() not in values:
            self.camera_var.set(values[0])
        self.status_label.config(text=f"Detected cameras: {', '.join(values)}")

    def start(self) -> None:
        if self.running:
            return
        cam_idx = int(self.camera_var.get())
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            self.status_label.config(text=f"Failed to open camera {cam_idx}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap = cap
        self.motion_tracker = LateralMotionTracker()
        self.running = True
        self.prev_back_alert_on = False
        self.status_label.config(text=f"Streaming camera {cam_idx}...")
        self._update_frame()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.motion_tracker = None
        self.alert_label.config(text="No active alert", bg="#e8f5e9", fg="#1b5e20")
        self.status_label.config(text="Stopped")

    def _update_frame(self) -> None:
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status_label.config(text="No frame from camera.")
            self.root.after(40, self._update_frame)
            return

        tracker = self.motion_tracker if self.motion_tracker is not None else LateralMotionTracker()
        annotated, statuses, summary = process_frame_multi(
            frame,
            None,
            max_persons=int(self.max_people.get()),
            conf=float(self.conf.get()),
            imgsz=960,
            iou=0.55,
            kpt_conf_floor=0.12,
            object_detect_conf=0.18,
            object_detect_match_pose_imgsz=False,
            object_detect_augment=False,
            motion_tracker=tracker,
            exam_overlay="minimal",
        )

        back_count = int(summary.get("back_turned", 0))
        back_alert_on = back_count > 0
        if back_alert_on:
            self.alert_label.config(
                text=f"ALERT: {back_count} person(s) are looking back",
                bg="#ffebee",
                fg="#b71c1c",
            )
            if not self.prev_back_alert_on:
                self.root.bell()
        else:
            self.alert_label.config(text="No active alert", bg="#e8f5e9", fg="#1b5e20")
        self.prev_back_alert_on = back_alert_on

        n = len(statuses)
        susp = int(summary.get("suspicious", 0))
        self.status_label.config(text=f"People: {n} | Suspicious: {susp} | Risk: {summary.get('overall_risk', 'low')}")

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.photo)

        self.root.after(30, self._update_frame)

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = TkDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
