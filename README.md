# Cheating Detection Prototype (MediaPipe + OpenCV)

This project is a posture-based cheating suspicion detection prototype built with MediaPipe Pose, OpenCV, and NumPy.  
It analyzes a student's posture from webcam, video, or image input and overlays a simple **Normal / Suspicious** status on the frame.

## Features

- **Input sources**
  - Live webcam
  - Video file
  - Image file
- **Pose-based rules**
  - Head turning left/right relative to shoulders
  - Leaning too far forward/down
  - Excessive shoulder tilt
- **On-screen overlays**
  - Pose landmarks and connections
  - Bounding box around detected person
  - Status label with optional confidence percentage

> This is a prototype for posture-based suspicion detection only.  
> It does **not** perform face recognition, identity tracking, or any deep-learning training.

## Installation

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

From the `cheating_detection` directory, run:

```bash
python -m cheating_detection.main
```

You will be prompted to select an input source:

```text
Select input source:
1 - Live Webcam
2 - Video File
3 - Image File
Enter choice (1/2/3):
```

- **1 - Live Webcam**
  - Uses your default webcam (`cv2.VideoCapture(0)`).
  - Press `q` or `ESC` in the window to exit.
- **2 - Video File**
  - Prompts for a video path.
  - Press `q` or `ESC` to stop playback.
- **3 - Image File**
  - Prompts for an image path.
  - Processes once and waits for any key press in the window to close.

## Notes

- The system runs entirely on CPU and uses MediaPipe Pose in streaming mode for real-time webcam performance.
- Detection logic is purely rule-based on pose landmarks; there is no training step.

