
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = Path("models/weapon_detector_candidate2.pt")
OUTPUT_DIR = Path("runs/weapon_test")


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python test_weapon_detector.py "path\\to\\image.jpg"'
        )
        return 1

    image_path = Path(sys.argv[1])

    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print("Could not read image.")
        return 1

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Inference device:", device)

    model = YOLO(str(MODEL_PATH))
    print("Model classes:", model.names)

    names = {
        str(name).strip().lower()
        for name in model.names.values()
    }

    if not any("knife" in name for name in names):
        print("WARNING: No knife class found.")
    if not any(
        any(word in name for word in ("handgun", "gun", "firearm", "pistol"))
        for name in names
    ):
        print("WARNING: No gun-related class found.")

    results = model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        device=device,
        verbose=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "annotated_test.jpg"

    result = results[0]
    annotated = result.plot()
    if not cv2.imwrite(str(output_path), annotated):
        print("Failed to save annotated image.")
        return 1

    print("\nDetections:")
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections in this image.")
    else:
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(model.names[class_id])
            confidence = float(box.conf.item())
            coordinates = [
                round(float(value), 1)
                for value in box.xyxy[0].tolist()
            ]
            print(
                f"{label}: confidence={confidence:.3f}, "
                f"box={coordinates}"
            )

    print(f"\nAnnotated image saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
