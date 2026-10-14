
from pathlib import Path

import torch
from ultralytics import YOLO

model_path = Path("models/weapon_detector_candidate2.pt")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is unavailable; inference may use CPU.")

model = YOLO(str(model_path))

print("\nModel classes:")
for class_id, class_name in model.names.items():
    print(f"  {class_id}: {class_name}")

names = {str(name).strip().lower() for name in model.names.values()}

print("\nGun-like classes:", [
    name for name in names
    if any(word in name for word in ("gun", "firearm", "pistol", "rifle"))
])

print("Knife-like classes:", [
    name for name in names if "knife" in name
])
