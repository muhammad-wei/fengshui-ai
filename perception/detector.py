"""YOLO-World open-vocabulary detector.

Stock COCO classes don't include door/window/mirror/beam/column, which are
central to the Feng Shui rules, so we use YOLO-World's free-text class
prompting instead of a COCO-pretrained YOLOv8n (no fine-tuning needed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLOWorld

RAW_SPACE_CLASSES = ["door", "window", "wall", "beam", "column"]
FURNISHED_CLASSES = [
    "bed", "sofa", "desk", "chair", "wardrobe", "door", "window",
    "mirror", "dining table", "tv", "potted plant",
]


@dataclass
class Detection:
    label: str
    bbox: tuple[float, float, float, float]
    conf: float

    def to_dict(self) -> dict:
        return {"label": self.label, "bbox": list(self.bbox), "conf": self.conf}


class YOLOWorldDetector:
    """Loaded once at app startup; stays resident (<1.5GB) alongside the LLM server."""

    def __init__(self, weights: str = "yolov8s-worldv2.pt", conf: float = 0.15):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLOWorld(weights)
        # set_classes()'s CLIP text encoder infers its device from the main model's
        # parameters — without this, the model defaults to CPU while .predict() below
        # runs inference on CUDA, causing a "tensors on different devices" crash.
        self.model.to(self.device)
        self.conf = conf

    def detect(self, image: np.ndarray, classes: list[str]) -> list[Detection]:
        self.model.set_classes(classes)
        results = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]

        detections: list[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(Detection(label=label, bbox=(x1, y1, x2, y2), conf=conf))
        return detections
