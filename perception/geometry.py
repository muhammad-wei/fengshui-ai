"""Pure, deterministic geometry checks for the Section-4 Feng Shui rule base.

No LLM involvement here: every function takes bounding boxes / pixel data and
returns a number or bool. The LLM only ever sees the output dict from
perception.facts.build_scene_facts(), never raw coordinates or images.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

BBox = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def bbox_center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def angle_between_centers(box_a: BBox, box_b: BBox) -> float:
    """Angle (degrees, 0-90) between the line joining the two bbox centers and the horizontal."""
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return 0.0
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    return angle


def region_brightness(image: np.ndarray, box: BBox) -> float:
    """Mean V-channel (HSV) brightness of a bbox region, 0-255."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def horizontal_dark_band(
    image: np.ndarray,
    furniture_bbox: BBox,
    min_width_ratio: float = 0.5,
    dark_delta: float = 30,
) -> tuple[bool, BBox | None]:
    """Look for a horizontal dark band directly above furniture_bbox (candidate exposed beam).

    Crops a strip above the furniture, computes row-wise mean grayscale intensity,
    flags a contiguous dark row-run spanning >= min_width_ratio of the furniture's width,
    optionally confirmed by a near-horizontal Hough line.
    """
    fx1, fy1, fx2, fy2 = furniture_bbox
    h, w = image.shape[:2]
    strip_y1 = max(0, int(fy1 - (fy2 - fy1) * 0.6))
    strip_y2 = int(fy1)
    x1, x2 = max(0, int(fx1)), min(w, int(fx2))
    if strip_y2 <= strip_y1 or x2 <= x1:
        return False, None

    strip = image[strip_y1:strip_y2, x1:x2]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    overall_mean = row_means.mean()
    dark_rows = np.where(row_means < overall_mean - dark_delta)[0]
    if dark_rows.size == 0:
        return False, None

    band_bbox = (float(x1), float(strip_y1 + dark_rows.min()), float(x2), float(strip_y1 + dark_rows.max() + 1))
    band_width_ratio = (x2 - x1) / max(1.0, (fx2 - fx1))
    if band_width_ratio < min_width_ratio:
        return False, None

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=int((x2 - x1) * 0.4), maxLineGap=10)
    has_horizontal_line = False
    if lines is not None:
        for x1_l, y1_l, x2_l, y2_l in lines[:, 0]:
            line_angle = math.degrees(math.atan2(abs(y2_l - y1_l), abs(x2_l - x1_l) + 1e-6))
            if line_angle <= 10:
                has_horizontal_line = True
                break

    return (dark_rows.size > 0 and (has_horizontal_line or band_width_ratio >= min_width_ratio)), band_bbox


def left_right_height_balance(detections: list[dict], image_width: float) -> dict:
    """Average bbox pixel-height per side of the image midline, as a proxy for prominence."""
    midline = image_width / 2
    left_heights, right_heights = [], []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2
        height = y2 - y1
        (left_heights if cx < midline else right_heights).append(height)
    left_avg = sum(left_heights) / len(left_heights) if left_heights else 0.0
    right_avg = sum(right_heights) / len(right_heights) if right_heights else 0.0
    return {"left_avg_height": left_avg, "right_avg_height": right_avg, "left_higher": left_avg > right_avg}


def open_area_ratio(detections: list[dict], anchor_bbox: BBox, image_shape: tuple[int, int]) -> float:
    """Fraction of the trapezoid 'in front of' anchor_bbox (toward the camera/bottom of frame)
    that is NOT covered by any other detection's bbox.
    """
    h, w = image_shape[:2]
    ax1, ay1, ax2, ay2 = anchor_bbox
    front_box = (ax1, ay2, ax2, min(h, ay2 + (ay2 - ay1) * 1.5))
    fx1, fy1, fx2, fy2 = front_box
    front_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
    if front_area <= 0:
        return 1.0

    mask = np.zeros((int(fy2 - fy1), int(fx2 - fx1)), dtype=np.uint8)
    for det in detections:
        if tuple(det["bbox"]) == anchor_bbox:
            continue
        dx1, dy1, dx2, dy2 = det["bbox"]
        ix1, iy1 = max(dx1, fx1), max(dy1, fy1)
        ix2, iy2 = min(dx2, fx2), min(dy2, fy2)
        if ix2 > ix1 and iy2 > iy1:
            mask[int(iy1 - fy1):int(iy2 - fy1), int(ix1 - fx1):int(ix2 - fx1)] = 1

    covered_ratio = float(mask.sum()) / mask.size if mask.size else 0.0
    return 1.0 - covered_ratio


def color_balance(image: np.ndarray) -> dict:
    """Warm (red/orange) vs cool (blue/green) hue-histogram balance, 0-1 ratios."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32)  # OpenCV hue range 0-179
    warm_mask = ((hue >= 0) & (hue <= 20)) | (hue >= 150)
    cool_mask = (hue >= 45) & (hue <= 130)
    total = hue.size
    return {
        "warm_ratio": float(warm_mask.sum()) / total,
        "cool_ratio": float(cool_mask.sum()) / total,
    }


def area_filter(detections: list[dict], image_area: float, max_ratio: float = 0.6) -> list[dict]:
    """Drop detections whose bbox covers an implausibly large fraction of the frame
    (spec risk mitigation: avoids e.g. curtains misread as full walls/doors)."""
    return [d for d in detections if bbox_area(d["bbox"]) < max_ratio * image_area]
