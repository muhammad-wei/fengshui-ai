"""Turns detections + geometry into the rule_verdicts dict — the Python-side
cross-check the spec demands. The LLM narrates these verdicts, it never
recomputes geometry itself, which confines hallucination risk to phrasing.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import geometry as geo

RULE_BASE_PATH = Path(__file__).resolve().parent.parent / "rules" / "rule_base.json"


def load_rule_base() -> list[dict]:
    return json.loads(RULE_BASE_PATH.read_text())


def _find(detections: list[dict], label: str) -> dict | None:
    matches = [d for d in detections if d["label"] == label]
    return max(matches, key=lambda d: d["conf"]) if matches else None


def _anchor_bbox(detections: list[dict]) -> dict | None:
    for label in ("bed", "sofa"):
        found = _find(detections, label)
        if found:
            return found
    return None


def build_scene_facts(
    detections: list[dict],
    image,
    scenario: str,
    rule_base: list[dict],
    room_purpose: str | None = None,
) -> dict:
    h, w = image.shape[:2]
    verdicts: dict[str, dict] = {}

    for rule in rule_base:
        rid = rule["id"]
        params = rule.get("params", {})
        applies = rule["scenario"] in (scenario, "both")
        verdict = {"applicable": applies, "triggered": False, "evidence": {}}

        if not applies:
            verdicts[rid] = verdict
            continue

        if rid == "solid_back_empty_front":
            window = _find(detections, "window")
            if window:
                wx = (window["bbox"][0] + window["bbox"][2]) / 2
                # Opposite third of frame from the window is the candidate headboard wall.
                opposite_zone = (0, 0, w / 3, h) if wx > w / 2 else (2 * w / 3, 0, w, h)
                blockers = [
                    d for d in detections
                    if d["label"] in ("door", "window")
                    and geo.iou(tuple(d["bbox"]), opposite_zone) > 0
                ]
                verdict["triggered"] = len(blockers) == 0
                verdict["evidence"] = {"opposite_zone": opposite_zone, "blockers": blockers}
            else:
                verdict["evidence"] = {"note": "no window detected"}

        elif rid == "open_bright_hall":
            anchor = _anchor_bbox(detections)
            if anchor:
                ratio = geo.open_area_ratio(detections, tuple(anchor["bbox"]), (h, w))
                verdict["triggered"] = ratio < params["min_open_ratio"]
                verdict["evidence"] = {"open_ratio": ratio, "anchor": anchor["label"]}

        elif rid == "door_window_alignment":
            door, window = _find(detections, "door"), _find(detections, "window")
            if door and window:
                angle = geo.angle_between_centers(tuple(door["bbox"]), tuple(window["bbox"]))
                verdict["triggered"] = angle < params["max_angle_deg"]
                verdict["evidence"] = {"angle_deg": angle}

        elif rid == "mirror_facing_bed_or_door":
            mirror = _find(detections, "mirror")
            if mirror:
                best_iou, against = 0.0, None
                for label in ("bed", "door"):
                    target = _find(detections, label)
                    if target:
                        val = geo.iou(tuple(mirror["bbox"]), tuple(target["bbox"]))
                        if val > best_iou:
                            best_iou, against = val, label
                verdict["triggered"] = best_iou > params["min_iou"]
                verdict["evidence"] = {"iou": best_iou, "against": against}

        elif rid == "beam_pressing_down":
            anchor = _anchor_bbox(detections)
            if anchor:
                triggered, band = geo.horizontal_dark_band(
                    image, tuple(anchor["bbox"]),
                    min_width_ratio=params["min_width_ratio"],
                    dark_delta=params["dark_delta"],
                )
                verdict["triggered"] = triggered
                verdict["evidence"] = {"band_bbox": band, "anchor": anchor["label"]}

        elif rid == "left_high_right_low":
            balance = geo.left_right_height_balance(detections, w)
            verdict["triggered"] = not balance["left_higher"]
            verdict["evidence"] = balance

        elif rid == "color_balance":
            cb = geo.color_balance(image)
            verdict["triggered"] = cb["warm_ratio"] > params["warm_ratio_threshold"]
            verdict["evidence"] = cb

        elif rid == "plants_neutralize_sha":
            plant = _find(detections, "potted plant")
            verdict["triggered"] = plant is None
            verdict["evidence"] = {"has_plant": plant is not None}

        elif rid == "dynamic_static_separation":
            table, door = _find(detections, "dining table"), _find(detections, "door")
            if table and door:
                diag = (w**2 + h**2) ** 0.5
                tx, ty = geo.bbox_center(tuple(table["bbox"]))
                dx, dy = geo.bbox_center(tuple(door["bbox"]))
                dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
                dist_frac = dist / diag
                verdict["triggered"] = dist_frac < params["min_distance_frac"]
                verdict["evidence"] = {"distance_frac": dist_frac}

        elif rid == "wealth_area_bright":
            door = _find(detections, "door")
            corners = {
                "top_left": (0, 0, w * 0.3, h * 0.3),
                "top_right": (w * 0.7, 0, w, h * 0.3),
                "bottom_left": (0, h * 0.7, w * 0.3, h),
                "bottom_right": (w * 0.7, h * 0.7, w, h),
            }
            brightness = {name: geo.region_brightness(image, box) for name, box in corners.items()}
            brightest = max(brightness, key=brightness.get)
            if door:
                dx, _ = geo.bbox_center(tuple(door["bbox"]))
                door_side = "left" if dx < w / 2 else "right"
                target_corner = "bottom_right" if door_side == "left" else "bottom_left"
                verdict["triggered"] = brightest != target_corner
                verdict["evidence"] = {"brightness": brightness, "target_corner": target_corner}
            else:
                verdict["evidence"] = {"brightness": brightness, "note": "no door detected, target corner unknown"}

        verdicts[rid] = verdict

    return {
        "scenario": scenario,
        "room_purpose": room_purpose,
        "image_size": {"width": w, "height": h},
        "detections": detections,
        "rule_verdicts": verdicts,
    }
