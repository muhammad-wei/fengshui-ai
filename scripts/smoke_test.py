#!/usr/bin/env python
"""CLI smoke tests: one mode per stage, run independently before wiring
everything together. Usage:
    uv run python scripts/smoke_test.py --mode perception fixtures/raw_bedroom.jpg
    uv run python scripts/smoke_test.py --mode step
    uv run python scripts/smoke_test.py --mode deepseek
    uv run python scripts/smoke_test.py --mode e2e-a fixtures/raw_bedroom.jpg "master bedroom"
    uv run python scripts/smoke_test.py --mode e2e-b fixtures/furnished_room.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def cmd_perception(image_path: str):
    from perception.detector import FURNISHED_CLASSES, RAW_SPACE_CLASSES, YOLOWorldDetector

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: could not read {image_path}")
        return
    detector = YOLOWorldDetector()
    for name, classes in [("raw", RAW_SPACE_CLASSES), ("furnished", FURNISHED_CLASSES)]:
        dets = detector.detect(image, classes)
        print(f"[{name}] {len(dets)} detections:")
        for d in dets:
            print(f"  {d.label} conf={d.conf:.2f} bbox={d.bbox}")
        annotated = image.copy()
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, d.label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        out_path = f"logs/smoke_{name}_annotated.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"  saved: {out_path}")


def cmd_step():
    from api_clients.step_client import StepClient

    client = StepClient()
    result = client.format_json("A", {"scenario": "A"}, "Place the bed against the north wall.")
    print("format_json result:", result)


def cmd_deepseek():
    from api_clients.deepseek_client import DeepSeekClient

    client = DeepSeekClient()
    facts = {"scenario": "A", "room_purpose": "bedroom", "rule_verdicts": {}}
    result = client.narrate(facts, "A")
    print("narrate result:", result)


def cmd_wan():
    from api_clients.wan_client import WanClient

    client = WanClient()
    url = client.text_to_image("bed against the north wall, desk by the window")
    print("text_to_image result:", url)


def cmd_audio():
    from api_clients.audio_client import speak_text

    path = speak_text("Your bed should be placed against the north wall.")
    print("speak_text result:", path)


def cmd_e2e(scenario: str, image_path: str, room_purpose: str | None):
    from orchestrator import Orchestrator

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orch = Orchestrator()
    if scenario == "a":
        result = orch.run_scenario_a(image_rgb, room_purpose or "bedroom")
    else:
        result = orch.run_scenario_b(image_rgb, image_path)
    print("RESULT:", result)
    print("TIMING:", result.get("timing"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["perception", "step", "deepseek", "wan", "audio", "e2e-a", "e2e-b"])
    parser.add_argument("args", nargs="*")
    ns = parser.parse_args()

    if ns.mode == "perception":
        cmd_perception(ns.args[0])
    elif ns.mode == "step":
        cmd_step()
    elif ns.mode == "deepseek":
        cmd_deepseek()
    elif ns.mode == "wan":
        cmd_wan()
    elif ns.mode == "audio":
        cmd_audio()
    elif ns.mode == "e2e-a":
        cmd_e2e("a", ns.args[0], ns.args[1] if len(ns.args) > 1 else None)
    elif ns.mode == "e2e-b":
        cmd_e2e("b", ns.args[0], None)


if __name__ == "__main__":
    main()
