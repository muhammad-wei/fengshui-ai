"""Structured output contracts for the two scenarios, plus JSON validation/repair."""
from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError


class Placement(BaseModel):
    item: str  # "bed" | "desk" | "wardrobe"
    wall: str  # "N" | "S" | "E" | "W"
    bbox_frac: list[float]  # [x1, y1, x2, y2] as fractions of image size
    rationale: str


class LayoutPlan(BaseModel):
    orientation: str
    placements: list[Placement]
    summary_text: str


class Issue(BaseModel):
    rule: str
    detection_refs: list[int] = []
    severity: str  # "low" | "med" | "high"
    explanation: str


class AdjustmentReport(BaseModel):
    issues: list[Issue]
    dos: list[str]
    donts: list[str]
    summary_text: str


def validate_and_repair(raw_text: str, model_cls: type[BaseModel]) -> BaseModel | None:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import json_repair
            data = json_repair.loads(raw_text)
        except Exception:
            return None

    try:
        return model_cls.model_validate(data)
    except ValidationError:
        return None
