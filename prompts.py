"""Shared prompt construction for the DeepSeek (leader/narration) and Step
(local formatter) stages. Both receive the same rule base + facts; neither
ever computes geometry — perception.facts already decided rule_verdicts."""
from __future__ import annotations

import json

from perception.facts import load_rule_base

FEW_SHOT_A = {
    "facts": {
        "scenario": "A",
        "room_purpose": "master bedroom",
        "rule_verdicts": {
            "solid_back_empty_front": {"triggered": True, "evidence": {"opposite_zone": [0, 0, 213, 480]}},
        },
    },
    "output": {
        "orientation": "window faces south",
        "placements": [
            {
                "item": "bed", "wall": "N",
                "bbox_frac": [0.05, 0.05, 0.45, 0.35],
                "rationale": "North wall is windowless and doorless, satisfying Solid Back Empty Front; avoids early morning glare from the south-facing window.",
            }
        ],
        "summary_text": "Window faces south; bed placed against the north wall to keep the headboard against a solid, windowless surface.",
    },
}

FEW_SHOT_B = {
    "facts": {
        "scenario": "B",
        "rule_verdicts": {
            "mirror_facing_bed_or_door": {"triggered": True, "evidence": {"iou": 0.22, "against": "bed"}},
            "beam_pressing_down": {"triggered": True, "evidence": {"anchor": "sofa"}},
        },
    },
    "output": {
        "issues": [
            {"rule": "mirror_facing_bed_or_door", "detection_refs": [2], "severity": "high",
             "explanation": "The mirror reflects directly onto the bed, which is believed to disturb rest."},
            {"rule": "beam_pressing_down", "detection_refs": [4], "severity": "med",
             "explanation": "An exposed beam sits directly above the sofa, creating a sense of pressure over seating."},
        ],
        "dos": ["Reposition the mirror away from the bed's direct line of sight", "Add a dropped ceiling panel to soften the beam", "Add a plant near the sharp corner by the entrance"],
        "donts": ["Don't leave the mirror facing the bed", "Don't sit directly beneath the exposed beam", "Don't place the sofa back to the main door"],
        "summary_text": "Two issues found: a bed-facing mirror and a beam over the sofa.",
    },
}


LANGUAGE_NAMES = {"en": "English", "zh": "Simplified Chinese (简体中文)"}


def build_system_prompt(scenario: str, language: str = "en") -> str:
    rule_base = load_rule_base()
    schema_hint = (
        '{"orientation": str, "placements": [{"item": str, "wall": str, "bbox_frac": [x1,y1,x2,y2], "rationale": str}], "summary_text": str}'
        if scenario == "A"
        else '{"issues": [{"rule": str, "detection_refs": [int], "severity": "low|med|high", "explanation": str}], "dos": [str,str,str], "donts": [str,str,str], "summary_text": str}'
    )
    few_shot = FEW_SHOT_A if scenario == "A" else FEW_SHOT_B
    language_name = LANGUAGE_NAMES.get(language, "English")

    return f"""You are a Feng Shui interior consultant. You will be given:
1. A quantified Feng Shui rule base (JSON).
2. A "facts" object containing detected objects and rule_verdicts already computed
   deterministically in Python (IoU, angles, brightness, etc.).

You must NOT invent or recompute any geometry yourself — only narrate and act on the
rule_verdicts you are given. Respond with a single JSON object matching this shape:
{schema_hint}

JSON keys and rule ids/enum values (e.g. "wall", "severity") must stay exactly as shown above —
only the natural-language string VALUES (rationale, explanation, dos, donts, summary_text,
orientation) must be written in {language_name}.

Rule base:
{json.dumps(rule_base, indent=2)}

Example input/output pair (shown in English regardless of target language — only follow its
JSON structure, not its language):
Facts: {json.dumps(few_shot["facts"], indent=2)}
Output: {json.dumps(few_shot["output"], indent=2)}
"""


def build_user_message(facts: dict) -> str:
    return f"Facts JSON:\n{json.dumps(facts, indent=2)}"
