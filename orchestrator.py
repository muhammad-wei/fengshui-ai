"""Linear flow: YOLO -> geometry -> DeepSeek(leader) -> Step(local, format) -> Wan(image) -> TTS.
No dynamic agent loop — the Scenario A/B toggle hard-codes which path runs."""
from __future__ import annotations

import time

import cv2
import numpy as np

import config
from api_clients.deepseek_client import DeepSeekClient
from api_clients.step_client import StepClient
from api_clients.wan_client import WanClient
from perception.detector import FURNISHED_CLASSES, RAW_SPACE_CLASSES, YOLOWorldDetector
from perception.facts import build_scene_facts, load_rule_base
from perception.geometry import area_filter
from schema import AdjustmentReport, LayoutPlan, validate_and_repair

GOLDEN_FALLBACK_A = "assets/golden_scenario_a.jpg"
GOLDEN_FALLBACK_B = "assets/golden_scenario_b.jpg"


_FALLBACK_TEXT = {
    "en": {
        "layout_summary": "We couldn't generate a detailed layout this time — please try again with a clearer photo of the room's doors and windows.",
        "adjustment_summary": "We couldn't analyze this photo in detail this time — here is general Feng Shui guidance instead.",
        "dos": ["Keep the main walking path clear", "Add a plant near sharp corners", "Keep mirrors away from the bed"],
        "donts": ["Don't place the bed facing the door", "Don't sit with your back to the door", "Don't block the entrance"],
        "dos_header": "**Do's**",
        "donts_header": "**Don'ts**",
    },
    "zh": {
        "layout_summary": "这次未能生成详细的布局方案——请换一张能清楚看到房间门窗的照片再试一次。",
        "adjustment_summary": "这次未能详细分析这张照片——以下是通用的风水建议。",
        "dos": ["保持主要通道畅通", "在尖锐墙角附近摆放绿植", "让镜子远离床铺"],
        "donts": ["不要让床正对房门", "不要背对房门而坐", "不要堵塞入口"],
        "dos_header": "**宜**",
        "donts_header": "**忌**",
    },
}


def _fallback_layout_plan(facts: dict, language: str = "en") -> LayoutPlan:
    t = _FALLBACK_TEXT.get(language, _FALLBACK_TEXT["en"])
    return LayoutPlan(orientation="unknown", placements=[], summary_text=t["layout_summary"])


def _fallback_adjustment_report(facts: dict, language: str = "en") -> AdjustmentReport:
    t = _FALLBACK_TEXT.get(language, _FALLBACK_TEXT["en"])
    return AdjustmentReport(issues=[], dos=t["dos"], donts=t["donts"], summary_text=t["adjustment_summary"])


class Orchestrator:
    def __init__(self):
        self.detector = YOLOWorldDetector(conf=config.YOLO_CONF_THRESHOLD)
        self.deepseek = DeepSeekClient()
        self.step = StepClient()
        self.wan = WanClient()
        self.rule_base = load_rule_base()

    @staticmethod
    def _prep_image(image_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_rgb, config.IMAGE_RESIZE)
        return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    def _get_plan(self, facts: dict, scenario: str, model_cls, language: str = "en"):
        t0 = time.perf_counter()
        draft = self.deepseek.narrate(facts, scenario, language)
        t1 = time.perf_counter()
        result = None
        if draft is not None:
            formatted = self.step.format_json(scenario, facts, draft, language)
            result = validate_and_repair(formatted, model_cls) if formatted else None
            if result is None and formatted is not None:
                # one retry with a corrective nudge
                retry_draft = self.deepseek.narrate(facts, scenario, language)
                if retry_draft:
                    formatted_retry = self.step.format_json(scenario, facts, retry_draft, language)
                    result = validate_and_repair(formatted_retry, model_cls) if formatted_retry else None
        t2 = time.perf_counter()
        timing = {"deepseek_s": t1 - t0, "step_s": t2 - t1}
        return result, timing

    def run_scenario_a(self, image_rgb: np.ndarray, room_purpose: str, language: str = "en") -> dict:
        t_start = time.perf_counter()
        image = self._prep_image(image_rgb)

        t0 = time.perf_counter()
        detections = self.detector.detect(image, RAW_SPACE_CLASSES)
        det_dicts = area_filter([d.to_dict() for d in detections], image.shape[0] * image.shape[1])
        t1 = time.perf_counter()

        facts = build_scene_facts(det_dicts, image, "A", self.rule_base, room_purpose)

        plan, llm_timing = self._get_plan(facts, "A", LayoutPlan, language)
        if plan is None:
            plan = _fallback_layout_plan(facts, language)

        t2 = time.perf_counter()
        image_url = self.wan.text_to_image(plan.summary_text)
        if image_url is None:
            image_url = GOLDEN_FALLBACK_A
        t3 = time.perf_counter()

        return {
            "image": image_url,
            "text": plan.summary_text,
            "timing": {
                "perception_s": t1 - t0,
                **llm_timing,
                "generation_s": t3 - t2,
                "total_s": time.perf_counter() - t_start,
            },
        }

    def run_scenario_b(self, image_rgb: np.ndarray, image_path: str, language: str = "en") -> dict:
        t_start = time.perf_counter()
        image = self._prep_image(image_rgb)

        t0 = time.perf_counter()
        detections = self.detector.detect(image, FURNISHED_CLASSES)
        det_dicts = area_filter([d.to_dict() for d in detections], image.shape[0] * image.shape[1])
        t1 = time.perf_counter()

        facts = build_scene_facts(det_dicts, image, "B", self.rule_base, None)

        report, llm_timing = self._get_plan(facts, "B", AdjustmentReport, language)
        if report is None:
            report = _fallback_adjustment_report(facts, language)

        t2 = time.perf_counter()
        # Use the actionable "dos" (concrete edits) as the Wan instruction, not the issues'
        # diagnostic explanations — Wan needs something to actually DO, not a description of
        # what's wrong, or the image comes back essentially unchanged.
        relocation_instr = "; ".join(report.dos) or "improve the room's Feng Shui"
        color_change = "adjust the wall color to a calmer, cooler tone"
        image_url = self.wan.edit_image_with_fallback(image_path, relocation_instr, color_change)
        if image_url is None:
            image_url = GOLDEN_FALLBACK_B
        t3 = time.perf_counter()

        t = _FALLBACK_TEXT.get(language, _FALLBACK_TEXT["en"])
        checklist_md = f"{t['dos_header']}\n" + "\n".join(f"- {d}" for d in report.dos)
        checklist_md += f"\n\n{t['donts_header']}\n" + "\n".join(f"- {d}" for d in report.donts)

        return {
            "image": image_url,
            "text": report.summary_text,
            "checklist": checklist_md,
            "timing": {
                "perception_s": t1 - t0,
                **llm_timing,
                "generation_s": t3 - t2,
                "total_s": time.perf_counter() - t_start,
            },
        }
