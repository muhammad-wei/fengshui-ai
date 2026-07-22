"""UI string translations for the bilingual (English/Chinese) interface.
Keyed by language code ("en" / "zh"), matching prompts.LANGUAGE_NAMES."""
from __future__ import annotations

UI = {
    "en": {
        "title": "# AI Interior Feng Shui Consultant",
        "mode_label": "Scenario",
        "mode_choices": [("Raw Space", "raw"), ("Furnished Space", "furnished")],
        "purpose_label": "Intended use (Raw Space only)",
        "purpose_placeholder": "e.g. master bedroom — or click \U0001F3A4 to speak",
        "generate_btn": "Generate",
        "before": "### Before",
        "after": "### After",
        "read_aloud_btn": "\U0001F50A Read Aloud",
        "upload_prompt": "Please upload a photo first.",
        "error_prefix": "Something went wrong generating your report: ",
        "default_purpose": "bedroom",
    },
    "zh": {
        "title": "# AI 室内风水顾问",
        "mode_label": "场景",
        "mode_choices": [("毛坯空间", "raw"), ("已装修空间", "furnished")],
        "purpose_label": "用途（仅毛坯空间）",
        "purpose_placeholder": "例如：主卧 —— 或点击 \U0001F3A4 说出来",
        "generate_btn": "生成",
        "before": "### 之前",
        "after": "### 之后",
        "read_aloud_btn": "\U0001F50A 朗读",
        "upload_prompt": "请先上传一张照片。",
        "error_prefix": "生成报告时出现问题：",
        "default_purpose": "卧室",
    },
}
