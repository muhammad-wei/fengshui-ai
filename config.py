"""Single source of truth for paths, endpoints, and feature flags."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# Step-3.7-Flash via StepFun's cloud "Step Plan" API (OpenAI-compatible) — not local llama.cpp.
STEP_API_KEY = os.environ.get("STEP_API_KEY", "")
STEP_BASE_URL = os.environ.get("STEP_BASE_URL", "https://api.stepfun.com/step_plan/v1")
STEP_MODEL = os.environ.get("STEP_MODEL", "step-3.7-flash")

SEND_IMAGE_TO_LLM = os.environ.get("SEND_IMAGE_TO_LLM", "false").lower() == "true"

CLIENT_TIMEOUT_S = 10
IMAGE_RESIZE = (640, 640)
YOLO_CONF_THRESHOLD = 0.15
