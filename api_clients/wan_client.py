"""Wan2.7-Image (Alibaba Cloud Model Studio / DashScope): text-to-image for
Scenario A, image-to-image/editing for Scenario B. If literal object
relocation fails (diffusion models are unreliable at strict spatial moves),
callers should fall back to a color/texture-change prompt instead."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import dashscope
from dashscope import MultiModalConversation

import config

dashscope.api_key = config.DASHSCOPE_API_KEY

STYLE_TEMPLATE = (
    "Photorealistic interior design, {style}, {layout}, natural lighting, 4k, architectural render."
)

COLOR_FALLBACK_TEMPLATE = "Keep the room layout unchanged; only {change}."

CALL_TIMEOUT_S = 45
_executor = ThreadPoolExecutor(max_workers=4)


def _call_with_timeout(**kwargs):
    # dashscope's own `timeout` kwarg is not honored by MultiModalConversation.call()
    # (observed hangs of ~300s regardless of what's passed) — enforce a real client-side
    # deadline instead so a slow/stuck call can never block a user-facing request.
    future = _executor.submit(MultiModalConversation.call, **kwargs)
    try:
        return future.result(timeout=CALL_TIMEOUT_S)
    except FutureTimeoutError:
        return None


class WanClient:
    def text_to_image(self, layout_summary: str, style: str = "modern minimalist") -> str | None:
        prompt = STYLE_TEMPLATE.format(style=style, layout=layout_summary)
        try:
            resp = _call_with_timeout(
                model="wan2.7-image",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )
            return self._extract_image_url(resp) if resp is not None else None
        except Exception:
            return None

    def edit_image(self, image_path: str, instruction: str) -> str | None:
        # DashScope's SDK expects local images as file:// URIs (it uploads them
        # behind the scenes); a bare filesystem path is not a valid image ref.
        image_ref = image_path if image_path.startswith(("http://", "https://", "file://")) else f"file://{image_path}"
        try:
            resp = _call_with_timeout(
                model="wan2.7-image",
                messages=[{
                    "role": "user",
                    "content": [{"image": image_ref}, {"text": instruction}],
                }],
            )
            return self._extract_image_url(resp) if resp is not None else None
        except Exception:
            return None

    def edit_image_with_fallback(self, image_path: str, relocation_instruction: str, color_change: str) -> str | None:
        result = self.edit_image(image_path, relocation_instruction)
        if result:
            return result
        return self.edit_image(image_path, COLOR_FALLBACK_TEMPLATE.format(change=color_change))

    @staticmethod
    def _extract_image_url(resp) -> str | None:
        try:
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                for item in content:
                    if "image" in item:
                        return item["image"]
        except Exception:
            pass
        return None
