"""DeepSeek: the orchestration "leader". Narrates rule_verdicts into draft
advice; does NOT decide which pipeline stage runs next (the UI toggle
hard-codes that) and never computes geometry itself."""
from __future__ import annotations

from openai import OpenAI

import config
from prompts import build_system_prompt, build_user_message


class DeepSeekClient:
    def __init__(self):
        # OpenAI() raises immediately on an empty-string key, not just None — fall back to a
        # placeholder so construction never crashes the app; a real auth failure still surfaces
        # (caught below) the first time a call is actually made.
        self._client = OpenAI(api_key=config.DEEPSEEK_API_KEY or "unset", base_url=config.DEEPSEEK_BASE_URL)

    def narrate(self, facts: dict, scenario: str) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": build_system_prompt(scenario)},
                    {"role": "user", "content": build_user_message(facts)},
                ],
                temperature=0.1,
                max_tokens=800,
                timeout=config.CLIENT_TIMEOUT_S,
            )
            return resp.choices[0].message.content
        except Exception:
            return None
