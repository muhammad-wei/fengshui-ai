"""Step-3.7-Flash via StepFun's cloud "Step Plan" API (OpenAI-compatible) —
not a local llama-server. Acts as a formatter here: turns DeepSeek's draft
narration into strict schema JSON, not a second reasoning pass."""
from __future__ import annotations

from openai import OpenAI

import config
from prompts import build_system_prompt


class StepClient:
    def __init__(self):
        # See deepseek_client.py — OpenAI() raises on an empty-string key at construction time.
        self._client = OpenAI(api_key=config.STEP_API_KEY or "unset", base_url=config.STEP_BASE_URL)

    def format_json(self, scenario: str, facts: dict, draft_advice: str) -> str | None:
        format_instruction = (
            "You are a strict JSON formatter. Given the rule base, the facts, and a draft "
            "of the advice, output ONLY a single valid JSON object matching the schema "
            "described in the system prompt. No prose, no markdown fences."
        )
        try:
            resp = self._client.chat.completions.create(
                model=config.STEP_MODEL,
                messages=[
                    {"role": "system", "content": build_system_prompt(scenario) + "\n\n" + format_instruction},
                    {"role": "user", "content": f"Draft advice:\n{draft_advice}"},
                ],
                temperature=0.1,
                # Step is a reasoning model — it burns completion tokens on a separate
                # "reasoning" field before emitting final `content`. Measured ~1700 completion
                # tokens for our full rule-base system prompt; reasoning_effort=low did not
                # reliably shrink that, so max_tokens needs real headroom or `content` comes
                # back truncated/empty (finish_reason="length"). This materially affects the
                # end-to-end latency budget — see README known limitations.
                max_tokens=4096,
                extra_body={"reasoning_effort": "low"},
                timeout=45,
            )
            return resp.choices[0].message.content
        except Exception:
            return None
