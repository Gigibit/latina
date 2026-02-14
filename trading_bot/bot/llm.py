from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are a trading assistant. Output JSON only with keys:
- action: one of BUY, SELL, HOLD
- confidence: integer 0-100
- reasoning: short string
- risk_notes: short string
Do not provide financial guarantees."""


class LLMDecider:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key

    def decide(self, prompt: str) -> dict[str, Any]:
        if self.provider == "openai":
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is missing.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai is not installed. Install dependencies from requirements.txt"
                ) from exc
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)

        if self.provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is not installed. Install dependencies from requirements.txt"
                ) from exc
            client = InferenceClient(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)

        raise ValueError("Unsupported LLM provider. Use 'openai' or 'huggingface'.")
