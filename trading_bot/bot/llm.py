from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are a trading assistant. Output JSON only with keys:
- action: one of BUY, SELL, HOLD
- confidence: integer 0-100
- reasoning: short string
- risk_notes: short string
Rules:
- Keep reasoning concise and grounded in provided evidence.
- Prefer HOLD when evidence is mixed or confidence is below 55.
- Do not provide financial guarantees."""

READABLE_SUMMARY_SYSTEM_PROMPT = """You are a trading research assistant.
Convert the provided JSON payload into a concise, human-readable summary.
Focus on actionable highlights, key risks, and confidence.
Use plain text with short bullet points.
Do not include markdown code blocks.
Do not repeat the full JSON.
"""


def _normalize_decision(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action", "HOLD")).upper().strip()
    if action not in {"BUY", "SELL", "HOLD"}:
        action = "HOLD"

    raw_confidence = payload.get("confidence", 50)
    try:
        confidence = int(round(float(raw_confidence)))
    except (TypeError, ValueError):
        confidence = 50
    confidence = max(0, min(100, confidence))

    reasoning = str(payload.get("reasoning", "Insufficient model reasoning provided.")).strip()
    risk_notes = str(payload.get("risk_notes", "No explicit risk notes provided.")).strip()

    return {
        "action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "risk_notes": risk_notes,
    }


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
            return _normalize_decision(json.loads(content))

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
            return _normalize_decision(json.loads(content))

        raise ValueError("Unsupported LLM provider. Use 'openai' or 'huggingface'.")

    def summarize_json(self, payload: dict[str, Any]) -> str:
        summary_prompt = (
            "Summarize this trading run payload for a dashboard user. "
            "Prioritize final decision, confidence, symbol, and risk notes.\n\n"
            f"JSON payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
        )

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
                messages=[
                    {"role": "system", "content": READABLE_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

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
                    {"role": "system", "content": READABLE_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        raise ValueError("Unsupported LLM provider. Use 'openai' or 'huggingface'.")
