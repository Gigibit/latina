from __future__ import annotations

import json
import logging
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

TREND_EVALUATION_SYSTEM_PROMPT = """You are a trading ranking assistant.
You receive candidate symbols with momentum and volume context.
Return JSON only with this schema:
{
  "evaluations": [
    {
      "symbol": "TICKER",
      "llm_score": number from 0 to 100,
      "summary": "short comparison-driven reason"
    }
  ]
}
Rules:
- Score should reflect relative attractiveness between provided candidates.
- Prefer higher 5d momentum confirmed by healthy relative volume.
- Penalize unstable downside momentum and weak participation.
- Keep each summary under 140 characters.
"""

logger = logging.getLogger(__name__)


def _truncate_for_log(value: str, max_chars: int = 700) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated {len(value) - max_chars} chars>"


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


def _normalize_trend_evaluation(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for item in evaluations:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        raw_score = item.get("llm_score", 50)
        try:
            llm_score = float(raw_score)
        except (TypeError, ValueError):
            llm_score = 50.0
        llm_score = max(0.0, min(100.0, llm_score))
        summary = str(item.get("summary", "No LLM comparison provided.")).strip()
        normalized[symbol] = {
            "llm_score": round(llm_score, 2),
            "summary": summary,
        }

    return normalized


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
            request_payload = {
                "model": self.model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            logger.info(
                ("External request service=openai endpoint=chat.completions "
                "provider=%s host=%s payload=%s"),
                self.provider,
                str(getattr(client, "base_url", "https://api.openai.com")).rstrip("/"),
                _truncate_for_log(json.dumps(request_payload, ensure_ascii=False)),
            )
            resp = client.chat.completions.create(
                **request_payload,
            )
            content = resp.choices[0].message.content or "{}"
            logger.info(
                ("External response service=openai endpoint=chat.completions "
                "provider=%s model=%s output_chars=%s"),
                self.provider,
                self.model,
                len(content),
            )
            return _normalize_decision(json.loads(content))

        if self.provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is not installed. Install dependencies from requirements.txt"
                ) from exc
            client = InferenceClient(api_key=self.api_key)
            request_payload = {
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            logger.info(
                ("External request service=huggingface endpoint=chat.completions "
                "provider=%s host=%s payload=%s"),
                self.provider,
                "https://api-inference.huggingface.co",
                _truncate_for_log(json.dumps(request_payload, ensure_ascii=False)),
            )
            resp = client.chat.completions.create(
                **request_payload,
            )
            content = resp.choices[0].message.content or "{}"
            logger.info(
                ("External response service=huggingface endpoint=chat.completions "
                "provider=%s model=%s output_chars=%s"),
                self.provider,
                self.model,
                len(content),
            )
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
            request_payload = {
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": READABLE_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
            }
            logger.info(
                ("External request service=openai endpoint=chat.completions "
                "provider=%s host=%s payload=%s"),
                self.provider,
                str(getattr(client, "base_url", "https://api.openai.com")).rstrip("/"),
                _truncate_for_log(json.dumps(request_payload, ensure_ascii=False)),
            )
            resp = client.chat.completions.create(
                **request_payload,
            )
            content = (resp.choices[0].message.content or "").strip()
            logger.info(
                ("External response service=openai endpoint=chat.completions "
                "provider=%s model=%s output_chars=%s"),
                self.provider,
                self.model,
                len(content),
            )
            return content

        if self.provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is not installed. Install dependencies from requirements.txt"
                ) from exc
            client = InferenceClient(api_key=self.api_key)
            request_payload = {
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": READABLE_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
            }
            logger.info(
                ("External request service=huggingface endpoint=chat.completions "
                "provider=%s host=%s payload=%s"),
                self.provider,
                "https://api-inference.huggingface.co",
                _truncate_for_log(json.dumps(request_payload, ensure_ascii=False)),
            )
            resp = client.chat.completions.create(
                **request_payload,
            )
            content = (resp.choices[0].message.content or "").strip()
            logger.info(
                ("External response service=huggingface endpoint=chat.completions "
                "provider=%s model=%s output_chars=%s"),
                self.provider,
                self.model,
                len(content),
            )
            return content

        raise ValueError("Unsupported LLM provider. Use 'openai' or 'huggingface'.")

    def evaluate_trending_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        ranking_prompt = (
            "Evaluate and compare these trending candidates for short-term trading quality. "
            "Use only the provided metrics and return compact JSON.\n\n"
            f"Candidates:\n{json.dumps(candidates, indent=2, sort_keys=True)}"
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
            request_payload = {
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": TREND_EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": ranking_prompt},
                ],
            }
            logger.info(
                (
                    "External request service=openai endpoint=chat.completions "
                    "provider=%s host=%s payload=%s"
                ),
                self.provider,
                str(getattr(client, "base_url", "https://api.openai.com")).rstrip("/"),
                _truncate_for_log(json.dumps(request_payload, ensure_ascii=False)),
            )
            resp = client.chat.completions.create(
                **request_payload,
            )
            content = resp.choices[0].message.content or "{}"
            logger.info(
                (
                    "External response service=openai endpoint=chat.completions "
                    "provider=%s model=%s output_chars=%s"
                ),
                self.provider,
                self.model,
                len(content),
            )
            return _normalize_trend_evaluation(json.loads(content))

        raise ValueError("Unsupported LLM provider for trend evaluation. Use 'openai'.")
