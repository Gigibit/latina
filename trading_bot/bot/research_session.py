from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from trading_bot.bot.data_sources import (
    fetch_trending_symbols,
    fetch_x_sentiment_scores,
    get_market_snapshot,
)
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.service import generate_suggestion


def _build_summary_decider() -> LLMDecider:
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        api_key = os.getenv("HUGGINGFACE_API_KEY")
    return LLMDecider(provider=provider, model=model, api_key=api_key)


def _summarize_result_with_llm(payload: dict[str, Any]) -> str:
    try:
        decider = _build_summary_decider()
        summary = decider.summarize_json(payload)
        if summary:
            return summary
    except Exception:
        pass
    decision = payload.get("decision", {})
    action = str(decision.get("action", "HOLD"))
    confidence = decision.get("confidence", "n/a")
    risk_notes = str(decision.get("risk_notes", "No risk notes available."))
    symbol = str(payload.get("symbol", "N/A"))
    return (
        f"- Symbol: {symbol}\n"
        f"- Action: {action} (confidence: {confidence})\n"
        f"- Risk notes: {risk_notes}"
    )


@dataclass
class ResearchJob:
    session_id: str
    risk_profile: str
    candidate_symbols: list[str] | None = None
    status: str = "running"
    stream_log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class ResearchSessionStore:
    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        session_id: str,
        risk_profile: str,
        candidate_symbols: list[str] | None = None,
    ) -> ResearchJob:
        with self._lock:
            existing = self._jobs.get(session_id)
            if existing and existing.status == "running":
                return existing

            job = ResearchJob(
                session_id=session_id,
                risk_profile=risk_profile,
                candidate_symbols=candidate_symbols,
            )
            self._jobs[session_id] = job
            thread = threading.Thread(
                target=self._run_job,
                args=(job,),
                name=f"research-{uuid.uuid4().hex[:8]}",
                daemon=True,
            )
            thread.start()
            return job

    def get(self, session_id: str) -> ResearchJob | None:
        with self._lock:
            return self._jobs.get(session_id)

    def _append_log(self, job: ResearchJob, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            job.stream_log.append(f"[{timestamp}] {message}")

    @staticmethod
    def _auto_detection_symbol_number() -> int:
        raw_value = os.getenv("AUTO_DETECTION_SYMBOL_NUMBER", "7").strip()
        if raw_value.lower() in {"false", "off", "no"}:
            return 0
        return max(int(raw_value), 0)

    def _run_job(self, job: ResearchJob) -> None:
        try:
            self._append_log(job, "Research session started.")
            symbol, score, ranking = self._discover_symbol(job)
            self._append_log(
                job,
                (
                    f"Selected symbol {symbol} with discovery score {score:.2f}. "
                    "Running suggestion model..."
                ),
            )

            result = generate_suggestion(symbol=symbol, user_risk_profile=job.risk_profile)
            threshold = float(os.getenv("SYMBOL_ACTION_ACCEPTANCE_THRESHOLD", "0"))
            confidence = float(result.get("decision", {}).get("confidence", 0) or 0)
            accepted = confidence >= threshold
            result["discovery"] = {
                "selected_symbol": symbol,
                "score": round(score, 4),
                "ranking": ranking,
                "acceptance_threshold": threshold,
                "accepted": accepted,
            }
            if not accepted:
                self._append_log(
                    job,
                    (
                        "Confidence below SYMBOL_ACTION_ACCEPTANCE_THRESHOLD. "
                        "Forcing HOLD to keep execution safe."
                    ),
                )
                result["decision"]["action"] = "HOLD"
                result["decision"]["risk_notes"] = (
                    f"Confidence {confidence:.2f} below threshold {threshold:.2f}."
                )

            result["readable_summary"] = _summarize_result_with_llm(result)
            self._append_log(job, "Research complete.")
            with self._lock:
                job.result = result
                job.status = "completed"
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                job.error = str(exc)
                job.status = "failed"
            self._append_log(job, f"Research failed: {exc}")

    def _discover_symbol(self, job: ResearchJob) -> tuple[str, float, list[dict[str, float | str]]]:
        self._append_log(job, "Scraping trending symbols and sentiment signals from web data.")
        auto_detection_symbol_number = self._auto_detection_symbol_number()
        if auto_detection_symbol_number > 0:
            symbols = fetch_trending_symbols(limit=auto_detection_symbol_number)
            self._append_log(
                job,
                (
                    "AUTO_DETECTION_SYMBOL_NUMBER="
                    f"{auto_detection_symbol_number}; using auto-selected symbols."
                ),
            )
        else:
            symbols = job.candidate_symbols or []
            if not symbols:
                raise ValueError(
                    ("AUTO_DETECTION_SYMBOL_NUMBER disables auto mode. "
                    "Provide symbols separated by commas.")
                )
            self._append_log(job, "Auto symbol detection disabled; using symbols from user input.")

        today = date.today()
        last_days = [today - timedelta(days=offset) for offset in range(5)]

        ranking: list[dict[str, float | str]] = []
        for symbol in symbols:
            try:
                snapshot = get_market_snapshot(symbol)
            except ValueError:
                self._append_log(job, f"Skipping {symbol}: insufficient market history.")
                continue

            sentiment_map = fetch_x_sentiment_scores(symbol=symbol, days=last_days)
            sentiment = sum(sentiment_map.values()) / max(len(sentiment_map), 1)
            relative_volume = snapshot.latest_volume / max(snapshot.avg_volume_20d, 1)
            score = snapshot.pct_change_5d + (relative_volume - 1) * 8 + sentiment * 10
            ranking.append(
                {
                    "symbol": snapshot.symbol,
                    "score": round(score, 4),
                    "sentiment": round(sentiment, 4),
                    "pct_change_5d": round(snapshot.pct_change_5d, 4),
                }
            )
            self._append_log(
                job,
                (
                    f"Analyzed {snapshot.symbol}: sentiment={sentiment:.3f}, "
                    f"5d_change={snapshot.pct_change_5d:.2f}, score={score:.2f}."
                ),
            )

        if not ranking:
            raise ValueError("No symbols available from scraping phase.")

        ranking.sort(key=lambda row: float(row["score"]), reverse=True)
        best = ranking[0]
        return str(best["symbol"]), float(best["score"]), ranking


research_sessions = ResearchSessionStore()
