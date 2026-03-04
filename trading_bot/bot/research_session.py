from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from trading_bot.bot.data_sources import (
    fetch_trending_symbols,
    fetch_x_sentiment_scores,
    get_market_snapshot,
)
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.service import generate_suggestion

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_MAX_ATTEMPTS = 8
DEFAULT_RATE_LIMIT_BASE_BACKOFF_SECONDS = 2.0
MAX_RATE_LIMIT_BACKOFF_SECONDS = 60.0


def _build_summary_decider() -> LLMDecider:
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        api_key = os.getenv("HUGGINGFACE_API_KEY")
    return LLMDecider(provider=provider, model=model, api_key=api_key)


def _is_env_flag_enabled(name: str, default: bool = True) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


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


def _is_rate_limited_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return "too many requests" in message or "rate limit" in message


def _rate_limit_backoff_seconds(attempt: int) -> float:
    base_delay = float(
        os.getenv(
            "SUGGESTION_RATE_LIMIT_BACKOFF_BASE_SECONDS",
            str(DEFAULT_RATE_LIMIT_BASE_BACKOFF_SECONDS),
        )
    )
    backoff = base_delay * (2 ** max(attempt - 1, 0))
    return min(backoff, MAX_RATE_LIMIT_BACKOFF_SECONDS)


def _suggestion_with_retry(
    symbol: str,
    risk_profile: str,
    log_callback: Callable[[str], None],
) -> dict[str, Any]:
    max_attempts = int(
        os.getenv("SUGGESTION_RATE_LIMIT_MAX_ATTEMPTS", str(DEFAULT_RATE_LIMIT_MAX_ATTEMPTS))
    )
    max_attempts = max(max_attempts, 1)

    attempt = 1
    while True:
        try:
            return generate_suggestion(symbol=symbol, user_risk_profile=risk_profile)
        except Exception as exc:
            if (not _is_rate_limited_exception(exc)) or attempt >= max_attempts:
                raise

            delay = _rate_limit_backoff_seconds(attempt)
            log_callback(
                (
                    "Suggestion model rate limited "
                    f"(attempt {attempt}/{max_attempts}). "
                    f"Retrying in {delay:.1f}s."
                )
            )
            time.sleep(delay)
            attempt += 1


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
            logger.info(
                "Research workload started session_id=%s risk_profile=%s candidate_symbols=%s",
                job.session_id,
                job.risk_profile,
                job.candidate_symbols,
            )
            symbol, score, ranking = self._discover_symbol(job)
            logger.info(
                "Research workload symbol discovered session_id=%s symbol=%s "
                "score=%.4f ranking_size=%s",
                job.session_id,
                symbol,
                score,
                len(ranking),
            )
            self._append_log(
                job,
                (
                    f"Selected symbol {symbol} with discovery score {score:.2f}. "
                    "Running suggestion model..."
                ),
            )

            logger.info(
                "Research workload generating suggestion session_id=%s symbol=%s",
                job.session_id,
                symbol,
            )
            result = _suggestion_with_retry(
                symbol=symbol,
                risk_profile=job.risk_profile,
                log_callback=lambda message: self._append_log(job, message),
            )
            logger.info(
                "Research workload suggestion generated session_id=%s symbol=%s decision=%s",
                job.session_id,
                symbol,
                result.get("decision", {}).get("action"),
            )
            threshold = float(os.getenv("SYMBOL_ACTION_ACCEPTANCE_THRESHOLD", "0"))
            decision = result.setdefault("decision", {})
            calibrated_confidence = float(
                decision.get("final_confidence", result.get("final_confidence", 0)) or 0
            )
            confidence_source = str(
                decision.get("confidence_source", result.get("confidence_source", "model"))
            ).strip()
            has_reliable_confidence = confidence_source == "model"
            accepted = calibrated_confidence >= threshold if has_reliable_confidence else True
            result["discovery"] = {
                "selected_symbol": symbol,
                "score": round(score, 4),
                "ranking": ranking,
                "acceptance_threshold": threshold,
                "accepted": accepted,
                "confidence_source": confidence_source,
                "confidence_reliable": has_reliable_confidence,
            }
            if not has_reliable_confidence:
                decision["decision_quality"] = "low_schema_reliability"
                result["discovery"]["warning"] = (
                    "Confidence schema deemed non reliable "
                    f"(confidence_source={confidence_source}). "
                    "Acceptance threshold was not applied."
                )
                self._append_log(
                    job,
                    (
                        "Confidence source not reliable; skipping "
                        "SYMBOL_ACTION_ACCEPTANCE_THRESHOLD enforcement."
                    ),
                )
            elif not accepted:
                self._append_log(
                    job,
                    (
                        "Confidence below SYMBOL_ACTION_ACCEPTANCE_THRESHOLD. "
                        "Forcing HOLD to keep execution safe."
                    ),
                )
                decision["action"] = "HOLD"
                decision["risk_notes"] = (
                    f"Calibrated confidence {calibrated_confidence:.2f} below "
                    f"threshold {threshold:.2f}."
                )

            if _is_env_flag_enabled("OUTPUTS_READABILITY_ENABLED", True):
                result["readable_summary"] = _summarize_result_with_llm(result)
            self._append_log(job, "Research complete.")
            logger.info(
                "Research workload completed session_id=%s selected_symbol=%s accepted=%s",
                job.session_id,
                symbol,
                accepted,
            )
            with self._lock:
                job.result = result
                job.status = "completed"
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                job.error = str(exc)
                job.status = "failed"
            logger.exception("Research workload failed session_id=%s error=%s", job.session_id, exc)
            self._append_log(job, f"Research failed: {exc}")

    def _discover_symbol(self, job: ResearchJob) -> tuple[str, float, list[dict[str, float | str]]]:
        self._append_log(job, "Scraping trending symbols and sentiment signals from web data.")
        logger.info("Research workload discovery started session_id=%s", job.session_id)
        auto_detection_symbol_number = self._auto_detection_symbol_number()
        if auto_detection_symbol_number > 0:
            symbols = fetch_trending_symbols(limit=auto_detection_symbol_number)
            logger.info(
                "Research workload auto-detection symbols fetched session_id=%s count=%s",
                job.session_id,
                len(symbols),
            )
            self._append_log(
                job,
                (
                    "AUTO_DETECTION_SYMBOL_NUMBER="
                    f"{auto_detection_symbol_number}; using auto-selected symbols."
                ),
            )
        else:
            symbols = job.candidate_symbols or []
            logger.info(
                "Research workload manual symbols session_id=%s count=%s",
                job.session_id,
                len(symbols),
            )
            if not symbols:
                raise ValueError(
                    (
                        "AUTO_DETECTION_SYMBOL_NUMBER disables auto mode. "
                        "Provide symbols separated by commas."
                    )
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
            logger.warning(
                "Research workload discovery produced no ranking session_id=%s", job.session_id
            )
            raise ValueError("No symbols available from scraping phase.")

        ranking.sort(key=lambda row: float(row["score"]), reverse=True)
        best = ranking[0]
        logger.info(
            "Research workload discovery completed session_id=%s top_symbol=%s top_score=%.4f",
            job.session_id,
            best["symbol"],
            float(best["score"]),
        )
        return str(best["symbol"]), float(best["score"]), ranking


research_sessions = ResearchSessionStore()
