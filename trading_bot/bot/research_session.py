from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from trading_bot.bot.data_sources import (
    compute_technical_indicators,
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
    except Exception as exc:
        logger.error("Failed to summarize result with llm error=%s", exc)
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
    FEATURE_KEYS = (
        "sentiment",
        "momentum_5d",
        "momentum_20d",
        "volume_stability",
        "short_drawdown",
        "tech_trend",
        "tech_rsi",
        "tech_macd",
        "tech_bollinger",
    )
    RISK_PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
        "low": {
            "sentiment": 0.8,
            "momentum_5d": 0.4,
            "momentum_20d": 1.2,
            "volume_stability": 1.1,
            "short_drawdown": 1.3,
            "tech_trend": 1.0,
            "tech_rsi": 1.0,
            "tech_macd": 0.8,
            "tech_bollinger": 0.9,
        },
        "medium": {
            "sentiment": 1.0,
            "momentum_5d": 0.8,
            "momentum_20d": 1.0,
            "volume_stability": 0.9,
            "short_drawdown": 1.0,
            "tech_trend": 1.0,
            "tech_rsi": 0.9,
            "tech_macd": 1.0,
            "tech_bollinger": 0.9,
        },
        "high": {
            "sentiment": 1.2,
            "momentum_5d": 1.3,
            "momentum_20d": 0.9,
            "volume_stability": 0.6,
            "short_drawdown": 0.5,
            "tech_trend": 1.1,
            "tech_rsi": 0.7,
            "tech_macd": 1.2,
            "tech_bollinger": 0.8,
        },
    }

    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()

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

    @staticmethod
    def _cache_ttl_seconds() -> int:
        raw_value = os.getenv("DISCOVERY_CACHE_TTL_SECONDS", "60").strip()
        try:
            ttl = int(raw_value)
        except ValueError:
            logger.error(
                "Invalid DISCOVERY_CACHE_TTL_SECONDS=%s, fallback to 60 seconds.",
                raw_value,
            )
            return 60
        return min(max(ttl, 30), 120)

    @staticmethod
    def _symbol_worker_timeout_seconds() -> float:
        raw_value = os.getenv("DISCOVERY_SYMBOL_TIMEOUT_SECONDS", "6").strip()
        try:
            timeout = float(raw_value)
        except ValueError:
            logger.error(
                "Invalid DISCOVERY_SYMBOL_TIMEOUT_SECONDS=%s, fallback to 6 seconds.",
                raw_value,
            )
            return 6.0
        return max(timeout, 1.0)

    @staticmethod
    def _symbol_worker_pool_size(symbol_count: int) -> int:
        configured = os.getenv("DISCOVERY_WORKER_THREADS", "").strip()
        if configured:
            try:
                return max(min(int(configured), max(symbol_count, 1)), 1)
            except ValueError:
                logger.error(
                    "Invalid DISCOVERY_WORKER_THREADS=%s, using adaptive worker size.",
                    configured,
                )
        return max(min(symbol_count, 8), 1)

    def _cache_get_or_set(self, key: str, loader: Callable[[], Any], ttl_seconds: int) -> Any:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return cached[1]

        value = loader()
        with self._cache_lock:
            self._cache[key] = (now + ttl_seconds, value)
        return value

    def _cached_fetch_trending_symbols(self, limit: int) -> list[str]:
        ttl_seconds = self._cache_ttl_seconds()
        cache_key = f"trending:{limit}"
        return self._cache_get_or_set(
            cache_key,
            loader=lambda: fetch_trending_symbols(limit=limit),
            ttl_seconds=ttl_seconds,
        )

    def _cached_market_snapshot(self, symbol: str) -> Any:
        ttl_seconds = self._cache_ttl_seconds()
        cache_key = f"snapshot:{symbol.upper()}"
        return self._cache_get_or_set(
            cache_key,
            loader=lambda: get_market_snapshot(symbol),
            ttl_seconds=ttl_seconds,
        )

    def _cached_sentiment_scores(self, symbol: str, days: tuple[date, ...]) -> dict[date, float]:
        ttl_seconds = self._cache_ttl_seconds()
        cache_key = f"sentiment:{symbol.upper()}:{','.join(str(day) for day in days)}"
        return self._cache_get_or_set(
            cache_key,
            loader=lambda: fetch_x_sentiment_scores(symbol=symbol, days=list(days)),
            ttl_seconds=ttl_seconds,
        )

    def _analyze_symbol(
        self,
        job: ResearchJob,
        symbol: str,
        last_days: tuple[date, ...],
    ) -> tuple[dict[str, Any] | None, float]:
        started_at = time.perf_counter()
        try:
            snapshot = self._cached_market_snapshot(symbol)
        except ValueError as exc:
            logger.error(
                "Skipping symbol due to insufficient market history "
                "session_id=%s symbol=%s error=%s",
                job.session_id,
                symbol,
                exc,
            )
            self._append_log(job, f"Skipping {symbol}: insufficient market history.")
            return None, (time.perf_counter() - started_at) * 1000
        except Exception as exc:
            logger.error(
                "Market snapshot failed session_id=%s symbol=%s error=%s",
                job.session_id,
                symbol,
                exc,
            )
            self._append_log(job, f"Skipping {symbol}: market snapshot unavailable.")
            return None, (time.perf_counter() - started_at) * 1000

        try:
            sentiment_map = self._cached_sentiment_scores(symbol=symbol, days=last_days)
        except Exception as exc:
            logger.error(
                "Sentiment fetch failed session_id=%s symbol=%s error=%s",
                job.session_id,
                symbol,
                exc,
            )
            self._append_log(job, f"Skipping {symbol}: sentiment source unavailable.")
            return None, (time.perf_counter() - started_at) * 1000

        min_history_days = self._min_history_days()
        try:
            technicals = compute_technical_indicators(symbol=symbol, lookback_days=min_history_days)
        except ValueError as exc:
            logger.error(
                "Skipping symbol due to insufficient technical history session_id=%s symbol=%s "
                "required_history_days=%s error=%s",
                job.session_id,
                symbol,
                min_history_days,
                exc,
            )
            self._append_log(job, f"Skipping {symbol}: insufficient technical history.")
            return None, (time.perf_counter() - started_at) * 1000
        except Exception as exc:
            logger.error(
                "Technical indicators failed session_id=%s symbol=%s error=%s",
                job.session_id,
                symbol,
                exc,
            )
            self._append_log(job, f"Skipping {symbol}: technical indicators unavailable.")
            return None, (time.perf_counter() - started_at) * 1000

        pct_change_5d = float(getattr(snapshot, "pct_change_5d", 0.0))
        pct_change_20d = float(getattr(snapshot, "pct_change_20d", pct_change_5d))
        latest_close = float(getattr(snapshot, "latest_close", 1.0))
        sentiment = sum(sentiment_map.values()) / max(len(sentiment_map), 1)
        relative_volume = snapshot.latest_volume / max(snapshot.avg_volume_20d, 1.0)
        volume_stability = -abs(relative_volume - 1.0)
        short_drawdown = -max(-pct_change_5d, 0.0)
        tech_trend = (technicals.sma_20 - technicals.sma_50) / max(abs(technicals.sma_50), 1e-6)
        tech_rsi = 1.0 - (abs(technicals.rsi_14 - 50.0) / 50.0)
        tech_macd = technicals.macd - technicals.macd_signal
        band_width = max(technicals.bollinger_upper - technicals.bollinger_lower, 1e-6)
        tech_bollinger = (((latest_close - technicals.bollinger_lower) / band_width) * 2.0) - 1.0

        analyzed = {
            "symbol": snapshot.symbol,
            "sentiment": round(sentiment, 6),
            "pct_change_5d": round(pct_change_5d, 4),
            "pct_change_20d": round(pct_change_20d, 4),
            "relative_volume": round(relative_volume, 6),
            "avg_volume_20d": round(snapshot.avg_volume_20d, 2),
            "liquidity_proxy": round(latest_close * snapshot.avg_volume_20d, 2),
            "volatility_proxy": round(abs(pct_change_5d), 4),
            "features": {
                "sentiment": sentiment,
                "momentum_5d": pct_change_5d,
                "momentum_20d": pct_change_20d,
                "volume_stability": volume_stability,
                "short_drawdown": short_drawdown,
                "tech_trend": tech_trend,
                "tech_rsi": tech_rsi,
                "tech_macd": tech_macd,
                "tech_bollinger": tech_bollinger,
            },
        }
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self._append_log(
            job,
            (
                f"Analyzed {snapshot.symbol}: sentiment={sentiment:.3f}, "
                f"5d_change={pct_change_5d:.2f}, "
                f"20d_change={pct_change_20d:.2f}, "
                f"volume_stability={volume_stability:.3f}, "
                f"per_symbol_ms={elapsed_ms:.1f}."
            ),
        )
        return analyzed, elapsed_ms

    @staticmethod
    def _min_history_days() -> int:
        raw_value = os.getenv("DISCOVERY_MIN_HISTORY_DAYS", "180").strip()
        try:
            return max(int(raw_value), 60)
        except ValueError:
            logger.error("Invalid DISCOVERY_MIN_HISTORY_DAYS=%s, fallback to 180.", raw_value)
            return 180

    @staticmethod
    def _min_liquidity_proxy() -> float:
        raw_value = os.getenv("DISCOVERY_MIN_LIQUIDITY_PROXY", "0").strip()
        try:
            return max(float(raw_value), 0.0)
        except ValueError:
            logger.error(
                "Invalid DISCOVERY_MIN_LIQUIDITY_PROXY=%s, fallback to 0.",
                raw_value,
            )
            return 0.0

    @staticmethod
    def _max_volatility_proxy() -> float:
        raw_value = os.getenv("DISCOVERY_MAX_VOLATILITY_5D_PCT", "12").strip()
        try:
            return max(float(raw_value), 0.1)
        except ValueError:
            logger.error("Invalid DISCOVERY_MAX_VOLATILITY_5D_PCT=%s, fallback to 12.", raw_value)
            return 12.0

    @staticmethod
    def _normalization_mode() -> str:
        raw_value = os.getenv("DISCOVERY_FEATURE_NORMALIZATION", "zscore").strip().lower()
        if raw_value in {"zscore", "minmax"}:
            return raw_value
        logger.error(
            "Invalid DISCOVERY_FEATURE_NORMALIZATION=%s, fallback to zscore.",
            raw_value,
        )
        return "zscore"

    def _resolve_feature_weights(self, risk_profile: str) -> dict[str, float]:
        profile = (risk_profile or "medium").strip().lower()
        if profile not in self.RISK_PROFILE_WEIGHTS:
            logger.error(
                "Invalid risk profile=%s for discovery weights. Fallback to medium.",
                risk_profile,
            )
            profile = "medium"
        weights = dict(self.RISK_PROFILE_WEIGHTS[profile])
        for key in self.FEATURE_KEYS:
            env_name = f"DISCOVERY_WEIGHT_{profile.upper()}_{key.upper()}"
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            try:
                weights[key] = float(raw_value)
            except ValueError:
                logger.error("Invalid %s=%s, keeping weight=%s.", env_name, raw_value, weights[key])
        return weights

    def _normalize_feature_map(self, ranking: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        normalization_mode = self._normalization_mode()
        normalized: dict[str, dict[str, float]] = {
            str(row["symbol"]): {} for row in ranking
        }
        for key in self.FEATURE_KEYS:
            values = [float(row["features"][key]) for row in ranking]
            if normalization_mode == "minmax":
                min_value = min(values)
                max_value = max(values)
                spread = max_value - min_value
                for row in ranking:
                    symbol = str(row["symbol"])
                    current = float(row["features"][key])
                    normalized[symbol][key] = 0.0 if spread == 0 else (current - min_value) / spread
                continue

            mean_value = sum(values) / max(len(values), 1)
            variance = sum((value - mean_value) ** 2 for value in values) / max(len(values), 1)
            stddev = variance**0.5
            for row in ranking:
                symbol = str(row["symbol"])
                current = float(row["features"][key])
                normalized[symbol][key] = 0.0 if stddev == 0 else (current - mean_value) / stddev
        return normalized

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
            anomaly_notes: list[str] = []
            selected_symbol = symbol
            selected_score = score
            result: dict[str, Any] | None = None

            for candidate in ranking:
                candidate_symbol = str(candidate["symbol"])
                candidate_score = float(candidate["score"])
                self._append_log(
                    job,
                    (
                        f"Selected symbol {candidate_symbol} with discovery score "
                        f"{candidate_score:.2f}. Running suggestion model..."
                    ),
                )

                logger.info(
                    "Research workload generating suggestion session_id=%s symbol=%s",
                    job.session_id,
                    candidate_symbol,
                )
                try:
                    result = _suggestion_with_retry(
                        symbol=candidate_symbol,
                        risk_profile=job.risk_profile,
                        log_callback=lambda message: self._append_log(job, message),
                    )
                    selected_symbol = candidate_symbol
                    selected_score = candidate_score
                    break
                except Exception as exc:
                    if not _is_rate_limited_exception(exc):
                        raise
                    note = (
                        f"Suggestion model anomaly on {candidate_symbol}: {exc}. "
                        "Continuing with next candidate."
                    )
                    anomaly_notes.append(note)
                    self._append_log(job, note)

            if result is None:
                raise RuntimeError(
                    "Suggestion model unavailable for all candidate symbols due to rate limiting."
                )

            logger.info(
                "Research workload suggestion generated session_id=%s symbol=%s decision=%s",
                job.session_id,
                selected_symbol,
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
                "selected_symbol": selected_symbol,
                "score": round(selected_score, 4),
                "ranking": ranking,
                "acceptance_threshold": threshold,
                "accepted": accepted,
                "confidence_source": confidence_source,
                "confidence_reliable": has_reliable_confidence,
            }
            if anomaly_notes:
                result["discovery"]["anomalies"] = anomaly_notes
                result["discovery"]["warning"] = (
                    "One or more symbols showed suggestion model anomalies; "
                    "fallback symbol was used."
                )
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
                selected_symbol,
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

    def _discover_symbol(self, job: ResearchJob) -> tuple[str, float, list[dict[str, Any]]]:
        discovery_started_at = time.perf_counter()
        self._append_log(job, "Scraping trending symbols and sentiment signals from web data.")
        logger.info("Research workload discovery started session_id=%s", job.session_id)
        auto_detection_symbol_number = self._auto_detection_symbol_number()
        manual_symbols = [symbol.upper() for symbol in (job.candidate_symbols or []) if symbol]

        if manual_symbols:
            symbols = manual_symbols
            logger.info(
                "Research workload explicit symbols provided session_id=%s count=%s",
                job.session_id,
                len(symbols),
            )
            self._append_log(job, "Using symbols provided by user input.")
        elif auto_detection_symbol_number > 0:
            symbols = self._cached_fetch_trending_symbols(limit=auto_detection_symbol_number)
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
        last_days = tuple(today - timedelta(days=offset) for offset in range(5))
        worker_timeout_seconds = self._symbol_worker_timeout_seconds()
        max_workers = self._symbol_worker_pool_size(len(symbols))

        ranking: list[dict[str, Any]] = []
        per_symbol_ms: list[float] = []
        success_count = 0
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="discovery",
        ) as executor:
            futures = {
                executor.submit(self._analyze_symbol, job, symbol, last_days): symbol
                for symbol in symbols
            }
            for future, symbol in futures.items():
                try:
                    analyzed, elapsed_ms = future.result(timeout=worker_timeout_seconds)
                    per_symbol_ms.append(elapsed_ms)
                    if analyzed is None:
                        continue
                    success_count += 1
                    ranking.append(analyzed)
                except TimeoutError:
                    logger.error(
                        "Discovery timeout session_id=%s symbol=%s timeout_seconds=%.2f",
                        job.session_id,
                        symbol,
                        worker_timeout_seconds,
                    )
                    self._append_log(
                        job,
                        (
                            f"Skipping {symbol}: discovery timed out after "
                            f"{worker_timeout_seconds:.1f}s."
                        ),
                    )
                except Exception as exc:
                    logger.error(
                        "Discovery worker failed session_id=%s symbol=%s error=%s",
                        job.session_id,
                        symbol,
                        exc,
                    )
                    self._append_log(job, f"Skipping {symbol}: unexpected discovery failure.")

        min_liquidity_proxy = self._min_liquidity_proxy()
        max_volatility_proxy = self._max_volatility_proxy()
        filtered_out = 0
        filtered_ranking: list[dict[str, Any]] = []
        for row in ranking:
            liquidity_proxy = float(row["liquidity_proxy"])
            volatility_proxy = float(row["volatility_proxy"])
            if liquidity_proxy < min_liquidity_proxy:
                filtered_out += 1
                self._append_log(
                    job,
                    (
                        f"Filtered {row['symbol']}: liquidity {liquidity_proxy:.2f} "
                        f"< min {min_liquidity_proxy:.2f}."
                    ),
                )
                continue
            if volatility_proxy > max_volatility_proxy:
                filtered_out += 1
                self._append_log(
                    job,
                    (
                        f"Filtered {row['symbol']}: volatility {volatility_proxy:.2f}% "
                        f"> max {max_volatility_proxy:.2f}%."
                    ),
                )
                continue
            filtered_ranking.append(row)
        ranking = filtered_ranking

        if not ranking:
            logger.warning(
                "Research workload discovery produced no ranking session_id=%s", job.session_id
            )
            logger.error(
                "Discovery failed session_id=%s reason=no_symbols_available symbols_count=%s",
                job.session_id,
                len(symbols),
            )
            raise ValueError("No symbols available from scraping phase.")

        normalized_feature_map = self._normalize_feature_map(ranking)
        weights = self._resolve_feature_weights(job.risk_profile)
        for row in ranking:
            symbol = str(row["symbol"])
            normalized = normalized_feature_map[symbol]
            score_breakdown: dict[str, float] = {}
            total_score = 0.0
            for key in self.FEATURE_KEYS:
                contribution = normalized[key] * weights[key]
                score_breakdown[key] = round(contribution, 6)
                total_score += contribution
            row["score_breakdown"] = score_breakdown
            row["feature_values"] = {
                key: round(float(row["features"][key]), 6) for key in self.FEATURE_KEYS
            }
            row["normalized_features"] = {
                key: round(normalized[key], 6) for key in self.FEATURE_KEYS
            }
            row["applied_weights"] = {key: round(weights[key], 6) for key in self.FEATURE_KEYS}
            row["score"] = round(total_score, 6)
            del row["features"]

        ranking.sort(key=lambda row: float(row["score"]), reverse=True)
        best = ranking[0]
        discovery_ms = (time.perf_counter() - discovery_started_at) * 1000
        success_ratio = success_count / max(len(symbols), 1)
        avg_per_symbol_ms = sum(per_symbol_ms) / max(len(per_symbol_ms), 1)
        logger.info(
            "Research workload discovery completed session_id=%s top_symbol=%s top_score=%.4f "
            "discovery_ms=%.1f per_symbol_ms=%.1f success_ratio=%.2f filtered_out=%s",
            job.session_id,
            best["symbol"],
            float(best["score"]),
            discovery_ms,
            avg_per_symbol_ms,
            success_ratio,
            filtered_out,
        )
        return str(best["symbol"]), float(best["score"]), ranking


research_sessions = ResearchSessionStore()
