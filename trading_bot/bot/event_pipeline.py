from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from django.utils import timezone

from trading_bot.bot.llm import LLMDecider

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_id: str
    event_type: str
    symbol: str
    ts: datetime
    payload: dict[str, Any]


class EventBus:
    def __init__(self, debounce_ms: int = 100, workers: int = 4) -> None:
        self._debounce_ms = debounce_ms
        self._handlers: dict[str, list[Callable[[Event], None]]] = {}
        self._processed: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._queue: dict[tuple[str, str], Event] = {}
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._metrics = {"events_total": 0, "events_dispatched": 0}

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, symbol: str, payload: dict[str, Any]) -> str:
        event_id = payload.get("eventId") or str(uuid.uuid4())
        event = Event(
            event_id=event_id,
            event_type=event_type,
            symbol=symbol.upper(),
            ts=timezone.now(),
            payload=payload,
        )
        with self._lock:
            self._metrics["events_total"] += 1
            self._queue[(event_type, event.symbol)] = event
        threading.Timer(
            self._debounce_ms / 1000.0,
            self._flush,
            args=(event_type, event.symbol),
        ).start()
        return event_id

    def _flush(self, event_type: str, symbol: str) -> None:
        with self._lock:
            event = self._queue.pop((event_type, symbol), None)
        if not event:
            return
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            self._executor.submit(self._dispatch, handler, event)

    def _dispatch(self, handler: Callable[[Event], None], event: Event) -> None:
        key = f"{event.event_type}:{id(handler)}"
        processed = self._processed.setdefault(key, set())
        if event.event_id in processed:
            return
        processed.add(event.event_id)
        started = time.perf_counter()
        try:
            handler(event)
            latency = int((time.perf_counter() - started) * 1000)
            self._metrics["events_dispatched"] += 1
            logger.info(
                "event_bus dispatch_ok symbol=%s eventId=%s latencyMs=%s snapshotHash=%s",
                event.symbol,
                event.event_id,
                latency,
                event.payload.get("snapshotHash", ""),
            )
        except Exception as exc:
            logger.error(
                "event_bus dispatch_failed symbol=%s eventId=%s error=%s",
                event.symbol,
                event.event_id,
                exc,
            )

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._metrics)


class FeatureSnapshotBuilder:
    def __init__(self, stale_after_ms: int = 15000) -> None:
        self._stale_after_ms = stale_after_ms

    def build(
        self,
        symbol: str,
        event_id: str,
        provider_results: dict[str, Any],
        fused: dict[str, Any],
        account: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        technical = provider_results["technical"].raw_inputs
        risk = provider_results["portfolio_risk"].raw_inputs
        regime = provider_results["market_regime"].raw_inputs
        social = provider_results["social_sentiment"].raw_inputs
        watchlist = provider_results["watchlist_interest"].raw_inputs
        curated = provider_results["curated_interest"].raw_inputs
        pos = next((x for x in positions if str(x.get("symbol", "")).upper() == symbol), {})

        short_vol = abs(float(fused.get("priceMovePct", 0.0)))
        ts = timezone.now()
        data_age = int((timezone.now() - ts).total_seconds() * 1000)

        snapshot = {
            "symbol": symbol,
            "ts": ts.isoformat(),
            "technical": {
                "trend": technical.get("trend", "neutral"),
                "breakout": bool(technical.get("breakout", False)),
                "overextended": bool(technical.get("overextended", False)),
                "maAlignment": technical.get("ma_alignment", "mixed"),
            },
            "microstructure": {
                "localLow": technical.get("local_low", technical.get("support", 0)),
                "localHigh": technical.get("local_high", technical.get("resistance", 0)),
                "range": technical.get("range", 0),
                "spread": technical.get("spread", 0),
                "liquidity": technical.get("liquidity", 0),
                "shortTermVol": short_vol,
            },
            "portfolio": {
                "exposure": abs(float(pos.get("marketValue", 0))),
                "positionSize": abs(float(pos.get("quantity", 0))),
                "drawdown": float(account.get("drawdown", 0.0) or 0.0),
                "riskState": risk.get("risk_state", "normal"),
            },
            "marketRegime": {
                "regime": regime.get("regime", "unknown"),
                "volRegime": regime.get("vol_regime", "unknown"),
            },
            "context": {
                "attentionScore": watchlist.get("attention_score", 0),
                "crowdingScore": curated.get("crowding_score", 0),
                "narrativeVelocity": social.get("narrative_velocity", 0),
            },
            "freshness": {
                "dataAgeMs": data_age,
                "isStale": data_age > self._stale_after_ms,
            },
            "eventId": event_id,
        }
        logger.info(
            "snapshot_build symbol=%s eventId=%s latencyMs=0 snapshotHash=%s",
            symbol,
            event_id,
            self.snapshot_hash(snapshot),
        )
        return snapshot

    @staticmethod
    def snapshot_hash(snapshot: dict[str, Any]) -> str:
        src = json.dumps(snapshot, sort_keys=True)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()


class LLMReasoningEngine:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._decider = LLMDecider(provider="openai", model=model, api_key=api_key)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cooldown_by_symbol: dict[str, float] = {}
        self._cooldown_s = float(os.getenv("LLM_SYMBOL_COOLDOWN_SEC", "4"))
        self._eps = float(os.getenv("LLM_PRICE_NOISE_EPSILON", "0.05"))

    def interpret(
        self,
        symbol: str,
        snapshot: dict[str, Any],
        deterministic_score: float,
        prior: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        s_hash = FeatureSnapshotBuilder.snapshot_hash(snapshot)
        if s_hash in self._cache:
            return self._cache[s_hash]

        if (
            abs(float(snapshot["microstructure"].get("shortTermVol", 0.0))) < self._eps
            and not force
        ):
            return self._neutral("noise_skip")

        last = self._cooldown_by_symbol.get(symbol, 0.0)
        if time.time() - last < self._cooldown_s and not force:
            return self._neutral("cooldown_skip")

        payload = {
            "snapshot": snapshot,
            "deterministicScore": deterministic_score,
            "priorProposal": prior or {},
            "schema": {
                "interpretation": "string",
                "confidence_adjustment": "number [-0.3, 0.3]",
                "risk_note": "string",
                "action_bias": "favor|neutral|avoid",
                "conflicts": ["string"],
                "uncertainty": "low|medium|high",
            },
            "rules": [
                "never execute trades",
                "never bypass approval",
                "never override risk constraints",
                "return strict json only",
            ],
        }
        prompt = json.dumps(payload, ensure_ascii=False)

        for attempt in range(2):
            try:
                started = time.perf_counter()
                raw = self._decider.decide(prompt)
                latency = int((time.perf_counter() - started) * 1000)
                result = self._validate(raw)
                self._cache[s_hash] = result
                self._cooldown_by_symbol[symbol] = time.time()
                logger.info(
                    "llm_call symbol=%s latencyMs=%s snapshotHash=%s",
                    symbol,
                    latency,
                    s_hash,
                )
                return result
            except Exception as exc:
                logger.error(
                    "llm_timeout/fallback symbol=%s attempt=%s error=%s",
                    symbol,
                    attempt,
                    exc,
                )
                if attempt == 1:
                    return self._neutral("llm_failure")

        return self._neutral("llm_fallback")

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        result = {
            "interpretation": str(raw.get("reasoning", raw.get("interpretation", ""))).strip(),
            "confidence_adjustment": float(raw.get("confidence_adjustment", 0.0)),
            "risk_note": str(raw.get("risk_notes", raw.get("risk_note", ""))).strip(),
            "action_bias": str(raw.get("action_bias", "neutral")).strip(),
            "conflicts": raw.get("conflicts", []),
            "uncertainty": str(raw.get("uncertainty", "medium")).strip(),
        }
        if not isinstance(result["conflicts"], list):
            result["conflicts"] = []
        if result["action_bias"] not in {"favor", "neutral", "avoid"}:
            result["action_bias"] = "neutral"
        if result["uncertainty"] not in {"low", "medium", "high"}:
            result["uncertainty"] = "medium"
        result["confidence_adjustment"] = max(-0.3, min(0.3, result["confidence_adjustment"]))
        return result

    @staticmethod
    def _neutral(reason: str) -> dict[str, Any]:
        return {
            "interpretation": "Neutral fallback reasoning.",
            "confidence_adjustment": 0.0,
            "risk_note": f"Neutral adjustment due to {reason}.",
            "action_bias": "neutral",
            "conflicts": [],
            "uncertainty": "medium",
        }


class FusionEngine:
    def combine(
        self,
        base_score: float,
        llm_result: dict[str, Any],
        risk_veto: bool,
        is_stale: bool,
    ) -> dict[str, Any]:
        final_score = base_score * (1 + float(llm_result.get("confidence_adjustment", 0.0)))
        uncertainty = llm_result.get("uncertainty", "medium")
        if uncertainty == "high":
            final_score *= 0.7
        if risk_veto:
            final_score = 0.0
        if is_stale:
            final_score = 0.0

        confidence = max(0.0, min(1.0, final_score))
        return {
            "proposalScore": final_score,
            "confidence": confidence,
            "flags": {
                "conflicts": llm_result.get("conflicts", []),
                "uncertainty": uncertainty,
            },
            "explanation": llm_result.get("interpretation", ""),
        }
