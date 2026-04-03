from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from django.utils import timezone

from trading_bot.bot.conflict_engine import ConflictEngine
from trading_bot.bot.llm import LLMDecider

logger = logging.getLogger(__name__)


CAPABILITY_FLAGS = {
    "supportsFeeds": "ETORO_CAP_SUPPORTS_FEEDS",
    "supportsSocialAnalytics": "ETORO_CAP_SUPPORTS_SOCIAL_ANALYTICS",
    "supportsCuratedLists": "ETORO_CAP_SUPPORTS_CURATED_LISTS",
    "supportsWatchlists": "ETORO_CAP_SUPPORTS_WATCHLISTS",
    "supportsMarketMonitorStreaming": "ETORO_CAP_SUPPORTS_MARKET_MONITOR_STREAMING",
    "supportsAgentPortfolios": "ETORO_CAP_SUPPORTS_AGENT_PORTFOLIOS",
    "supportsDemoTrading": "ETORO_CAP_SUPPORTS_DEMO_TRADING",
    "supportsRealTrading": "ETORO_CAP_SUPPORTS_REAL_TRADING",
}

TIER_1_WEIGHT = 0.62
TIER_2_WEIGHT = 0.24
TIER_3_WEIGHT = 0.10
TIER_4_WEIGHT = 0.04


@dataclass
class SignalResult:
    score: float
    confidence: float
    freshness: float
    rationale: str
    raw_inputs: dict[str, Any]
    warnings: list[str]


class SignalProvider(Protocol):
    name: str

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        ...


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def detect_capabilities(adapter: Any) -> dict[str, bool]:
    registry: dict[str, bool] = {}
    for capability, env_name in CAPABILITY_FLAGS.items():
        registry[capability] = _env_flag(env_name, default=True)

    optional_methods = {
        "supportsFeeds": "getInstrumentFeedPosts",
        "supportsSocialAnalytics": "getSocialAnalytics",
        "supportsCuratedLists": "getCuratedLists",
        "supportsWatchlists": "getWatchlists",
        "supportsMarketMonitorStreaming": "streamMarketMonitor",
        "supportsAgentPortfolios": "getAgentPortfolioCompatibility",
    }
    for capability, method_name in optional_methods.items():
        method_is_available = hasattr(adapter, method_name)
        adapter_supports = getattr(adapter, "supported_capabilities", {})
        if isinstance(adapter_supports, dict) and capability in adapter_supports:
            method_is_available = method_is_available and bool(adapter_supports[capability])
        registry[capability] = registry[capability] and method_is_available

    if (
        not registry.get("supportsDemoTrading", True)
        and not registry.get("supportsRealTrading", True)
    ):
        registry["supportsDemoTrading"] = True

    return registry


class TechnicalSignalProvider:
    name = "technical"

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        prices = context.get("portfolio_history", [])[-20:]
        closes = [
            float(item.get("equity", 0))
            for item in prices
            if item.get("equity") is not None
        ]
        if len(closes) < 3:
            return SignalResult(
                0.5,
                0.35,
                0.5,
                "Not enough history",
                {"points": len(closes)},
                ["limited_history"],
            )

        momentum = (closes[-1] - closes[0]) / max(abs(closes[0]), 1)
        slope = (closes[-1] - closes[-4]) / max(abs(closes[-4]), 1) if len(closes) > 4 else momentum
        score = max(0.0, min(1.0, 0.5 + (momentum * 2.0) + (slope * 1.2)))
        return SignalResult(
            score=score,
            confidence=0.7,
            freshness=1.0,
            rationale=f"Momentum {momentum:.3f}, short slope {slope:.3f}",
            raw_inputs={"momentum": momentum, "slope": slope},
            warnings=[],
        )


class PortfolioRiskProvider:
    name = "portfolio_risk"

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        positions = context.get("positions", [])
        account = context.get("account", {})
        total_mv = sum(abs(float(item.get("marketValue", 0))) for item in positions) or 1.0
        symbol_mv = sum(
            abs(float(item.get("marketValue", 0)))
            for item in positions
            if str(item.get("symbol", "")).upper() == symbol
        )
        concentration = symbol_mv / total_mv
        drawdown = abs(float(account.get("drawdown", 0) or 0))
        risk_penalty = min(1.0, concentration + drawdown)
        score = max(0.0, 1.0 - risk_penalty)
        warnings = ["risk_veto"] if score < 0.2 else []
        return SignalResult(
            score=score,
            confidence=0.9,
            freshness=1.0,
            rationale=f"Concentration {concentration:.2f}, drawdown {drawdown:.2f}",
            raw_inputs={"concentration": concentration, "drawdown": drawdown},
            warnings=warnings,
        )


class MarketRegimeProvider:
    name = "market_regime"

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        monitor = context.get("market_monitor", {})
        alerts = monitor.get("alerts", []) if isinstance(monitor, dict) else []
        risk_alerts = [
            item
            for item in alerts
            if "risk" in str(item).lower() or "vol" in str(item).lower()
        ]
        score = 0.4 if risk_alerts else 0.65
        freshness = 1.0 if monitor else 0.5
        warnings = ["volatile_regime"] if risk_alerts else []
        return SignalResult(
            score=score,
            confidence=0.6,
            freshness=freshness,
            rationale=f"Regime alerts={len(risk_alerts)}",
            raw_inputs={"alerts": alerts[:5]},
            warnings=warnings,
        )


class CryptoMicrostructureProvider:
    name = "microstructure"

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        order_book = context.get("order_book", {}) or {}
        spread_bps = float(order_book.get("spread_bps", 12.0) or 12.0)
        imbalance = float(order_book.get("imbalance", 0.0) or 0.0)
        freshness = float(order_book.get("freshness", 0.6) or 0.6)
        base = max(0.0, min(1.0, 1 - (spread_bps / 45)))
        score = max(0.0, min(1.0, (base * 0.75) + (max(imbalance, 0.0) * 0.25)))
        return SignalResult(
            score=score,
            confidence=0.78,
            freshness=freshness,
            rationale=f"Spread bps={spread_bps:.2f}, bid/ask imbalance={imbalance:.2f}",
            raw_inputs={"spread_bps": spread_bps, "imbalance": imbalance},
            warnings=["microstructure_unavailable"] if not order_book else [],
        )


class FeedSentimentProvider:
    name = "social_sentiment"

    def __init__(self, openai_api_key: str | None) -> None:
        self.openai_api_key = openai_api_key

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        adapter = context.get("adapter")
        capabilities = context.get("capabilities", {})
        if not capabilities.get("supportsFeeds"):
            return SignalResult(0.5, 0.2, 0.0, "Feeds not supported", {}, ["feeds_unavailable"])

        posts: list[dict[str, Any]] = []
        try:
            posts = adapter.getInstrumentFeedPosts(symbol=symbol, limit=25)
            user_posts = adapter.getUserFeedPosts(limit=25)
            posts.extend(user_posts)
        except Exception as exc:
            logger.error("feed sentiment collection failed: %s", exc)
            return SignalResult(0.5, 0.2, 0.0, "Feed fetch failed", {}, ["feed_fetch_failed"])

        sample = [str(item.get("text", ""))[:240] for item in posts if item.get("text")]
        sample = sample[:12]
        summary = {
            "sentimentDirection": "NEUTRAL",
            "sentimentStrength": 0.0,
            "crowdingRisk": 0.0,
            "narrativeVelocity": 0.0,
            "confidencePenaltyOrBoost": 0.0,
            "explanation": "No strong feed signal.",
        }
        if self.openai_api_key and sample:
            prompt = (
                "Return JSON with keys sentimentDirection, sentimentStrength, crowdingRisk,"
                " narrativeVelocity, confidencePenaltyOrBoost, explanation."
                f" Symbol={symbol}. Posts={json.dumps(sample)}"
            )
            try:
                decider = LLMDecider(
                    provider="openai",
                    model="gpt-4o-mini",
                    api_key=self.openai_api_key,
                )
                text = decider.summarize_json({"prompt": prompt})
                parsed = json.loads(text)
                summary.update(parsed)
            except Exception as exc:
                logger.error("feed sentiment ai synthesis failed: %s", exc)

        strength = float(summary.get("sentimentStrength", 0) or 0)
        score = max(0.0, min(1.0, 0.5 + strength * 0.25))
        return SignalResult(
            score=score,
            confidence=0.35,
            freshness=1.0,
            rationale=str(summary.get("explanation", "")),
            raw_inputs={"posts_count": len(posts), "posts": posts[:20], "summary": summary},
            warnings=["sentiment_context_only"],
        )


class WatchlistAttentionProvider:
    name = "watchlist_interest"

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        adapter = context.get("adapter")
        capabilities = context.get("capabilities", {})
        if not capabilities.get("supportsWatchlists"):
            return SignalResult(
                0.5,
                0.2,
                0.0,
                "Watchlists unavailable",
                {},
                ["watchlist_unavailable"],
            )

        try:
            lists = adapter.getWatchlists()
        except Exception as exc:
            logger.error("watchlist ingestion failed: %s", exc)
            return SignalResult(
                0.5,
                0.2,
                0.0,
                "Watchlists fetch failed",
                {},
                ["watchlist_fetch_failed"],
            )

        symbol_count = 0
        total = 0
        for watchlist in lists:
            entries = watchlist.get("symbols", [])
            total += len(entries)
            symbol_count += sum(1 for item in entries if str(item).upper() == symbol)
        score = max(0.0, min(1.0, 0.5 + (symbol_count / max(total, 1))))
        return SignalResult(
            score=score,
            confidence=0.45,
            freshness=1.0,
            rationale=f"symbol presence {symbol_count}/{max(total,1)}",
            raw_inputs={"watchlists": len(lists), "presence": symbol_count},
            warnings=[],
        )


class CuratedListProvider:
    name = "curated_interest"

    def collect(self, *, symbol: str, context: dict[str, Any]) -> SignalResult:
        adapter = context.get("adapter")
        capabilities = context.get("capabilities", {})
        if not capabilities.get("supportsCuratedLists"):
            return SignalResult(
                0.5,
                0.2,
                0.0,
                "Curated lists unavailable",
                {},
                ["curated_unavailable"],
            )
        try:
            curated = adapter.getCuratedLists()
        except Exception as exc:
            logger.error("curated list ingestion failed: %s", exc)
            return SignalResult(0.5, 0.2, 0.0, "Curated fetch failed", {}, ["curated_fetch_failed"])

        in_lists = 0
        total = 0
        for entry in curated:
            symbols = entry.get("symbols", [])
            total += len(symbols)
            in_lists += sum(1 for item in symbols if str(item).upper() == symbol)
        score = max(0.0, min(1.0, 0.5 + (in_lists / max(total, 1))))
        return SignalResult(
            score=score,
            confidence=0.4,
            freshness=1.0,
            rationale=f"curated presence {in_lists}/{max(total,1)}",
            raw_inputs={"curatedLists": len(curated), "presence": in_lists},
            warnings=[],
        )


def _is_fresh(timestamp: datetime | None, stale_seconds: int = 90) -> bool:
    if not timestamp:
        return False
    age = (timezone.now() - timestamp).total_seconds()
    return age <= stale_seconds


def fuse_signals(
    results: dict[str, SignalResult],
    *,
    price_move_pct: float,
    fresh_market_data: bool,
) -> dict[str, Any]:
    conflict_engine = ConflictEngine()
    technical = results["technical"].score
    breakout_quality = technical
    microstructure = results.get("microstructure", SignalResult(0.5, 0.4, 0.5, "", {}, [])).score
    risk = results["portfolio_risk"].score
    exposure = risk
    drawdown_score = risk
    regime = results["market_regime"].score
    sentiment = results["social_sentiment"].score
    attention = (results["watchlist_interest"].score + results["curated_interest"].score) / 2
    llm_meta = 0.5

    tier_1 = (technical + breakout_quality + microstructure) / 3
    tier_2 = (risk + exposure + drawdown_score) / 3
    tier_3 = (regime + sentiment + attention) / 3
    proposal_score = (
        (tier_1 * TIER_1_WEIGHT)
        + (tier_2 * TIER_2_WEIGHT)
        + (tier_3 * TIER_3_WEIGHT)
        + (llm_meta * TIER_4_WEIGHT)
    )

    conflict = conflict_engine.evaluate(
        technical_score=technical,
        breakout_quality=breakout_quality,
        microstructure_score=microstructure,
        portfolio_risk_score=risk,
        exposure_score=exposure,
        drawdown_score=drawdown_score,
        sentiment_score=sentiment,
        attention_score=attention,
        fresh_market_data=fresh_market_data,
        price_move_pct=price_move_pct,
    )

    avg_conf = sum(item.confidence for item in results.values()) / max(len(results), 1)
    min_freshness = min(item.freshness for item in results.values())
    confidence_score = max(0.0, min(1.0, (avg_conf * 0.75) + (min_freshness * 0.25)))
    confidence_score = min(confidence_score, conflict.adjusted_confidence)
    if risk < 0.2:
        confidence_score = min(confidence_score, 0.35)

    if risk < 0.2:
        proposal_type = "REQUIRE_REVIEW"
    elif abs(price_move_pct) >= 1.5:
        proposal_type = "INVALIDATE_PENDING"
    elif proposal_score >= 0.7:
        proposal_type = "BUY_CANDIDATE"
    elif proposal_score >= 0.58:
        proposal_type = "WATCH"
    elif proposal_score <= 0.32:
        proposal_type = "SELL_CANDIDATE"
    else:
        proposal_type = "HOLD"

    approval_priority = "HIGH" if proposal_score > 0.72 and risk >= 0.4 else "NORMAL"
    if proposal_type in {"INVALIDATE_PENDING", "REQUIRE_REVIEW"}:
        approval_priority = "HIGH"

    return {
        "proposalScore": round(proposal_score, 4),
        "confidenceScore": round(confidence_score, 4),
        "approvalPriority": approval_priority,
        "proposalType": proposal_type,
        "conflictFlags": conflict.conflict_flags,
        "explanationSummary": (
            f"Tier1(tech/breakout/micro)={tier_1:.2f}, Tier2(risk)={tier_2:.2f}, "
            f"Tier3(attention/sentiment)={tier_3:.2f}. {conflict.explanation_summary}"
        ),
        "signalBreakdown": {
            "technical": round(technical, 4),
            "risk": round(risk, 4),
            "sentiment": round(sentiment, 4),
            "attention": round(attention, 4),
            "microstructure": round(microstructure, 4),
        },
    }
