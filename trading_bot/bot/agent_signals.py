from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from django.utils import timezone

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

DEFAULT_WEIGHTS = {
    "technical_score": 0.60,
    "portfolio_risk_score": 0.20,
    "market_regime_score": 0.10,
    "social_sentiment_score": 0.05,
    "watchlist_interest_score": 0.05,
}


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
        registry[capability] = registry[capability] and hasattr(adapter, method_name)

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
            raw_inputs={"posts": len(posts), "summary": summary},
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
    technical = results["technical"].score
    risk = results["portfolio_risk"].score
    regime = results["market_regime"].score
    sentiment = results["social_sentiment"].score
    attention = (results["watchlist_interest"].score + results["curated_interest"].score) / 2

    proposal_score = (
        technical * DEFAULT_WEIGHTS["technical_score"]
        + risk * DEFAULT_WEIGHTS["portfolio_risk_score"]
        + regime * DEFAULT_WEIGHTS["market_regime_score"]
        + sentiment * DEFAULT_WEIGHTS["social_sentiment_score"]
        + attention * DEFAULT_WEIGHTS["watchlist_interest_score"]
    )

    conflict_flags: list[str] = []
    if technical >= 0.65 and sentiment <= 0.4:
        conflict_flags.append("technical_vs_sentiment_conflict")
    if technical >= 0.65 and risk < 0.3:
        conflict_flags.append("opportunity_vs_portfolio_risk_conflict")
    if technical >= 0.75 and regime <= 0.35:
        conflict_flags.append("breakout_vs_overextension_conflict")
    if not fresh_market_data:
        conflict_flags.append("strong_setup_vs_stale_data_conflict")
    if abs(price_move_pct) >= 1.5:
        conflict_flags.append("proposal_invalidated_material_price_move")

    avg_conf = sum(item.confidence for item in results.values()) / max(len(results), 1)
    min_freshness = min(item.freshness for item in results.values())
    confidence_score = max(0.0, min(1.0, (avg_conf * 0.75) + (min_freshness * 0.25)))
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
        "conflictFlags": conflict_flags,
        "explanationSummary": (
            f"Technical={technical:.2f}, Risk={risk:.2f}, Regime={regime:.2f}, "
            f"Sentiment={sentiment:.2f}, Attention={attention:.2f}."
        ),
    }
