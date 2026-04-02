from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from django.utils import timezone

from trading_bot.bot.agent_signals import (
    CuratedListProvider,
    FeedSentimentProvider,
    MarketRegimeProvider,
    PortfolioRiskProvider,
    TechnicalSignalProvider,
    WatchlistAttentionProvider,
    detect_capabilities,
    fuse_signals,
)
from trading_bot.bot.etoro_adapter import build_etoro_adapter
from trading_bot.bot.event_pipeline import (
    EventBus,
    FeatureSnapshotBuilder,
    FusionEngine,
    LLMReasoningEngine,
)
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.micro_variation_engine import (
    MicroVariationProposalEngine,
)
from trading_bot.bot.models import (
    AgentApproval,
    AgentLog,
    AgentProposal,
    AgentSession,
    ExecutionEvent,
    PortfolioSnapshot,
    SymbolFeatureSnapshot,
)

logger = logging.getLogger(__name__)

WRITE_ACTIONS = {"BUY", "SELL", "REDUCE", "CLOSE", "CANCEL_ORDER"}
EXECUTION_ACTIONS = {
    "BUY_CANDIDATE": "BUY",
    "SELL_CANDIDATE": "SELL",
    "REDUCE_CANDIDATE": "REDUCE",
    "CLOSE_CANDIDATE": "CLOSE",
}


@dataclass
class AgentConfig:
    etoro_api_key: str
    openai_api_key: str
    max_agent_loss: float
    loop_interval_ms: int
    approval_timeout_ms: int
    max_open_proposals: int
    max_position_size: float
    max_daily_trades: int
    proposal_invalidation_pct: float


def load_agent_config() -> AgentConfig:
    def parse_int(name: str, default: int) -> int:
        parsed = int(os.getenv(name, str(default)))
        if parsed <= 0:
            raise ValueError(f"{name} must be > 0")
        return parsed

    etoro_api_key = os.getenv("ETORO_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    max_agent_loss_raw = os.getenv("MAX_AGENT_LOSS")
    if not etoro_api_key:
        raise ValueError("ETORO_API_KEY is required")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if max_agent_loss_raw is None:
        raise ValueError("MAX_AGENT_LOSS is required")

    max_agent_loss = float(max_agent_loss_raw)
    if max_agent_loss <= 0:
        raise ValueError("MAX_AGENT_LOSS must be > 0")

    return AgentConfig(
        etoro_api_key=etoro_api_key,
        openai_api_key=openai_api_key,
        max_agent_loss=max_agent_loss,
        loop_interval_ms=parse_int("AGENT_LOOP_INTERVAL_MS", 15000),
        approval_timeout_ms=parse_int("APPROVAL_TIMEOUT_MS", 180000),
        max_open_proposals=parse_int("MAX_OPEN_PROPOSALS", 5),
        max_position_size=float(os.getenv("MAX_POSITION_SIZE", "10000")),
        max_daily_trades=parse_int("MAX_DAILY_TRADES", 20),
        proposal_invalidation_pct=float(os.getenv("PROPOSAL_INVALIDATION_PCT", "1.5")),
    )


class AgentRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._config: AgentConfig | None = None
        self._session: AgentSession | None = None
        self._last_prices: dict[str, float] = {}
        self._micro_engine: MicroVariationProposalEngine | None = None
        self._event_bus: EventBus | None = None
        self._snapshot_builder: FeatureSnapshotBuilder | None = None
        self._llm_reasoner: LLMReasoningEngine | None = None
        self._fusion_engine = FusionEngine()
        self._metrics = {
            "events_sec": 0.0,
            "llm_latency_ms": 0,
            "llm_error_rate": 0.0,
            "proposals_created": 0,
            "proposals_invalidated": 0,
            "approval_rate": 0.0,
        }
        self._llm_calls = 0
        self._llm_failures = 0

    def _log(self, level: str, category: str, message: str, **kwargs: Any) -> None:
        if not self._session:
            return
        proposal = kwargs.get("proposal")
        AgentLog.objects.create(
            session=self._session,
            proposal=proposal,
            level=level,
            category=category,
            message=message,
            payload=kwargs.get("payload", {}),
            symbol=kwargs.get("symbol", ""),
            ui_line=f"[{category}] {message}",
        )

    def _snapshot_hash(self, account: dict[str, Any], positions: list[dict[str, Any]]) -> str:
        src = json.dumps({"account": account, "positions": positions}, sort_keys=True)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()

    def start(self) -> AgentSession:
        with self._lock:
            if self._thread and self._thread.is_alive() and self._session:
                return self._session
            self._config = load_agent_config()
            self._micro_engine = MicroVariationProposalEngine.from_env(self._log)
            self._event_bus = EventBus(debounce_ms=int(os.getenv("EVENT_DEBOUNCE_MS", "100")))
            self._snapshot_builder = FeatureSnapshotBuilder(
                stale_after_ms=int(os.getenv("SNAPSHOT_STALE_AFTER_MS", "15000"))
            )
            self._llm_reasoner = LLMReasoningEngine(api_key=self._config.openai_api_key)
            self._session = AgentSession.objects.create(
                status="STARTING",
                worker_healthy=True,
            )
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="agent-review-worker",
                daemon=True,
            )
            self._thread.start()
            return self._session

    def stop(self) -> AgentSession | None:
        with self._lock:
            if not self._session:
                return None
            self._session.status = "STOPPING"
            self._session.save(update_fields=["status"])
            self._stop_event.set()
            return self._session

    def session_payload(self) -> dict[str, Any]:
        if not self._session:
            return {
                "sessionId": None,
                "status": "IDLE",
                "pendingApprovals": 0,
                "recentLogs": [],
                "capabilities": {},
            }
        session = self._session
        session.refresh_from_db()
        uptime_ms = 0
        if session.started_at:
            end_time = session.stopped_at or timezone.now()
            uptime_ms = int((end_time - session.started_at).total_seconds() * 1000)

        pending_count = AgentProposal.objects.filter(
            session=session,
            status="pending_user",
        ).count()
        latest_snapshot = (
            PortfolioSnapshot.objects.filter(session=session).order_by("-timestamp").first()
        )
        account = latest_snapshot.account_summary if latest_snapshot else {}
        positions = latest_snapshot.positions if latest_snapshot else []
        logs_qs = AgentLog.objects.filter(session=session).order_by("-timestamp")
        logs = list(logs_qs.values("timestamp", "level", "category", "message", "ui_line")[:60])
        snapshots_qs = SymbolFeatureSnapshot.objects.filter(session=session).order_by("-timestamp")
        last_snapshots = {item.symbol: item.payload for item in snapshots_qs[:20]}
        approval_count = AgentApproval.objects.filter(proposal__session=session).count()
        approved_count = AgentApproval.objects.filter(
            proposal__session=session,
            approved=True,
        ).count()
        self._metrics["approval_rate"] = (
            0.0 if approval_count == 0 else approved_count / approval_count
        )
        return {
            "sessionId": str(session.session_id),
            "status": session.status,
            "startedAt": session.started_at,
            "stoppedAt": session.stopped_at,
            "uptimeMs": uptime_ms,
            "heartbeatAt": session.heartbeat_at,
            "workerHealthy": session.worker_healthy,
            "latestSyncAt": session.latest_sync_at,
            "latestAnalysisAt": session.latest_analysis_at,
            "portfolioSummary": account,
            "cash": account.get("cash"),
            "openPositions": len(positions),
            "unrealizedPnL": account.get("unrealizedPnl", 0),
            "realizedPnL": account.get("realizedPnl", 0),
            "drawdown": account.get("drawdown", 0),
            "riskLevel": account.get("riskLevel", "unknown"),
            "pendingApprovals": pending_count,
            "recentLogs": logs,
            "lastError": session.last_error,
            "capabilities": session.capability_registry,
            "streamingConnected": session.streaming_connected,
            "streamStatus": "active" if session.streaming_connected else "polling",
            "lastFeedSync": session.last_feed_sync_at,
            "lastWatchlistSync": session.last_watchlist_sync_at,
            "lastSnapshot": last_snapshots,
            "metrics": self._metrics,
        }

    def proposals_payload(self) -> list[dict[str, Any]]:
        if not self._session:
            return []
        proposals = AgentProposal.objects.filter(
            session=self._session,
            status="pending_user",
        ).order_by("created_at")
        return [
            {
                "proposalId": str(item.proposal_id),
                "symbol": item.symbol,
                "action": item.action,
                "size": item.size,
                "expectedImpact": item.expected_impact,
                "riskSummary": item.risk_summary,
                "explanation": item.explanation,
                "confidence": item.confidence,
                "whyNow": item.why_now,
                "rationale": item.rationale,
                "createdAt": item.created_at,
                "expiresAt": item.expires_at,
                "status": item.status,
                "providerScores": item.provider_scores,
                "signalFreshness": item.signal_freshness,
                "conflictFlags": item.conflict_flags,
                "invalidationReason": item.invalidation_reason,
                "interpretation": item.interpretation,
                "riskNote": item.risk_note,
                "uncertainty": item.uncertainty,
                "confidenceAdjustment": item.confidence_adjustment,
                "llmConflicts": item.llm_conflicts,
                "sentimentSummary": item.sentiment_summary,
                "proposalOrigin": item.sentiment_summary.get("proposalOrigin", "core"),
                "microMetrics": item.provider_scores.get("micro_variation", {}),
                "ttlRemaining": max(
                    int((item.expires_at - timezone.now()).total_seconds()),
                    0,
                ),
            }
            for item in proposals
        ]

    def _is_risk_veto(self, fused: dict[str, Any]) -> bool:
        return fused.get("proposalType") == "REQUIRE_REVIEW"

    def _invalidate_stale_proposals(self, symbol: str, event_id: str, reason: str) -> None:
        assert self._session
        updated = AgentProposal.objects.filter(
            session=self._session,
            status="pending_user",
            symbol=symbol,
        ).update(status="invalidated", invalidation_reason=reason)
        if updated:
            self._metrics["proposals_invalidated"] += updated
            self._log(
                "WARNING",
                "proposal_invalidation",
                "Proposal invalidated from event trigger",
                symbol=symbol,
                payload={"eventId": event_id, "reason": reason},
            )

    def _process_invalidation_triggers(
        self,
        symbol: str,
        price_move_pct: float,
        snapshot: dict[str, Any],
        event_id: str,
    ) -> None:
        assert self._config
        if abs(price_move_pct) >= self._config.proposal_invalidation_pct:
            self._invalidate_stale_proposals(symbol, event_id, "price_left_entry_band")
            return
        spread = float(snapshot["microstructure"].get("spread", 0.0) or 0.0)
        liquidity = float(snapshot["microstructure"].get("liquidity", 0.0) or 0.0)
        short_term_vol = float(snapshot["microstructure"].get("shortTermVol", 0.0) or 0.0)
        if short_term_vol >= self._config.proposal_invalidation_pct:
            self._invalidate_stale_proposals(symbol, event_id, "volatility_spike")
        if spread > 1.5:
            self._invalidate_stale_proposals(symbol, event_id, "spread_deterioration")
        if liquidity < 0:
            self._invalidate_stale_proposals(symbol, event_id, "liquidity_deterioration")

    def _build_provider_scores(self, provider_results: dict[str, Any]) -> dict[str, Any]:
        return {
            key: {
                "score": value.score,
                "confidence": value.confidence,
                "freshness": value.freshness,
                "rationale": value.rationale,
                "warnings": value.warnings,
            }
            for key, value in provider_results.items()
        }

    def approve(self, proposal_id: str) -> dict[str, Any]:
        if not self._session:
            raise ValueError("No active session")
        proposal = AgentProposal.objects.get(
            session=self._session,
            proposal_id=proposal_id,
        )
        if proposal.status != "pending_user":
            raise ValueError("Proposal not pending")
        if proposal.expires_at <= timezone.now():
            proposal.status = "expired"
            proposal.save(update_fields=["status"])
            raise ValueError("Proposal expired")
        AgentApproval.objects.create(proposal=proposal, approved=True)
        proposal.status = "approved"
        proposal.save(update_fields=["status"])
        self._log(
            "INFO",
            "approval",
            "Proposal approved",
            proposal=proposal,
            symbol=proposal.symbol,
        )
        return {"proposalId": proposal_id, "status": proposal.status}

    def reject(self, proposal_id: str) -> dict[str, Any]:
        if not self._session:
            raise ValueError("No active session")
        proposal = AgentProposal.objects.get(
            session=self._session,
            proposal_id=proposal_id,
        )
        AgentApproval.objects.create(proposal=proposal, approved=False)
        proposal.status = "rejected"
        proposal.save(update_fields=["status"])
        self._log(
            "INFO",
            "approval",
            "Proposal rejected",
            proposal=proposal,
            symbol=proposal.symbol,
        )
        return {"proposalId": proposal_id, "status": proposal.status}

    def _build_signal_context(
        self,
        adapter: Any,
        symbol: str,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
        history: list[dict[str, Any]],
        capabilities: dict[str, bool],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "adapter": adapter,
            "symbol": symbol,
            "account": account,
            "positions": positions,
            "portfolio_history": history,
            "capabilities": capabilities,
            "market_monitor": {},
        }
        if capabilities.get("supportsMarketMonitorStreaming"):
            self._session.streaming_connected = True
            self._session.save(update_fields=["streaming_connected"])
            self._log("INFO", "capability", "Market monitor streaming available")
        else:
            self._session.streaming_connected = False
            self._session.save(update_fields=["streaming_connected"])
            self._log("INFO", "capability", "Streaming unavailable; using polling")

        return context

    def _draft_from_fusion(
        self,
        symbol: str,
        positions: list[dict[str, Any]],
        fused: dict[str, Any],
    ) -> dict[str, Any]:
        symbol_position = next(
            (item for item in positions if str(item.get("symbol", "")).upper() == symbol),
            {},
        )
        exposure = abs(float(symbol_position.get("marketValue", 0)))
        size = round(min(max(exposure * 0.1, 50.0), self._config.max_position_size), 2)
        proposal_type = fused["proposalType"]
        action = proposal_type
        if proposal_type == "HOLD":
            size = 0
        return {
            "symbol": symbol,
            "action": action,
            "size": size,
            "rationale": fused["explanationSummary"],
            "risk_summary": (
                "Risk engine veto active"
                if proposal_type == "REQUIRE_REVIEW"
                else "Risk within limits"
            ),
        }

    def _openai_synthesis(
        self,
        symbol: str,
        provider_scores: dict[str, Any],
        fused: dict[str, Any],
    ) -> dict[str, Any]:
        assert self._config
        prompt_payload = {
            "symbol": symbol,
            "providerScores": provider_scores,
            "fused": fused,
            "requirements": [
                "highlight technical first",
                "explicitly mention risk conflicts",
                "state uncertainty when confidence is low",
                "never authorize direct execution",
            ],
        }
        try:
            decider = LLMDecider(
                provider="openai",
                model="gpt-4o-mini",
                api_key=self._config.openai_api_key,
            )
            summary = decider.summarize_json(prompt_payload)
            return {
                "explanation": summary,
                "why_now": f"{symbol} moved into a {fused['proposalType']} window.",
                "trade_offs": "Human approval required; risk constraints can override opportunity.",
                "confidence": fused["confidenceScore"],
            }
        except Exception as exc:
            logger.error("openai synthesis failed: %s", exc)
            self._log("ERROR", "sentiment", "OpenAI synthesis failed", payload={"error": str(exc)})
            return {
                "explanation": fused["explanationSummary"],
                "why_now": "Signal thresholds crossed.",
                "trade_offs": "Fallback synthesis path in use.",
                "confidence": fused["confidenceScore"],
            }

    def _execute_approved(self, adapter: Any, snapshot_hash: str) -> None:
        assert self._session
        approved = AgentProposal.objects.filter(
            session=self._session,
            status="approved",
        ).order_by("created_at")
        for item in approved:
            if item.expires_at <= timezone.now():
                item.status = "expired"
                item.save(update_fields=["status"])
                continue
            approval = item.approvals.filter(approved=True).order_by("-approved_at").first()
            if not approval:
                continue
            if item.snapshot_hash != snapshot_hash:
                item.status = "failed"
                item.invalidation_reason = "snapshot_changed"
                item.save(update_fields=["status", "invalidation_reason"])
                self._log(
                    "WARNING",
                    "proposal_invalidation",
                    "Snapshot changed; execution blocked",
                    proposal=item,
                    symbol=item.symbol,
                )
                continue

            mapped_action = EXECUTION_ACTIONS.get(item.action)
            if not mapped_action or mapped_action not in WRITE_ACTIONS:
                item.status = "submitted"
                item.save(update_fields=["status"])
                continue

            exists = ExecutionEvent.objects.filter(
                proposal=item,
                event_type="submission",
                status="ok",
            ).exists()
            if exists:
                self._log(
                    "WARNING",
                    "execution",
                    "Duplicate execution prevented",
                    proposal=item,
                    symbol=item.symbol,
                )
                continue

            approved_at = approval.approved_at.isoformat()
            pid = str(item.proposal_id)
            common = {
                "approvedProposalId": pid,
                "approvalTimestamp": approved_at,
                "snapshotHash": snapshot_hash,
            }
            if mapped_action in {"BUY", "SELL"}:
                result = adapter.placeOrder(
                    **common,
                    symbol=item.symbol,
                    side=mapped_action,
                    size=item.size,
                )
            elif mapped_action == "REDUCE":
                result = adapter.reducePosition(**common, symbol=item.symbol, size=item.size)
            elif mapped_action == "CLOSE":
                result = adapter.closePosition(**common, symbol=item.symbol)
            else:
                result = adapter.cancelOrderIfSupported(**common, orderId=item.symbol)

            item.status = "submitted"
            item.save(update_fields=["status"])
            ExecutionEvent.objects.create(
                session=self._session,
                proposal=item,
                event_type="submission",
                status="ok",
                payload=result,
            )
            self._log(
                "INFO",
                "execution",
                "Approved proposal submitted",
                proposal=item,
                symbol=item.symbol,
                payload={"result": result},
            )

    def _loop(self) -> None:
        assert self._session and self._config and self._micro_engine
        self._session.started_at = timezone.now()
        self._session.status = "RUNNING"
        self._session.save(update_fields=["started_at", "status"])
        self._log("INFO", "lifecycle", "Agent started")
        try:
            adapter = build_etoro_adapter()
            capabilities = detect_capabilities(adapter)
            self._session.capability_registry = capabilities
            self._session.save(update_fields=["capability_registry"])
            self._log("INFO", "capability", "Capability registry detected", payload=capabilities)

            providers = {
                "technical": TechnicalSignalProvider(),
                "portfolio_risk": PortfolioRiskProvider(),
                "market_regime": MarketRegimeProvider(),
                "social_sentiment": FeedSentimentProvider(
                    openai_api_key=self._config.openai_api_key
                ),
                "watchlist_interest": WatchlistAttentionProvider(),
                "curated_interest": CuratedListProvider(),
            }

            while not self._stop_event.is_set():
                now = timezone.now()
                self._session.heartbeat_at = now
                self._session.worker_healthy = True
                self._session.save(update_fields=["heartbeat_at", "worker_healthy"])

                account = adapter.getAccountSummary()
                positions = adapter.getPositions()
                orders = adapter.getOpenOrders()
                history = adapter.getPortfolioHistory()
                if capabilities.get("supportsFeeds"):
                    self._session.last_feed_sync_at = timezone.now()
                if capabilities.get("supportsWatchlists"):
                    self._session.last_watchlist_sync_at = timezone.now()
                self._session.latest_sync_at = timezone.now()
                self._session.save(
                    update_fields=[
                        "latest_sync_at",
                        "last_feed_sync_at",
                        "last_watchlist_sync_at",
                    ]
                )

                snapshot_hash = self._snapshot_hash(account, positions)
                PortfolioSnapshot.objects.create(
                    session=self._session,
                    account_summary=account,
                    positions=positions,
                    open_orders=orders,
                    portfolio_history=history,
                    snapshot_hash=snapshot_hash,
                )
                self._log(
                    "INFO",
                    "broker_sync",
                    "Broker sync completed",
                    payload={"positions": len(positions), "openOrders": len(orders)},
                )

                expired_count = AgentProposal.objects.filter(
                    session=self._session,
                    status="pending_user",
                    expires_at__lte=now,
                ).update(status="expired")
                if expired_count:
                    self._log(
                        "INFO",
                        "proposal_invalidation",
                        "Proposal TTL expired",
                        payload={"reason": "PROPOSAL_TTL_EXPIRE", "count": expired_count},
                    )

                pending_qs = AgentProposal.objects.filter(
                    session=self._session,
                    status="pending_user",
                )
                pending = pending_qs.count()
                available_slots = self._config.max_open_proposals - pending
                if available_slots > 0:
                    new_candidates: list[dict[str, Any]] = []
                    symbols = [
                        str(item.get("symbol", "")).upper()
                        for item in positions
                        if item.get("symbol")
                    ]
                    symbol = symbols[0] if symbols else "SPY"
                    context = self._build_signal_context(
                        adapter,
                        symbol,
                        account,
                        positions,
                        history,
                        capabilities,
                    )
                    provider_results = {
                        name: provider.collect(symbol=symbol, context=context)
                        for name, provider in providers.items()
                    }
                    latest_price = float(account.get("equity", 0) or 0)
                    previous_price = self._last_prices.get(symbol, latest_price)
                    price_move_pct = (
                        0.0
                        if previous_price == 0
                        else ((latest_price - previous_price) / abs(previous_price)) * 100
                    )
                    self._last_prices[symbol] = latest_price
                    fused = fuse_signals(
                        provider_results,
                        price_move_pct=price_move_pct,
                        fresh_market_data=True,
                    )
                    fused["priceMovePct"] = price_move_pct
                    event_types = [
                        "PORTFOLIO_UPDATE",
                        "FEED_UPDATE",
                        "PRICE_UPDATE",
                        "VOLATILITY_CHANGE",
                    ]
                    if fused.get("proposalType") in {"BUY_CANDIDATE", "SELL_CANDIDATE"}:
                        event_types.append("BREAKOUT_DETECTED")
                    for event_type in event_types:
                        event_id = str(uuid.uuid4())
                        if self._event_bus:
                            self._event_bus.publish(
                                event_type=event_type,
                                symbol=symbol,
                                payload={"eventId": event_id, "snapshotHash": snapshot_hash},
                            )

                    assert self._snapshot_builder and self._llm_reasoner
                    event_id = str(uuid.uuid4())
                    snapshot = self._snapshot_builder.build(
                        symbol=symbol,
                        event_id=event_id,
                        provider_results=provider_results,
                        fused=fused,
                        account=account,
                        positions=positions,
                    )
                    snapshot_hash_key = self._snapshot_builder.snapshot_hash(snapshot)
                    SymbolFeatureSnapshot.objects.update_or_create(
                        session=self._session,
                        symbol=symbol,
                        defaults={
                            "event_id": event_id,
                            "snapshot_hash": snapshot_hash_key,
                            "payload": snapshot,
                        },
                    )
                    if snapshot["freshness"]["isStale"]:
                        self._log(
                            "WARNING",
                            "proposal_invalidation",
                            "Snapshot stale, proposal blocked",
                            symbol=symbol,
                            payload={"eventId": event_id, "snapshotHash": snapshot_hash_key},
                        )
                        self._invalidate_stale_proposals(
                            symbol=symbol,
                            event_id=event_id,
                            reason="stale_snapshot",
                        )
                    self._process_invalidation_triggers(
                        symbol=symbol,
                        price_move_pct=price_move_pct,
                        snapshot=snapshot,
                        event_id=event_id,
                    )
                    deterministic_score = float(fused.get("confidenceScore", 0.0))
                    llm_started = time.perf_counter()
                    self._llm_calls += 1
                    llm_result = self._llm_reasoner.interpret(
                        symbol=symbol,
                        snapshot=snapshot,
                        deterministic_score=deterministic_score,
                    )
                    self._metrics["llm_latency_ms"] = int(
                        (time.perf_counter() - llm_started) * 1000
                    )
                    if "fallback" in llm_result.get("risk_note", "").lower():
                        self._llm_failures += 1
                    self._metrics["llm_error_rate"] = (
                        0.0 if self._llm_calls == 0 else self._llm_failures / self._llm_calls
                    )
                    fusion = self._fusion_engine.combine(
                        base_score=deterministic_score,
                        llm_result=llm_result,
                        risk_veto=self._is_risk_veto(fused),
                        is_stale=snapshot["freshness"]["isStale"],
                    )
                    self._log(
                        "INFO",
                        "fusion",
                        "Deterministic + llm fusion completed",
                        symbol=symbol,
                        payload={
                            "eventId": event_id,
                            "snapshotHash": snapshot_hash_key,
                            "proposalScore": fusion["proposalScore"],
                        },
                    )
                    drafted = self._draft_from_fusion(symbol, positions, fused)
                    provider_scores = self._build_provider_scores(provider_results)
                    ai = self._openai_synthesis(symbol, provider_scores, fused)
                    threshold = 0.3 if llm_result.get("uncertainty") != "high" else 0.5
                    if fusion["proposalScore"] < threshold:
                        drafted["action"] = "HOLD"
                    new_candidates.append(
                        {
                            "symbol": drafted["symbol"],
                            "action": drafted["action"],
                            "size": drafted["size"],
                            "rationale": drafted["rationale"],
                            "risk_summary": drafted["risk_summary"],
                            "explanation": ai["explanation"],
                            "confidence": float(ai["confidence"]),
                            "expected_impact": ai["trade_offs"],
                            "why_now": ai["why_now"],
                            "expires_at": now
                            + timedelta(milliseconds=self._config.approval_timeout_ms),
                            "provider_scores": provider_scores,
                            "interpretation": llm_result["interpretation"],
                            "risk_note": llm_result["risk_note"],
                            "uncertainty": llm_result["uncertainty"],
                            "confidence_adjustment": llm_result["confidence_adjustment"],
                            "llm_conflicts": llm_result["conflicts"],
                            "snapshot_hash_key": snapshot_hash_key,
                            "signal_freshness": {
                                key: value.freshness for key, value in provider_results.items()
                            },
                            "conflict_flags": fused["conflictFlags"] + llm_result["conflicts"],
                            "invalidation_reason": (
                                "material_price_move"
                                if "proposal_invalidated_material_price_move"
                                in fused["conflictFlags"]
                                else ""
                            ),
                            "sentiment_summary": provider_results[
                                "social_sentiment"
                            ].raw_inputs.get("summary", {}),
                            "decision_path": ["core_signal_fusion"],
                        }
                    )

                    core_symbols = {candidate["symbol"] for candidate in new_candidates}
                    pending_symbols = set(
                        pending_qs.values_list("symbol", flat=True)
                    )
                    try:
                        def build_micro_context(
                            candidate_symbol: str,
                            account: dict[str, Any] = account,
                            positions: list[dict[str, Any]] = positions,
                            history: list[dict[str, Any]] = history,
                            capabilities: dict[str, bool] = capabilities,
                        ) -> dict[str, Any]:
                            return self._build_signal_context(
                                adapter,
                                candidate_symbol,
                                account=account,
                                positions=positions,
                                history=history,
                                capabilities=capabilities,
                            )

                        micro_candidates = self._micro_engine.generate(
                            session=self._session,
                            now=now,
                            account=account,
                            positions=positions,
                            history=history,
                            capabilities=capabilities,
                            providers=providers,
                            build_context=build_micro_context,
                            draft_from_fusion=self._draft_from_fusion,
                            openai_synthesis=self._openai_synthesis,
                            last_prices=self._last_prices,
                            max_agent_loss=self._config.max_agent_loss,
                            core_symbols=core_symbols,
                            pending_symbols=pending_symbols,
                        )
                    except Exception as exc:
                        logger.error("micro_engine failed: %s", exc)
                        self._log(
                            "ERROR",
                            "micro_engine",
                            "Micro engine failure",
                            payload={"error": str(exc), "reason": "engine_error"},
                        )
                        micro_candidates = []

                    merged_candidates = new_candidates + micro_candidates
                    deduped_candidates: list[dict[str, Any]] = []
                    seen_symbols: set[str] = set()
                    for candidate in merged_candidates:
                        if candidate["symbol"] in seen_symbols:
                            self._log(
                                "INFO",
                                "micro_dedup",
                                "Candidate symbol deduplicated",
                                symbol=candidate["symbol"],
                                payload={
                                    "reason": "merge_dedup",
                                    "decisionPath": candidate.get("decision_path", []),
                                },
                            )
                            continue
                        seen_symbols.add(candidate["symbol"])
                        deduped_candidates.append(candidate)
                        if len(deduped_candidates) >= available_slots:
                            break

                    for candidate in deduped_candidates:
                        duplicate = AgentProposal.objects.filter(
                            session=self._session,
                            symbol=candidate["symbol"],
                            action=candidate["action"],
                            status="pending_user",
                            snapshot_hash_key=candidate.get("snapshot_hash_key", ""),
                        ).exists()
                        if duplicate:
                            self._log(
                                "INFO",
                                "micro_dedup",
                                "Proposal already pending",
                                symbol=candidate["symbol"],
                                payload={"reason": "pending_duplicate", "decisionPath": ["db"]},
                            )
                            continue

                        proposal = AgentProposal.objects.create(
                            session=self._session,
                            symbol=candidate["symbol"],
                            action=candidate["action"],
                            size=candidate["size"],
                            rationale=candidate["rationale"],
                            risk_summary=candidate["risk_summary"],
                            explanation=candidate["explanation"],
                            confidence=candidate["confidence"],
                            expected_impact=candidate["expected_impact"],
                            why_now=candidate["why_now"],
                            expires_at=candidate["expires_at"],
                            status="pending_user",
                            snapshot_hash=snapshot_hash,
                            provider_scores=candidate["provider_scores"],
                            signal_freshness=candidate["signal_freshness"],
                            conflict_flags=candidate["conflict_flags"],
                            invalidation_reason=candidate["invalidation_reason"],
                            sentiment_summary=candidate["sentiment_summary"],
                            interpretation=candidate.get("interpretation", ""),
                            risk_note=candidate.get("risk_note", ""),
                            uncertainty=candidate.get("uncertainty", "medium"),
                            confidence_adjustment=candidate.get("confidence_adjustment", 0.0),
                            llm_conflicts=candidate.get("llm_conflicts", []),
                            snapshot_hash_key=candidate.get("snapshot_hash_key", ""),
                        )
                        self._log(
                            "INFO",
                            "proposal_update",
                            "New proposal pending approval",
                            proposal=proposal,
                            symbol=proposal.symbol,
                            payload={
                                "provider_scores": candidate["provider_scores"],
                                "decisionPath": candidate.get("decision_path", []),
                            },
                        )

                self._session.latest_analysis_at = timezone.now()
                if self._event_bus:
                    bus_metrics = self._event_bus.metrics()
                    self._metrics["events_sec"] = round(
                        bus_metrics.get("events_dispatched", 0)
                        / max(self._config.loop_interval_ms / 1000, 1),
                        2,
                    )
                waiting = AgentProposal.objects.filter(
                    session=self._session,
                    status="pending_user",
                ).exists()
                self._session.status = "WAITING_APPROVAL" if waiting else "RUNNING"
                self._session.save(update_fields=["latest_analysis_at", "status"])

                self._execute_approved(adapter=adapter, snapshot_hash=snapshot_hash)
                time.sleep(self._config.loop_interval_ms / 1000)
        except Exception as exc:
            logger.error("agent loop crashed: %s", exc)
            self._session.status = "ERROR"
            self._session.last_error = str(exc)
            self._session.worker_healthy = False
            self._session.save(update_fields=["status", "last_error", "worker_healthy"])
            self._log("ERROR", "error", "Agent loop failed", payload={"error": str(exc)})
        finally:
            self._session.stopped_at = timezone.now()
            if self._session.status != "ERROR":
                self._session.status = "STOPPED"
            self._session.save(update_fields=["stopped_at", "status"])
            self._log("INFO", "lifecycle", "Agent stopped")


agent_runtime = AgentRuntime()
