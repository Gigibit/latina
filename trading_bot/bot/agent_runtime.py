from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from django.utils import timezone

from trading_bot.bot.etoro_adapter import build_etoro_adapter
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.models import (
    AgentApproval,
    AgentLog,
    AgentProposal,
    AgentSession,
    ExecutionEvent,
    PortfolioSnapshot,
)

logger = logging.getLogger(__name__)

WRITE_ACTIONS = {"BUY", "SELL", "REDUCE", "CLOSE", "CANCEL_ORDER"}


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
    )


class AgentRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._config: AgentConfig | None = None
        self._session: AgentSession | None = None

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
            return {"sessionId": None, "status": "IDLE", "pendingApprovals": 0, "recentLogs": []}
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
        latest_snapshot = PortfolioSnapshot.objects.filter(
            session=session
        ).order_by("-timestamp").first()
        account = latest_snapshot.account_summary if latest_snapshot else {}
        positions = latest_snapshot.positions if latest_snapshot else []
        logs_qs = AgentLog.objects.filter(session=session).order_by("-timestamp")
        logs = list(logs_qs.values("timestamp", "level", "category", "message", "ui_line")[:60])
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
            }
            for item in proposals
        ]

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

    def _build_proposal(
        self,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        assert self._config
        unrealized = float(account.get("unrealizedPnl", 0))
        cash = float(account.get("cash", 0))
        if abs(unrealized) >= self._config.max_agent_loss:
            self._log(
                "WARNING",
                "risk",
                "MAX_AGENT_LOSS breached; proposal generation blocked",
            )
            return None
        if not positions:
            return {
                "symbol": "SPY",
                "action": "WATCH",
                "size": 0,
                "rationale": "No open positions. Watching benchmark.",
                "risk_summary": "No immediate portfolio risk.",
            }

        biggest = max(positions, key=lambda p: abs(float(p.get("marketValue", 0))))
        symbol = biggest.get("symbol", "UNKNOWN")
        exposure = abs(float(biggest.get("marketValue", 0)))
        action = "REDUCE" if exposure > self._config.max_position_size else "HOLD"
        size = round(min(exposure * 0.1, self._config.max_position_size), 2)
        rationale = (
            f"Exposure={exposure:.2f}, cash={cash:.2f}, "
            f"unrealizedPnL={unrealized:.2f}"
        )
        risk_summary = "Concentration risk elevated" if action == "REDUCE" else "Risk acceptable"
        return {
            "symbol": symbol,
            "action": action,
            "size": size,
            "rationale": rationale,
            "risk_summary": risk_summary,
        }

    def _ai_explain(self, proposal: dict[str, Any], account: dict[str, Any]) -> dict[str, Any]:
        assert self._config
        decider = LLMDecider(
            provider="openai",
            model="gpt-4o-mini",
            api_key=self._config.openai_api_key,
        )
        prompt = (
            "Return JSON keys: explanation, why_now, uncertainty_analysis, "
            "risk_note, trade_offs, confidence. "
            f"Proposal={json.dumps(proposal)} Account={json.dumps(account)}"
        )
        try:
            raw = decider.summarize_json({"prompt": prompt})
            return {
                "explanation": raw,
                "why_now": "Risk drift detected now.",
                "uncertainty_analysis": "Market regime can change abruptly.",
                "risk_note": proposal["risk_summary"],
                "trade_offs": "Reducing protects downside but can trim upside.",
                "confidence": 0.62,
            }
        except Exception as exc:
            logger.error("ai explanation failed: %s", exc)
            self._log(
                "ERROR",
                "ai_explanation",
                "AI explanation failed",
                payload={"error": str(exc)},
            )
            return {
                "explanation": proposal["rationale"],
                "why_now": "Signal threshold crossed.",
                "uncertainty_analysis": "Fallback explanation used.",
                "risk_note": proposal["risk_summary"],
                "trade_offs": "Limited confidence due to fallback path.",
                "confidence": 0.5,
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
                item.save(update_fields=["status"])
                self._log(
                    "WARNING",
                    "execution",
                    "Snapshot changed; execution blocked",
                    proposal=item,
                    symbol=item.symbol,
                )
                continue
            if item.action not in WRITE_ACTIONS:
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
            if item.action in {"BUY", "SELL"}:
                result = adapter.placeOrder(
                    **common,
                    symbol=item.symbol,
                    side=item.action,
                    size=item.size,
                )
            elif item.action == "REDUCE":
                result = adapter.reducePosition(**common, symbol=item.symbol, size=item.size)
            elif item.action == "CLOSE":
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
        assert self._session and self._config
        self._session.started_at = timezone.now()
        self._session.status = "RUNNING"
        self._session.save(update_fields=["started_at", "status"])
        self._log("INFO", "lifecycle", "Agent started")
        try:
            adapter = build_etoro_adapter()
            while not self._stop_event.is_set():
                now = timezone.now()
                self._session.heartbeat_at = now
                self._session.worker_healthy = True
                self._session.save(update_fields=["heartbeat_at", "worker_healthy"])

                account = adapter.getAccountSummary()
                positions = adapter.getPositions()
                orders = adapter.getOpenOrders()
                history = adapter.getPortfolioHistory()

                self._session.latest_sync_at = timezone.now()
                self._session.save(update_fields=["latest_sync_at"])

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

                AgentProposal.objects.filter(
                    session=self._session,
                    status="pending_user",
                    expires_at__lte=now,
                ).update(status="expired")

                pending = AgentProposal.objects.filter(
                    session=self._session,
                    status="pending_user",
                ).count()
                if pending < self._config.max_open_proposals:
                    drafted = self._build_proposal(account, positions)
                    if drafted:
                        duplicate = AgentProposal.objects.filter(
                            session=self._session,
                            symbol=drafted["symbol"],
                            action=drafted["action"],
                            status="pending_user",
                        ).exists()
                        if not duplicate:
                            ai = self._ai_explain(drafted, account)
                            proposal = AgentProposal.objects.create(
                                session=self._session,
                                symbol=drafted["symbol"],
                                action=drafted["action"],
                                size=drafted["size"],
                                rationale=drafted["rationale"],
                                risk_summary=drafted["risk_summary"],
                                explanation=ai["explanation"],
                                confidence=float(ai["confidence"]),
                                expected_impact=ai["trade_offs"],
                                why_now=ai["why_now"],
                                expires_at=now + timedelta(
                                    milliseconds=self._config.approval_timeout_ms
                                ),
                                status="pending_user",
                                snapshot_hash=snapshot_hash,
                            )
                            self._log(
                                "INFO",
                                "recommendation",
                                "New proposal pending approval",
                                proposal=proposal,
                                symbol=proposal.symbol,
                            )
                self._session.latest_analysis_at = timezone.now()
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
            self._log(
                "ERROR",
                "error",
                "Agent loop failed",
                payload={"error": str(exc)},
            )
        finally:
            self._session.stopped_at = timezone.now()
            if self._session.status != "ERROR":
                self._session.status = "STOPPED"
            self._session.save(update_fields=["stopped_at", "status"])
            self._log("INFO", "lifecycle", "Agent stopped")


agent_runtime = AgentRuntime()
