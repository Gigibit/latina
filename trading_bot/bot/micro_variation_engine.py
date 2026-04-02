from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from statistics import mean
from typing import Any, Callable

from trading_bot.bot.agent_signals import fuse_signals
from trading_bot.bot.models import AgentProposal, AgentSession

logger = logging.getLogger(__name__)

MICRO_PROPOSAL_ORIGIN = "micro"


@dataclass(frozen=True)
class MicroVariationConfig:
    max_active_proposals_global: int
    max_active_proposals_per_symbol: int
    max_new_proposals_per_cycle: int
    proposal_ttl_ms: int


class MicroVariationProposalEngine:
    def __init__(self, config: MicroVariationConfig, log_fn: Callable[..., None]) -> None:
        self._config = config
        self._log_fn = log_fn

    @classmethod
    def from_env(cls, log_fn: Callable[..., None]) -> "MicroVariationProposalEngine":
        def parse_int(name: str, default: int) -> int:
            value = int(os.getenv(name, str(default)))
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
            return value

        config = MicroVariationConfig(
            max_active_proposals_global=parse_int("MICRO_MAX_ACTIVE_PROPOSALS_GLOBAL", 2),
            max_active_proposals_per_symbol=parse_int("MICRO_MAX_ACTIVE_PROPOSALS_PER_SYMBOL", 1),
            max_new_proposals_per_cycle=parse_int("MICRO_MAX_NEW_PROPOSALS_PER_CYCLE", 1),
            proposal_ttl_ms=parse_int("MICRO_PROPOSAL_TTL_MS", 60000),
        )
        return cls(config=config, log_fn=log_fn)

    def generate(
        self,
        *,
        session: AgentSession,
        now,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
        history: list[dict[str, Any]],
        capabilities: dict[str, bool],
        providers: dict[str, Any],
        build_context: Callable[[str], dict[str, Any]],
        draft_from_fusion: Callable[[str, list[dict[str, Any]], dict[str, Any]], dict[str, Any]],
        openai_synthesis: Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]],
        last_prices: dict[str, float],
        max_agent_loss: float,
        core_symbols: set[str],
        pending_symbols: set[str],
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        self._expire_pending(session=session, now=now)

        active_global = self._active_micro_queryset(session=session).count()
        if active_global >= self._config.max_active_proposals_global:
            self._log_rate_limit("*", "global_limit")
            return created

        for symbol in self._symbol_universe(positions):
            if len(created) >= self._config.max_new_proposals_per_cycle:
                self._log_rate_limit(symbol, "per_cycle_limit")
                break

            if active_global + len(created) >= self._config.max_active_proposals_global:
                self._log_rate_limit(symbol, "global_limit")
                break

            if symbol in core_symbols:
                self._log_dedup(symbol, "core_proposal_exists")
                continue

            if symbol in pending_symbols:
                self._log_dedup(symbol, "pending_approval_exists")
                continue

            active_symbol_count = self._active_micro_queryset(
                session=session,
                symbol=symbol,
            ).count()
            if active_symbol_count >= self._config.max_active_proposals_per_symbol:
                self._log_rate_limit(symbol, "per_symbol_limit")
                continue

            if self._portfolio_conflict(symbol, account, positions, max_agent_loss):
                self._log_skip(symbol, "portfolio_conflict", "portfolio_gate")
                continue

            try:
                context = build_context(
                    symbol,
                )
                provider_results = {
                    name: provider.collect(symbol=symbol, context=context)
                    for name, provider in providers.items()
                }
                latest_price = self._latest_symbol_price(symbol, positions)
                previous_price = last_prices.get(symbol, latest_price)
                price_move_pct = (
                    0.0
                    if previous_price == 0
                    else ((latest_price - previous_price) / abs(previous_price)) * 100
                )
                last_prices[symbol] = latest_price

                fused = fuse_signals(
                    provider_results,
                    price_move_pct=price_move_pct,
                    fresh_market_data=True,
                )
                drafted = draft_from_fusion(symbol, positions, fused)
                if drafted["action"] not in {"BUY_CANDIDATE", "SELL_CANDIDATE"}:
                    self._log_skip(symbol, "non_actionable", "fused_to_draft")
                    continue

                provider_scores = {
                    key: {
                        "score": value.score,
                        "confidence": value.confidence,
                        "freshness": value.freshness,
                        "rationale": value.rationale,
                        "warnings": value.warnings,
                    }
                    for key, value in provider_results.items()
                }
                micro_metrics = self._micro_metrics(history)
                provider_scores["micro_variation"] = {
                    "tag": "MICRO",
                    **micro_metrics,
                }
                ai = openai_synthesis(symbol, provider_scores, fused)
                created.append(
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
                        "expires_at": now + timedelta(milliseconds=self._config.proposal_ttl_ms),
                        "provider_scores": provider_scores,
                        "signal_freshness": {
                            key: value.freshness for key, value in provider_results.items()
                        },
                        "conflict_flags": fused["conflictFlags"],
                        "invalidation_reason": (
                            "material_price_move"
                            if "proposal_invalidated_material_price_move" in fused["conflictFlags"]
                            else ""
                        ),
                        "sentiment_summary": {
                            **provider_results["social_sentiment"].raw_inputs.get("summary", {}),
                            "proposalOrigin": MICRO_PROPOSAL_ORIGIN,
                        },
                        "decision_path": ["scan", "risk_validation", "create"],
                    }
                )
                self._log_fn(
                    "INFO",
                    "micro_engine",
                    "Micro proposal candidate generated",
                    symbol=symbol,
                    payload={"decisionPath": ["scan", "risk_validation", "create"]},
                )
            except Exception as exc:
                logger.error("micro_engine symbol failed symbol=%s error=%s", symbol, exc)
                self._log_fn(
                    "ERROR",
                    "micro_engine",
                    "Micro engine symbol failure",
                    symbol=symbol,
                    payload={"reason": "engine_error", "decisionPath": ["scan"], "error": str(exc)},
                )

        return created

    def _active_micro_queryset(self, *, session: AgentSession, symbol: str | None = None):
        filters: dict[str, Any] = {
            "session": session,
            "status__in": ["pending_user", "approved"],
            "sentiment_summary__proposalOrigin": MICRO_PROPOSAL_ORIGIN,
        }
        if symbol:
            filters["symbol"] = symbol
        return AgentProposal.objects.filter(**filters)

    def _expire_pending(self, *, session: AgentSession, now) -> None:
        expired = AgentProposal.objects.filter(
            session=session,
            status="pending_user",
            expires_at__lte=now,
            sentiment_summary__proposalOrigin=MICRO_PROPOSAL_ORIGIN,
        )
        for proposal in expired:
            proposal.status = "expired"
            proposal.save(update_fields=["status"])
            self._log_fn(
                "INFO",
                "micro_ttl_expire",
                "Micro proposal expired",
                symbol=proposal.symbol,
                proposal=proposal,
                payload={"reason": "ttl_elapsed", "decisionPath": ["ttl_check"]},
            )

    def _symbol_universe(self, positions: list[dict[str, Any]]) -> list[str]:
        symbols = {
            str(item.get("symbol", "")).upper()
            for item in positions
            if item.get("symbol")
        }
        if not symbols:
            symbols = {"SPY"}
        return sorted(symbols)

    def _portfolio_conflict(
        self,
        symbol: str,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
        max_agent_loss: float,
    ) -> bool:
        total_exposure = sum(abs(float(item.get("marketValue", 0) or 0)) for item in positions)
        symbol_exposure = sum(
            abs(float(item.get("marketValue", 0) or 0))
            for item in positions
            if str(item.get("symbol", "")).upper() == symbol
        )
        equity = float(account.get("equity", 0) or 0)
        drawdown = abs(float(account.get("drawdown", 0) or 0))
        if drawdown >= max_agent_loss:
            return True
        if equity > 0 and total_exposure / equity > 1.5:
            return True
        if equity > 0 and symbol_exposure / equity > 0.5:
            return True
        return False

    def _micro_metrics(self, history: list[dict[str, Any]]) -> dict[str, float]:
        closes = [float(item.get("equity", 0) or 0) for item in history[-12:]]
        closes = [value for value in closes if value > 0]
        if len(closes) < 3:
            return {
                "localLow": 0.0,
                "localHigh": 0.0,
                "microRange": 0.0,
                "expectedEdge": 0.0,
            }
        local_low = min(closes)
        local_high = max(closes)
        micro_range = local_high - local_low
        baseline = mean(closes)
        expected_edge = 0.0 if baseline == 0 else micro_range / baseline
        return {
            "localLow": round(local_low, 4),
            "localHigh": round(local_high, 4),
            "microRange": round(micro_range, 4),
            "expectedEdge": round(expected_edge, 4),
        }

    def _latest_symbol_price(self, symbol: str, positions: list[dict[str, Any]]) -> float:
        for position in positions:
            if str(position.get("symbol", "")).upper() == symbol:
                raw = position.get("currentRate") or position.get("price") or 0
                return float(raw or 0)
        return 0.0

    def _log_skip(self, symbol: str, reason: str, stage: str) -> None:
        self._log_fn(
            "INFO",
            "micro_skip",
            "Micro proposal skipped",
            symbol=symbol,
            payload={"reason": reason, "decisionPath": [stage]},
        )

    def _log_rate_limit(self, symbol: str, reason: str) -> None:
        self._log_fn(
            "INFO",
            "micro_rate_limit",
            "rate_limited",
            symbol=symbol,
            payload={"reason": reason, "decisionPath": ["rate_limit"]},
        )

    def _log_dedup(self, symbol: str, reason: str) -> None:
        self._log_fn(
            "INFO",
            "micro_dedup",
            "Micro proposal deduplicated",
            symbol=symbol,
            payload={"reason": reason, "decisionPath": ["dedup"]},
        )
