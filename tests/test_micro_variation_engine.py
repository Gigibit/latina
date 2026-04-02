from types import SimpleNamespace

from django.utils import timezone

from trading_bot.bot.agent_signals import SignalResult
from trading_bot.bot.micro_variation_engine import (
    MICRO_PROPOSAL_ORIGIN,
    MicroVariationConfig,
    MicroVariationProposalEngine,
)


class DummyProvider:
    def __init__(self, score: float):
        self.score = score

    def collect(self, *, symbol: str, context: dict):
        return SignalResult(
            score=self.score,
            confidence=0.8,
            freshness=1.0,
            rationale=f"score={self.score}",
            raw_inputs={"summary": {}},
            warnings=[],
        )


class DummyQuerySet:
    def __init__(self, count_value: int):
        self._count_value = count_value

    def count(self):
        return self._count_value


def _providers():
    return {
        "technical": DummyProvider(0.8),
        "portfolio_risk": DummyProvider(0.9),
        "market_regime": DummyProvider(0.7),
        "social_sentiment": DummyProvider(0.6),
        "watchlist_interest": DummyProvider(0.6),
        "curated_interest": DummyProvider(0.6),
    }


def _build_context(symbol: str):
    return {
        "symbol": symbol,
        "adapter": object(),
        "account": {},
        "positions": [],
        "portfolio_history": [],
        "capabilities": {},
        "market_monitor": {},
    }


def _draft(symbol, _positions, _fused):
    return {
        "symbol": symbol,
        "action": "BUY_CANDIDATE",
        "size": 100.0,
        "rationale": "micro",
        "risk_summary": "ok",
    }


def _ai(_symbol, _provider_scores, fused):
    return {
        "explanation": "micro explanation",
        "why_now": "because",
        "trade_offs": "human approval",
        "confidence": fused["confidenceScore"],
    }


def test_micro_engine_skips_symbol_when_core_candidate_exists(monkeypatch):
    logs = []
    engine = MicroVariationProposalEngine(
        MicroVariationConfig(2, 1, 1, 60000),
        lambda level, category, message, **kwargs: logs.append((level, category, kwargs)),
    )
    monkeypatch.setattr(engine, "_active_micro_queryset", lambda **_kwargs: DummyQuerySet(0))
    monkeypatch.setattr(engine, "_expire_pending", lambda **_kwargs: None)

    result = engine.generate(
        session=SimpleNamespace(),
        now=timezone.now(),
        account={"equity": 10000, "drawdown": 0.1},
        positions=[{"symbol": "SPY", "marketValue": 1000, "currentRate": 100}],
        history=[{"equity": 10000}, {"equity": 10100}, {"equity": 10050}],
        capabilities={},
        providers=_providers(),
        build_context=_build_context,
        draft_from_fusion=_draft,
        openai_synthesis=_ai,
        last_prices={},
        max_agent_loss=10,
        core_symbols={"SPY"},
        pending_symbols=set(),
    )

    assert result == []
    assert any(category == "micro_dedup" for _, category, _ in logs)


def test_micro_engine_generates_candidate_with_micro_metadata(monkeypatch):
    engine = MicroVariationProposalEngine(
        MicroVariationConfig(2, 1, 1, 45000),
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_active_micro_queryset", lambda **_kwargs: DummyQuerySet(0))
    monkeypatch.setattr(engine, "_expire_pending", lambda **_kwargs: None)

    result = engine.generate(
        session=SimpleNamespace(),
        now=timezone.now(),
        account={"equity": 10000, "drawdown": 0.05},
        positions=[{"symbol": "SPY", "marketValue": 1000, "currentRate": 100}],
        history=[{"equity": 10000}, {"equity": 10100}, {"equity": 10050}, {"equity": 10200}],
        capabilities={},
        providers=_providers(),
        build_context=_build_context,
        draft_from_fusion=_draft,
        openai_synthesis=_ai,
        last_prices={},
        max_agent_loss=10,
        core_symbols=set(),
        pending_symbols=set(),
    )

    assert len(result) == 1
    candidate = result[0]
    assert candidate["sentiment_summary"]["proposalOrigin"] == MICRO_PROPOSAL_ORIGIN
    assert candidate["provider_scores"]["micro_variation"]["tag"] == "MICRO"
