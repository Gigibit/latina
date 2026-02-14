from trading_bot.bot.data_sources import MarketSnapshot
from trading_bot.bot.service import (
    _extract_symbols_from_message,
    _rank_weekly_candidates,
    generate_suggestion,
    generate_weekly_chat_suggestion,
)


class DummyRetriever:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def top_k(self, query: str, corpus: list[str], k: int = 3):
        return [type("Chunk", (), {"text": item, "score": 0.9})() for item in corpus[:k]]


class DummyDecider:
    def __init__(self, provider: str, model: str, api_key: str | None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def decide(self, prompt: str):
        return {
            "action": "HOLD",
            "confidence": 60,
            "reasoning": "test reasoning",
            "risk_notes": "test risk",
            "top_pick": "NVDA",
            "answer": "Prefer NVDA for next week with moderate risk.",
        }


def _patch_dependencies(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "trading_bot.bot.service.get_market_snapshot",
        lambda symbol, lookback_days=90: MarketSnapshot(
            symbol=symbol.upper(),
            latest_close=100.0,
            pct_change_5d=3.0 if symbol.upper() == "NVDA" else 1.0,
            pct_change_20d=8.0 if symbol.upper() == "NVDA" else 2.0,
            avg_volume_20d=1000.0,
            latest_volume=1500.0,
        ),
    )
    monkeypatch.setattr("trading_bot.bot.service.EmbeddingRetriever", DummyRetriever)
    monkeypatch.setattr("trading_bot.bot.service.LLMDecider", DummyDecider)


def test_generate_suggestion_with_mocks(monkeypatch):
    _patch_dependencies(monkeypatch)

    result = generate_suggestion("AAPL", "medium")

    assert result["symbol"] == "AAPL"
    assert result["provider"] == "openai"
    assert result["decision"]["action"] == "HOLD"
    assert len(result["selected_context"]) == 4


def test_extract_symbols_with_fallback(monkeypatch):
    monkeypatch.setenv("DEFAULT_CANDIDATES", "SPY,QQQ")
    assert _extract_symbols_from_message("what should I buy next week?") == ["SPY", "QQQ"]
    assert _extract_symbols_from_message("is nvda better than aapl?") == ["AAPL", "NVDA"]


def test_rank_weekly_candidates(monkeypatch):
    _patch_dependencies(monkeypatch)

    ranked = _rank_weekly_candidates(["AAPL", "NVDA"])

    assert ranked[0]["symbol"] == "NVDA"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_generate_weekly_chat_suggestion(monkeypatch):
    _patch_dependencies(monkeypatch)

    result = generate_weekly_chat_suggestion(
        "what suggestion do you have for the next week?",
        "medium",
    )

    assert result["decision"]["top_pick"] == "NVDA"
    assert result["ranked_candidates"][0]["symbol"] == "NVDA"
    assert "Prefer NVDA" in result["answer"]
