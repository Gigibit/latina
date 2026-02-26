from types import SimpleNamespace

import pytest

from trading_bot.bot.service import generate_suggestion, get_best_candidates, get_market_monitor


class DummyRetriever:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def top_k(self, query: str, corpus: list[str], k: int = 3):
        return [
            type("Chunk", (), {"text": item, "score": 0.9, "index": idx})()
            for idx, item in enumerate(corpus[:k])
        ]


class DummyDecider:
    def __init__(self, provider: str, model: str, api_key: str | None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def decide(self, prompt: str):
        return {
            "action": "BUY",
            "confidence": 74,
            "reasoning": "test reasoning",
            "risk_notes": "test risk",
        }

    def evaluate_trending_candidates(self, candidates):
        return {
            "AAA": {"llm_score": 30.0, "summary": "weaker momentum quality"},
            "BBB": {"llm_score": 90.0, "summary": "best balance of momentum/volume"},
        }


def test_generate_suggestion_with_mocks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("CANDLE_SIZE", "1d")
    monkeypatch.setenv("CANDLES_RAG_INFERENCE_NUMER", "10")

    monkeypatch.setattr(
        "trading_bot.bot.service._build_behavior_samples",
        lambda symbol, candle_size, rag_inference_number, rag_granularity_size=None: (
            [
                {"text": "context 1", "action": "BUY", "next_pct_change": 1.1},
                {"text": "context 2", "action": "SELL", "next_pct_change": -0.6},
                {"text": "context 3", "action": "BUY", "next_pct_change": 0.7},
                {"text": "context 4", "action": "BUY", "next_pct_change": 0.5},
            ],
            "query",
            {
                "enabled": True,
                "weight": 0.3,
                "avg_query_sentiment": 0.2,
                "weighted_avg_query_sentiment": 0.06,
            },
            [100.0, 101.0, 102.0],
        ),
    )
    monkeypatch.setattr(
        "trading_bot.bot.service.compute_technical_indicators",
        lambda symbol: SimpleNamespace(
            sma_20=120.0,
            sma_50=110.0,
            rsi_14=55.0,
            macd=1.2,
            macd_signal=0.8,
            bollinger_upper=130.0,
            bollinger_lower=100.0,
        ),
    )
    monkeypatch.setattr(
        "trading_bot.bot.service.fetch_fundamental_metrics",
        lambda symbol: SimpleNamespace(
            pe_ratio=20.0,
            eps=3.5,
            debt_to_equity=90.0,
            market_cap=1000000.0,
        ),
    )
    monkeypatch.setattr("trading_bot.bot.service.fetch_macro_indicators", lambda: [])
    monkeypatch.setattr("trading_bot.bot.service.average_macro_delta", lambda indicators: 0.0)
    monkeypatch.setattr("trading_bot.bot.service.FaissEmbeddingRetriever", DummyRetriever)
    monkeypatch.setattr("trading_bot.bot.service.LLMDecider", DummyDecider)
    monkeypatch.setattr(
        "trading_bot.bot.service.execute_etoro_action",
        lambda **kwargs: {"status": "skipped"},
    )

    result = generate_suggestion("AAPL", "medium")

    assert result["symbol"] == "AAPL"
    assert result["provider"] == "openai"
    assert result["decision"]["action"] == "BUY"
    assert len(result["selected_context"]) == 4
    assert result["buy_probability"] > 0.75
    assert result["x_sentiment"]["weight"] == 0.3
    assert result["technical_indicators"]["sma_20"] == 120.0
    assert result["etoro_execution"]["status"] == "skipped"


def test_get_best_candidates_ranks_by_score(monkeypatch):
    monkeypatch.setattr(
        "trading_bot.bot.service.fetch_trending_symbols",
        lambda region, limit: ["AAA", "BBB"] if region == "US" else ["BBB", "CCC"],
    )

    snapshots = {
        "AAA": type(
            "Snapshot",
            (),
            {
                "symbol": "AAA",
                "latest_close": 100.0,
                "pct_change_5d": 4.0,
                "pct_change_20d": 9.0,
                "avg_volume_20d": 100.0,
                "latest_volume": 120.0,
            },
        )(),
        "BBB": type(
            "Snapshot",
            (),
            {
                "symbol": "BBB",
                "latest_close": 50.0,
                "pct_change_5d": 6.0,
                "pct_change_20d": 11.0,
                "avg_volume_20d": 100.0,
                "latest_volume": 160.0,
            },
        )(),
        "CCC": type(
            "Snapshot",
            (),
            {
                "symbol": "CCC",
                "latest_close": 30.0,
                "pct_change_5d": -1.0,
                "pct_change_20d": 2.0,
                "avg_volume_20d": 100.0,
                "latest_volume": 90.0,
            },
        )(),
    }
    monkeypatch.setattr(
        "trading_bot.bot.service.get_market_snapshot",
        lambda symbol: snapshots[symbol],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr("trading_bot.bot.service.LLMDecider", DummyDecider)

    result = get_best_candidates(limit=2, user_risk_profile="medium")

    assert result["risk_profile"] == "medium"
    assert [candidate["symbol"] for candidate in result["candidates"]] == ["BBB", "AAA"]
    assert len(result["candidates"]) == 2
    assert result["source"] == "Yahoo Finance trending (US + EU)"
    assert result["ranking_strategy"] == "quant_score + openai_comparison"
    assert result["candidates"][0]["llm_score"] == 90.0
    assert result["candidates"][0]["llm_summary"] == "best balance of momentum/volume"
    assert result["candidates"][0]["combined_score"] >= result["candidates"][1]["combined_score"]


def test_get_market_monitor(monkeypatch):
    monkeypatch.setattr(
        "trading_bot.bot.service.fetch_market_news",
        lambda limit: [{"title": "Fed decision", "link": "https://example.com/news"}],
    )
    monkeypatch.setattr(
        "trading_bot.bot.service.fetch_macro_indicators",
        lambda: [
            SimpleNamespace(
                series="US_CPI",
                latest_value=310.0,
                previous_value=309.2,
                delta=0.8,
            )
        ],
    )

    result = get_market_monitor(limit=3)
    assert result["news_count"] == 1
    assert result["market_regime"] == "risk_off"
    assert result["alerts"]

def test_generate_suggestion_requires_granularity_for_introspective_mode(monkeypatch):
    monkeypatch.setenv("CHUNKIZATION_MODE", "INTROSPECTIVE_CANDLE")
    monkeypatch.delenv("CANDLE_RAG_GRANULARITY_SIZE", raising=False)

    with pytest.raises(ValueError, match="CANDLE_RAG_GRANULARITY_SIZE is mandatory"):
        generate_suggestion("AAPL", "medium")


def test_generate_suggestion_validates_granularity_is_less_than_candle_size(monkeypatch):
    monkeypatch.setenv("CHUNKIZATION_MODE", "INTROSPECTIVE_CANDLE")
    monkeypatch.setenv("CANDLE_SIZE", "1d")
    monkeypatch.setenv("CANDLE_RAG_GRANULARITY_SIZE", "1d")

    with pytest.raises(ValueError, match="must be strictly less than CANDLE_SIZE"):
        generate_suggestion("AAPL", "medium")


def test_get_best_candidates_skips_failed_region(monkeypatch):
    def fake_fetch(region, limit):
        if region == "EU":
            raise RuntimeError("Yahoo Finance returned no trending symbols.")
        return ["AAA", "BBB"]

    monkeypatch.setattr("trading_bot.bot.service.fetch_trending_symbols", fake_fetch)

    snapshots = {
        "AAA": type(
            "Snapshot",
            (),
            {
                "symbol": "AAA",
                "latest_close": 100.0,
                "pct_change_5d": 4.0,
                "pct_change_20d": 9.0,
                "avg_volume_20d": 100.0,
                "latest_volume": 120.0,
            },
        )(),
        "BBB": type(
            "Snapshot",
            (),
            {
                "symbol": "BBB",
                "latest_close": 50.0,
                "pct_change_5d": 6.0,
                "pct_change_20d": 11.0,
                "avg_volume_20d": 100.0,
                "latest_volume": 160.0,
            },
        )(),
    }
    monkeypatch.setattr(
        "trading_bot.bot.service.get_market_snapshot",
        lambda symbol: snapshots[symbol],
    )

    result = get_best_candidates(limit=2, user_risk_profile="medium")

    assert [candidate["symbol"] for candidate in result["candidates"]] == ["BBB", "AAA"]


def test_get_best_candidates_fails_when_all_regions_fail(monkeypatch):
    monkeypatch.setattr(
        "trading_bot.bot.service.fetch_trending_symbols",
        lambda region, limit: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="Unable to fetch trending symbols"):
        get_best_candidates(limit=2, user_risk_profile="medium")
