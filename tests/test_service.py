from trading_bot.bot.service import generate_suggestion, get_best_candidates


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


def test_generate_suggestion_with_mocks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("CANDLE_SIZE", "1d")
    monkeypatch.setenv("CANDLES_RAG_INFERENCE_NUMER", "10")

    monkeypatch.setattr(
        "trading_bot.bot.service._build_behavior_samples",
        lambda symbol, candle_size, rag_inference_number: (
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
        ),
    )
    monkeypatch.setattr("trading_bot.bot.service.FaissEmbeddingRetriever", DummyRetriever)
    monkeypatch.setattr("trading_bot.bot.service.LLMDecider", DummyDecider)

    result = generate_suggestion("AAPL", "medium")

    assert result["symbol"] == "AAPL"
    assert result["provider"] == "openai"
    assert result["decision"]["action"] == "BUY"
    assert len(result["selected_context"]) == 4
    assert result["buy_probability"] == 0.75
    assert result["x_sentiment"]["weight"] == 0.3


def test_get_best_candidates_ranks_by_score(monkeypatch):
    monkeypatch.setattr(
        "trading_bot.bot.service.fetch_trending_symbols",
        lambda limit: ["AAA", "BBB", "CCC"],
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

    result = get_best_candidates(limit=2, user_risk_profile="medium")

    assert result["risk_profile"] == "medium"
    assert [candidate["symbol"] for candidate in result["candidates"]] == ["BBB", "AAA"]
    assert len(result["candidates"]) == 2
