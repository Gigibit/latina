from trading_bot.bot.service import generate_suggestion


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
        }


def test_generate_suggestion_with_mocks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    monkeypatch.setattr(
        "trading_bot.bot.service.build_symbol_corpus",
        lambda symbol: ["context 1", "context 2", "context 3", "context 4"],
    )
    monkeypatch.setattr("trading_bot.bot.service.EmbeddingRetriever", DummyRetriever)
    monkeypatch.setattr("trading_bot.bot.service.LLMDecider", DummyDecider)

    result = generate_suggestion("AAPL", "medium")

    assert result["symbol"] == "AAPL"
    assert result["provider"] == "openai"
    assert result["decision"]["action"] == "HOLD"
    assert len(result["selected_context"]) == 4
