import json
from datetime import date

import pandas as pd
from django.test import RequestFactory

from trading_bot.bot.views import candle_playground_view


class DummyLLMDecider:
    def __init__(self, provider: str, model: str, api_key: str | None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def predict_next_candle_character(self, sequence_text: str) -> str:
        return "G"


def test_candle_playground_uses_earliest_available_when_requested_date_is_out_of_range(monkeypatch):
    history = pd.DataFrame(
        [
            {"Open": 100.0, "Close": 101.0},
            {"Open": 101.0, "Close": 99.0},
        ],
        index=pd.to_datetime(["2025-12-24 15:00:00", "2025-12-26 15:00:00"]),
    )
    monkeypatch.setattr("trading_bot.bot.views.get_candle_history", lambda **kwargs: history)
    monkeypatch.setattr("trading_bot.bot.views.LLMDecider", DummyLLMDecider)

    request = RequestFactory().get(
        "/api/playground/",
        {
            "symbol": "AAPL",
            "date": "2025-12-27",
            "size": "1h",
            "history_limit": "10000",
        },
    )

    response = candle_playground_view(request)

    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["symbol"] == "AAPL"
    assert payload["candles_count"] == 2
    assert payload["sequence"] == "G( probability=0.0099 ) R( probability=0.0198 )"


def test_candle_playground_rejects_future_date():
    tomorrow = date.today().fromordinal(date.today().toordinal() + 1).isoformat()
    request = RequestFactory().get(
        "/api/playground/",
        {
            "symbol": "AAPL",
            "date": tomorrow,
            "size": "1h",
            "history_limit": "10000",
        },
    )

    response = candle_playground_view(request)

    assert response.status_code == 400
    payload = json.loads(response.content)
    assert "future" in payload["error"].lower()


def test_candle_playground_disables_probability_when_possibility_env_is_false(monkeypatch):
    history = pd.DataFrame(
        [
            {"Open": 100.0, "Close": 101.0},
            {"Open": 101.0, "Close": 99.0},
        ],
        index=pd.to_datetime(["2025-12-24 15:00:00", "2025-12-26 15:00:00"]),
    )
    monkeypatch.setattr("trading_bot.bot.views.get_candle_history", lambda **kwargs: history)
    monkeypatch.setattr("trading_bot.bot.views.LLMDecider", DummyLLMDecider)
    monkeypatch.setenv("POSSIBILITY_ENABLED", "false")

    request = RequestFactory().get(
        "/api/playground/",
        {
            "symbol": "AAPL",
            "date": "2025-12-24",
            "size": "1h",
            "history_limit": "10000",
        },
    )

    response = candle_playground_view(request)

    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["sequence"] == "G R"
