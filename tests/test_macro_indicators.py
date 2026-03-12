import pytest

from trading_bot.bot.data_sources import fetch_macro_indicators


def test_fetch_macro_indicators_skips_unavailable_series(monkeypatch):
    values_by_code = {
        "CPIAUCSL": [100.0, 101.5],
        "UNRATE": RuntimeError("timeout"),
        "DGS10": [4.1, 4.2],
    }

    def fake_fetch(series_code: str):
        value = values_by_code[series_code]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("trading_bot.bot.data_sources._fetch_fred_series", fake_fetch)

    indicators = fetch_macro_indicators()

    assert [item.series for item in indicators] == ["US_CPI", "US_10Y_TREASURY"]
    assert indicators[0].delta == 1.5
    assert indicators[1].delta == pytest.approx(0.1)


def test_fetch_macro_indicators_skips_series_with_insufficient_points(monkeypatch):
    values_by_code = {
        "CPIAUCSL": [100.0],
        "UNRATE": [3.9, 4.0],
        "DGS10": [4.1],
    }

    def fake_fetch(series_code: str):
        return values_by_code[series_code]

    monkeypatch.setattr("trading_bot.bot.data_sources._fetch_fred_series", fake_fetch)

    indicators = fetch_macro_indicators()

    assert len(indicators) == 1
    assert indicators[0].series == "US_UNEMPLOYMENT"
    assert indicators[0].delta == pytest.approx(0.1)
