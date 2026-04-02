import pandas as pd

from trading_bot.bot.symbol_engine import SymbolSuggestionEngine, rank_symbols


def _build_candles(*, start=100.0, drift=0.15, vol=1.2, spike_at=None, spike_size=18.0):
    rows = []
    close = start
    for i in range(260):
        close = close + drift + ((i % 5) - 2) * 0.05
        if spike_at is not None and i == spike_at:
            close *= 1 + (spike_size / 100)
        open_price = close - 0.3
        high_price = close + vol
        low_price = close - vol
        volume = 1000 + (i % 20) * 30
        if spike_at is not None and i == spike_at:
            volume *= 4
        rows.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close,
                "Volume": volume,
            }
        )
    return pd.DataFrame(rows)


def test_engine_detects_early_breakout_with_energy_buildup():
    engine = SymbolSuggestionEngine()

    base = _build_candles(drift=0.015, vol=0.35)
    # compress range on latest bars and then break to upside
    base.loc[220:255, "High"] = base.loc[220:255, "Close"] + 0.08
    base.loc[220:255, "Low"] = base.loc[220:255, "Close"] - 0.08
    base.loc[259, "Close"] = base.loc[258, "Close"] * 1.005
    base.loc[259, "High"] = base.loc[259, "Close"] + 0.2
    base.loc[259, "Low"] = base.loc[259, "Close"] - 0.2
    base.loc[259, "Volume"] = int(base.loc[220:258, "Volume"].mean() * 1.5)

    result = engine.analyze_symbol(symbol="CL=F", timeframe_data={"4h": base, "1d": base.copy()})

    assert result["symbol"] == "CL=F"
    timeframe = result["timeframes"][0]
    assert result["overall_score"] >= 45
    assert result["take_signal"] is True
    assert timeframe["breakout"]["label"] in {"confirmed breakout", "explosive breakout"}
    assert timeframe["meta"]["decision"] == "take"


def test_engine_downranks_event_driven_spike_signal():
    engine = SymbolSuggestionEngine()
    spiky = _build_candles(drift=0.03, vol=0.8, spike_at=259, spike_size=28.0)

    result = engine.analyze_symbol(symbol="BTC-USD", timeframe_data={"1h": spiky})
    timeframe = result["timeframes"][0]

    assert timeframe["regime"]["move_classification"] == "event-driven move"
    assert timeframe["meta"]["decision"] == "ignore"
    assert result["take_signal"] is False


def test_rank_symbols_orders_best_opportunities_first():
    engine = SymbolSuggestionEngine()

    trending = _build_candles(drift=0.1, vol=0.7)
    sideways = _build_candles(drift=0.0, vol=0.25)
    sideways["Close"] = 100 + ((sideways.index % 6) - 3) * 0.3
    sideways["Open"] = sideways["Close"] - 0.05
    sideways["High"] = sideways["Close"] + 0.15
    sideways["Low"] = sideways["Close"] - 0.15

    ranked = rank_symbols(
        engine,
        {
            "AAPL": {"1d": trending},
            "EURUSD": {"1d": sideways},
        },
    )

    assert len(ranked) == 2
    assert ranked[0]["overall_score"] >= ranked[1]["overall_score"]
