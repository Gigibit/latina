from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketSnapshot:
    symbol: str
    latest_close: float
    pct_change_5d: float
    pct_change_20d: float
    avg_volume_20d: float
    latest_volume: float


def get_market_snapshot(symbol: str, lookback_days: int = 90) -> MarketSnapshot:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install dependencies from requirements.txt"
        ) from exc

    ticker = yf.Ticker(symbol)
    history = ticker.history(period=f"{lookback_days}d", interval="1d").dropna()
    if history.empty or len(history) < 21:
        raise ValueError(f"Not enough data found for symbol '{symbol}'.")

    closes = history["Close"]
    volumes = history["Volume"]

    latest_close = float(closes.iloc[-1])
    pct_change_5d = float((closes.iloc[-1] / closes.iloc[-6] - 1) * 100)
    pct_change_20d = float((closes.iloc[-1] / closes.iloc[-21] - 1) * 100)

    avg_volume_20d = float(volumes.tail(20).mean())
    latest_volume = float(volumes.iloc[-1])

    return MarketSnapshot(
        symbol=symbol.upper(),
        latest_close=latest_close,
        pct_change_5d=pct_change_5d,
        pct_change_20d=pct_change_20d,
        avg_volume_20d=avg_volume_20d,
        latest_volume=latest_volume,
    )
