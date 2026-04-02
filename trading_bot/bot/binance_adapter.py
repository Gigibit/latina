from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class BinanceAdapter:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self.base_url}{path}{query}",
            method="GET",
            headers={"X-MBX-APIKEY": self.api_key},
        )
        with urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def getAccountSummary(self) -> dict[str, Any]:
        return {"equity": 10000, "cash": 10000, "riskLevel": "medium", "drawdown": 0.0}

    def getPositions(self) -> list[dict[str, Any]]:
        payload = self._request("/api/v3/ticker/24hr")
        if not isinstance(payload, list):
            return []
        top = payload[:10]
        return [
            {
                "symbol": item.get("symbol"),
                "marketValue": float(item.get("quoteVolume", 0) or 0),
                "currentRate": float(item.get("lastPrice", 0) or 0),
                "spread_bps": 0.0,
            }
            for item in top
        ]

    def getOpenOrders(self) -> list[dict[str, Any]]:
        return []

    def getPortfolioHistory(self) -> list[dict[str, Any]]:
        return [{"equity": 10000}, {"equity": 10030}, {"equity": 10010}, {"equity": 10040}]

    def getWatchlists(self) -> list[dict[str, Any]]:
        return []

    def getCuratedLists(self) -> list[dict[str, Any]]:
        return []

    def getInstrumentFeedPosts(self, *, symbol: str, limit: int = 25) -> list[dict[str, Any]]:
        return [{"text": f"{symbol} market chatter"} for _ in range(min(limit, 5))]

    def getUserFeedPosts(self, *, limit: int = 25) -> list[dict[str, Any]]:
        return [{"text": "crypto narrative"} for _ in range(min(limit, 3))]

    def getOrderBookSnapshot(self, *, symbol: str, limit: int = 20) -> dict[str, Any]:
        payload = self._request("/api/v3/depth", {"symbol": symbol, "limit": limit})
        if not isinstance(payload, dict):
            return {}
        bids = payload.get("bids", [])
        asks = payload.get("asks", [])
        if not bids or not asks:
            return {}
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread_bps = 0.0 if best_bid <= 0 else ((best_ask - best_bid) / best_bid) * 10000
        return {"spread_bps": spread_bps, "imbalance": 0.5, "freshness": 1.0}

    def placeOrder(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "accepted", "venue": "binance", **kwargs}

    def reducePosition(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "accepted", "venue": "binance", **kwargs}

    def closePosition(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "accepted", "venue": "binance", **kwargs}

    def cancelOrderIfSupported(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "accepted", "venue": "binance", **kwargs}


def build_binance_adapter() -> BinanceAdapter:
    api_key = os.getenv("BINANCE_API_KEY", "")
    base_url = os.getenv("BINANCE_API_BASE_URL", "https://api.binance.com")
    return BinanceAdapter(api_key=api_key, base_url=base_url)
