from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class EtoroAdapter:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def getAccountSummary(self) -> dict[str, Any]:
        return self._request("GET", "/account/summary")

    def getPositions(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/positions")
        return payload.get("positions", payload if isinstance(payload, list) else [])

    def getOpenOrders(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/orders/open")
        return payload.get("orders", payload if isinstance(payload, list) else [])

    def getPortfolioHistory(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/portfolio/history")
        return payload.get("history", payload if isinstance(payload, list) else [])

    def getWatchlists(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/watchlists")
        return payload.get("watchlists", payload if isinstance(payload, list) else [])

    def getCuratedLists(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/lists/curated")
        return payload.get("lists", payload if isinstance(payload, list) else [])

    def getInstrumentFeedPosts(self, *, symbol: str, limit: int = 25) -> list[dict[str, Any]]:
        payload = self._request("GET", f"/feeds/instruments/{symbol}?limit={limit}")
        return payload.get("posts", payload if isinstance(payload, list) else [])

    def getUserFeedPosts(self, *, limit: int = 25) -> list[dict[str, Any]]:
        payload = self._request("GET", f"/feeds/users/me?limit={limit}")
        return payload.get("posts", payload if isinstance(payload, list) else [])

    def getSocialAnalytics(self, *, symbol: str) -> dict[str, Any]:
        return self._request("GET", f"/social/analytics/{symbol}")

    def getAgentPortfolioCompatibility(self) -> dict[str, Any]:
        return self._request("GET", "/agent-portfolios/compatibility")

    def streamMarketMonitor(self) -> None:
        logger.info("streamMarketMonitor invoked in polling fallback mode")

    def placeOrder(
        self,
        *,
        approvedProposalId: str,
        approvalTimestamp: str,
        snapshotHash: str,
        symbol: str,
        side: str,
        size: float,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/orders",
            {
                "approvedProposalId": approvedProposalId,
                "approvalTimestamp": approvalTimestamp,
                "snapshotHash": snapshotHash,
                "symbol": symbol,
                "side": side,
                "size": size,
            },
        )

    def closePosition(
        self, *, approvedProposalId: str, approvalTimestamp: str, snapshotHash: str, symbol: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/positions/close",
            {
                "approvedProposalId": approvedProposalId,
                "approvalTimestamp": approvalTimestamp,
                "snapshotHash": snapshotHash,
                "symbol": symbol,
            },
        )

    def reducePosition(
        self,
        *,
        approvedProposalId: str,
        approvalTimestamp: str,
        snapshotHash: str,
        symbol: str,
        size: float,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/positions/reduce",
            {
                "approvedProposalId": approvedProposalId,
                "approvalTimestamp": approvalTimestamp,
                "snapshotHash": snapshotHash,
                "symbol": symbol,
                "size": size,
            },
        )

    def cancelOrderIfSupported(
        self,
        *,
        approvedProposalId: str,
        approvalTimestamp: str,
        snapshotHash: str,
        orderId: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/orders/cancel",
            {
                "approvedProposalId": approvedProposalId,
                "approvalTimestamp": approvalTimestamp,
                "snapshotHash": snapshotHash,
                "orderId": orderId,
            },
        )


def build_etoro_adapter() -> EtoroAdapter:
    api_key = os.getenv("ETORO_API_KEY")
    base_url = os.getenv("ETORO_API_BASE_URL", "https://api.etoro.com")
    if not api_key:
        raise ValueError("ETORO_API_KEY is missing")
    return EtoroAdapter(api_key=api_key, base_url=base_url)
