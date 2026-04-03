from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class EtoroAdapter:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        ws_url: str | None = None,
        user_key: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.ws_url = (ws_url or "").strip()
        self.user_key = (user_key or "").strip()

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        endpoint = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "x-request-id": str(uuid.uuid4()),
        }
        if self.user_key:
            headers["x-user-key"] = self.user_key
        request = Request(
            endpoint,
            data=payload,
            method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()
            message = (
                "eToro API request failed "
                f"status={exc.code} method={method} url={endpoint} "
                f"reason={exc.reason} body={error_body or '<empty>'}"
            )
            logger.error(message)
            raise RuntimeError(message) from exc
        except URLError as exc:
            message = (
                "eToro API request failed "
                f"method={method} url={endpoint} reason={exc.reason}"
            )
            logger.error(message)
            raise RuntimeError(message) from exc

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

    def authenticateWebsocket(self) -> dict[str, Any]:
        if not self.ws_url:
            message = "eToro websocket authentication failed: ETORO_WS_URL is missing"
            logger.error(message)
            raise RuntimeError(message)
        if not self.user_key:
            message = "eToro websocket authentication failed: ETORO_USER_KEY is missing"
            logger.error(message)
            raise RuntimeError(message)

        payload = {
            "id": str(uuid.uuid4()),
            "operation": "Authenticate",
            "data": {"userKey": self.user_key, "apiKey": self.api_key},
        }

        try:
            from websocket import create_connection
        except ImportError as exc:
            message = (
                "eToro websocket authentication failed: websocket-client dependency is missing"
            )
            logger.error(message)
            raise RuntimeError(message) from exc

        try:
            with create_connection(self.ws_url, timeout=15) as socket:
                socket.send(json.dumps(payload))
                raw_response = socket.recv()
        except Exception as exc:
            message = (
                "eToro websocket authentication failed "
                f"url={self.ws_url} reason={exc}"
            )
            logger.error(message)
            raise RuntimeError(message) from exc

        try:
            response = json.loads(raw_response) if raw_response else {}
        except json.JSONDecodeError as exc:
            message = "eToro websocket authentication failed: invalid JSON response"
            logger.error("%s payload=%s", message, raw_response)
            raise RuntimeError(message) from exc

        success = bool(response.get("success"))
        operation = str(response.get("operation", ""))
        if not success or operation != "Authenticate":
            error_code = str(response.get("errorCode", "unknown"))
            error_message = str(response.get("errorMessage", "unknown"))
            message = (
                "eToro websocket authentication rejected "
                f"url={self.ws_url} errorCode={error_code} errorMessage={error_message}"
            )
            logger.error(message)
            raise RuntimeError(message)
        return response

    def streamMarketMonitor(self) -> None:
        if self.ws_url and self.user_key:
            self.authenticateWebsocket()
            logger.info("streamMarketMonitor websocket auth succeeded url=%s", self.ws_url)
            return
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
    base_url = os.getenv("ETORO_API_BASE_URL", "https://public-api.etoro.com/api/v1")
    if not api_key:
        raise ValueError("ETORO_API_KEY is missing")
    return EtoroAdapter(
        api_key=api_key,
        base_url=base_url,
        ws_url=os.getenv("ETORO_WS_URL"),
        user_key=os.getenv("ETORO_USER_KEY"),
    )
