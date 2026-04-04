from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from trading_bot.bot.data_sources import get_trending_tickers_from_etoro

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
        self._latest_portfolio_info: dict[str, Any] | None = None
        self.supported_capabilities = {
            "supportsFeeds": False,
            "supportsSocialAnalytics": False,
            "supportsCuratedLists": False,
            "supportsWatchlists": bool(self.user_key),
            "supportsAgentPortfolios": False,
            "supportsMarketMonitorStreaming": bool(self.ws_url and self.user_key),
        }

    @staticmethod
    def _normalized_env() -> str:
        raw_env = os.getenv("ETORO_ENV", "REAL").strip().upper()
        if raw_env not in {"REAL", "DEMO"}:
            message = (
                "eToro adapter configuration invalid: ETORO_ENV must be REAL or DEMO "
                f"(got: {raw_env or '<empty>'})"
            )
            logger.error(message)
            raise RuntimeError(message)
        return raw_env

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request_body = payload.decode("utf-8") if payload is not None else "<empty>"
        endpoint = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self.api_key,
            "x-request-id": str(uuid.uuid4()),
            "User-Agent": "curl/8.7.1",
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
            raw_headers = exc.headers.items() if exc.headers else []
            response_headers = {key.lower(): value for key, value in raw_headers}
            content_type = response_headers.get("content-type", "")
            body_preview = (error_body[:500] + "...") if len(error_body) > 500 else error_body
            error_json: dict[str, Any] | None = None
            if "json" in content_type and error_body:
                try:
                    parsed_json = json.loads(error_body)
                    if isinstance(parsed_json, dict):
                        error_json = parsed_json
                except json.JSONDecodeError:
                    logger.error(
                        "eToro API error response is not valid JSON "
                        "status=%s method=%s url=%s content_type=%s body_preview=%s",
                        exc.code,
                        method,
                        endpoint,
                        content_type,
                        body_preview or "<empty>",
                    )
            message = (
                "eToro API request failed "
                f"status={exc.code} method={method} url={endpoint} "
                f"reason={exc.reason} body={error_body or '<empty>'}"
            )
            logger.error(
                "%s request_headers=%s request_body=%s "
                "response_headers=%s response_json=%s response_body_preview=%s",
                message,
                headers,
                request_body,
                response_headers or {},
                error_json,
                body_preview or "<empty>",
            )
            raise RuntimeError(message) from exc
        except URLError as exc:
            message = (
                "eToro API request failed "
                f"method={method} url={endpoint} reason={exc.reason}"
            )
            logger.error(
                "%s request_headers=%s request_body=%s",
                message,
                headers,
                request_body,
            )
            raise RuntimeError(message) from exc

    def getAccountSummary(self) -> dict[str, Any]:
        env_segment = "demo" if self._normalized_env() == "DEMO" else "real"
        payload = self._request("GET", f"/trading/info/{env_segment}/pnl")
        portfolio = payload.get("clientPortfolio", payload) if isinstance(payload, dict) else {}
        self._latest_portfolio_info = portfolio if isinstance(portfolio, dict) else {}
        return self._latest_portfolio_info

    def _portfolio_info(self) -> dict[str, Any]:
        if self._latest_portfolio_info is None:
            return self.getAccountSummary()
        return self._latest_portfolio_info

    def getPositions(self) -> list[dict[str, Any]]:
        portfolio = self._portfolio_info()
        positions = portfolio.get("positions", [])
        return positions if isinstance(positions, list) else []

    def getOpenOrders(self) -> list[dict[str, Any]]:
        portfolio = self._portfolio_info()
        orders = portfolio.get("orders", [])
        return orders if isinstance(orders, list) else []

    def getPortfolioHistory(self) -> list[dict[str, Any]]:
        portfolio = self._portfolio_info()
        history = portfolio.get("portfolioHistory", [])
        return history if isinstance(history, list) else []

    def getWatchlists(self) -> list[dict[str, Any]]:
        try:
            symbols = get_trending_tickers_from_etoro(limit=100)
        except Exception as exc:
            logger.error(
                "eToro watchlists fetch failed: unable to resolve watchlist symbols error=%s",
                exc,
            )
            return []
        return [{"name": "etoro_watchlists", "symbols": symbols}]

    def getCuratedLists(self) -> list[dict[str, Any]]:
        logger.error(
            "eToro curated lists fetch skipped: endpoint not enabled because only documented and "
            "explicitly implemented routes are allowed."
        )
        return []

    def getInstrumentFeedPosts(self, *, symbol: str, limit: int = 25) -> list[dict[str, Any]]:
        logger.error(
            "eToro instrument feed fetch skipped symbol=%s limit=%s: endpoint not enabled because "
            "only documented and explicitly implemented routes are allowed.",
            symbol,
            limit,
        )
        return []

    def getUserFeedPosts(self, *, limit: int = 25) -> list[dict[str, Any]]:
        logger.error(
            "eToro user feed fetch skipped limit=%s: endpoint not enabled because only documented "
            "and explicitly implemented routes are allowed.",
            limit,
        )
        return []

    def getSocialAnalytics(self, *, symbol: str) -> dict[str, Any]:
        logger.error(
            "eToro social analytics fetch skipped symbol=%s: endpoint not enabled because only "
            "documented and explicitly implemented routes are allowed.",
            symbol,
        )
        return {}

    def getAgentPortfolioCompatibility(self) -> dict[str, Any]:
        logger.error(
            "eToro agent portfolio compatibility fetch skipped: endpoint not enabled because only "
            "documented and explicitly implemented routes are allowed."
        )
        return {}

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
        env_segment = "demo" if self._normalized_env() == "DEMO" else "real"
        instrument_id = os.getenv("ETORO_INSTRUMENT_ID", "").strip()
        if not instrument_id:
            message = "eToro placeOrder failed: ETORO_INSTRUMENT_ID is missing"
            logger.error(message)
            raise RuntimeError(message)
        return self._request(
            "POST",
            f"/trading/execution/{env_segment}/market-open-orders/by-amount",
            {
                "InstrumentID": int(instrument_id),
                "IsBuy": side.upper() == "BUY",
                "Leverage": int(os.getenv("ETORO_LEVERAGE", "1")),
                "Amount": float(size),
                "CID": int(os.getenv("ETORO_CID", "0")),
            },
        )

    def closePosition(
        self, *, approvedProposalId: str, approvalTimestamp: str, snapshotHash: str, symbol: str
    ) -> dict[str, Any]:
        message = (
            "eToro closePosition is not executed: only documented API routes are enabled "
            "and this operation is currently unsupported in this app."
        )
        logger.error(
            "%s symbol=%s approvedProposalId=%s approvalTimestamp=%s snapshotHash=%s",
            message,
            symbol,
            approvedProposalId,
            approvalTimestamp,
            snapshotHash,
        )
        return {"status": "unsupported", "reason": message}

    def reducePosition(
        self,
        *,
        approvedProposalId: str,
        approvalTimestamp: str,
        snapshotHash: str,
        symbol: str,
        size: float,
    ) -> dict[str, Any]:
        message = (
            "eToro reducePosition is not executed: only documented API routes are enabled "
            "and this operation is currently unsupported in this app."
        )
        logger.error(
            "%s symbol=%s size=%s approvedProposalId=%s approvalTimestamp=%s snapshotHash=%s",
            message,
            symbol,
            size,
            approvedProposalId,
            approvalTimestamp,
            snapshotHash,
        )
        return {"status": "unsupported", "reason": message}

    def cancelOrderIfSupported(
        self,
        *,
        approvedProposalId: str,
        approvalTimestamp: str,
        snapshotHash: str,
        orderId: str,
    ) -> dict[str, Any]:
        message = (
            "eToro cancelOrderIfSupported is not executed: only documented API routes are enabled "
            "and this operation is currently unsupported in this app."
        )
        logger.error(
            "%s orderId=%s approvedProposalId=%s approvalTimestamp=%s snapshotHash=%s",
            message,
            orderId,
            approvedProposalId,
            approvalTimestamp,
            snapshotHash,
        )
        return {"status": "unsupported", "reason": message}


def build_etoro_adapter() -> EtoroAdapter:
    api_key = os.getenv("ETORO_API_KEY")
    base_url = os.getenv("ETORO_API_BASE_URL", "https://public-api.etoro.com/api/v1").strip()
    user_key = os.getenv("ETORO_USER_KEY", "").strip()
    if not api_key:
        logger.error("eToro adapter configuration invalid: ETORO_API_KEY is missing")
        raise ValueError("ETORO_API_KEY is missing")
    if "public-api.etoro.com" in base_url and not user_key:
        message = (
            "eToro adapter configuration invalid: ETORO_USER_KEY is required when "
            "ETORO_API_BASE_URL points to public-api.etoro.com"
        )
        logger.error(message)
        raise ValueError(message)
    return EtoroAdapter(
        api_key=api_key,
        base_url=base_url,
        ws_url=os.getenv("ETORO_WS_URL"),
        user_key=user_key,
    )
