from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _is_env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _state_file_path() -> Path:
    configured_path = os.getenv("ETORO_AUTOTRADE_STATE_FILE", ".etoro_autotrade_state.json")
    return Path(configured_path)


def _http_log_enabled() -> bool:
    return _is_env_flag_enabled("HTTP_LOG_ENABLED", True)


def _normalized_etoro_env() -> str:
    raw_env = os.getenv("ETORO_ENV", "REAL").strip().upper()
    if raw_env not in {"REAL", "DEMO"}:
        message = (
            "Invalid ETORO_ENV value: expected REAL or DEMO "
            f"(got {raw_env or '<empty>'})"
        )
        logger.error(message)
        raise ValueError(message)
    return raw_env


def _load_state() -> dict:
    path = _state_file_path()
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_state(payload: dict) -> None:
    path = _state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _can_execute(symbol: str, action: str) -> tuple[bool, str | None, float]:
    wait_seconds = max(0.0, float(os.getenv("ETORO_NEXT_SUGGESTION_WAIT_SECONDS", "0")))
    if wait_seconds == 0:
        return True, None, 0.0

    state = _load_state()
    state_key = f"{symbol.upper()}:{action.upper()}"
    now = datetime.now(tz=timezone.utc).timestamp()
    last_timestamp = float(state.get(state_key, 0.0))
    elapsed = now - last_timestamp

    if elapsed >= wait_seconds:
        return True, None, wait_seconds

    return False, state_key, wait_seconds - elapsed


def _record_execution(symbol: str, action: str) -> None:
    state = _load_state()
    state_key = f"{symbol.upper()}:{action.upper()}"
    state[state_key] = datetime.now(tz=timezone.utc).timestamp()
    _save_state(state)


def execute_etoro_action(symbol: str, action: str, confidence: int | float | None = None) -> dict:
    normalized_action = action.upper()
    logger.debug(
        "Evaluating this symbol for eToro execution: symbol=%s action=%s",
        symbol.upper(),
        normalized_action,
    )
    if not _is_env_flag_enabled("ETORO_AUTOTRADE_ENABLED", False):
        return {"status": "disabled", "reason": "ETORO_AUTOTRADE_ENABLED is false"}

    action_flags = {
        "BUY": _is_env_flag_enabled("ETORO_ENABLE_BUY_ACTION", True),
        "SELL": _is_env_flag_enabled("ETORO_ENABLE_SELL_ACTION", True),
        "HOLD": _is_env_flag_enabled("ETORO_ENABLE_HOLD_ACTION", True),
    }
    if normalized_action not in action_flags:
        return {"status": "skipped", "reason": f"Unsupported action: {normalized_action}"}
    if not action_flags[normalized_action]:
        return {
            "status": "skipped",
            "reason": f"{normalized_action} action is disabled by environment",
        }

    can_execute, state_key, cooldown_seconds = _can_execute(symbol, normalized_action)
    if not can_execute:
        return {
            "status": "cooldown",
            "reason": "Waiting for next suggestion execution window",
            "retry_after_seconds": round(cooldown_seconds, 2),
            "state_key": state_key,
        }

    if normalized_action == "HOLD":
        _record_execution(symbol, normalized_action)
        return {"status": "skipped", "reason": "HOLD action enabled: no order sent"}

    base_url = os.getenv("ETORO_API_BASE_URL", "").rstrip("/")
    api_key = os.getenv("ETORO_API_KEY")
    user_key = os.getenv("ETORO_USER_KEY", "").strip()
    instrument_id = os.getenv("ETORO_INSTRUMENT_ID", "").strip()
    if not base_url or not api_key or not user_key or not instrument_id:
        logger.error(
            "eToro execution failed for symbol=%s action=%s: "
            "missing one of ETORO_API_BASE_URL, ETORO_API_KEY, ETORO_USER_KEY, ETORO_INSTRUMENT_ID",
            symbol.upper(),
            normalized_action,
        )
        return {
            "status": "error",
            "reason": (
                "Missing ETORO_API_BASE_URL, ETORO_API_KEY, ETORO_USER_KEY, "
                "or ETORO_INSTRUMENT_ID"
            ),
        }

    try:
        etoro_env = _normalized_etoro_env()
    except ValueError as exc:
        return {"status": "error", "reason": str(exc)}

    execution_path = (
        "/trading/execution/demo/market-open-orders/by-amount"
        if etoro_env == "DEMO"
        else "/trading/execution/market-open-orders/by-amount"
    )
    endpoint = f"{base_url}{execution_path}"
    payload = {
        "InstrumentID": int(instrument_id),
        "IsBuy": normalized_action == "BUY",
        "Leverage": int(os.getenv("ETORO_LEVERAGE", "1")),
        "Amount": float(os.getenv("ETORO_ORDER_AMOUNT_USD", "100")),
        "CID": int(os.getenv("ETORO_CID", "0")),
    }
    if confidence is not None and _http_log_enabled():
        logger.debug("eToro decision confidence=%s for symbol=%s", confidence, symbol.upper())

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "x-user-key": user_key,
            "x-request-id": str(uuid.uuid4()),
        },
        method="POST",
    )

    logger.debug("Calling this service for eToro order execution: %s", endpoint)
    if _http_log_enabled():
        logger.debug("eToro request body: %s", payload)

    try:
        with urlopen(request, timeout=15) as response:
            response_body = response.read().decode("utf-8")
        if _http_log_enabled():
            logger.debug("eToro response body: %s", response_body)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        if error_body and _http_log_enabled():
            logger.debug("eToro error response body: %s", error_body)
        logger.error(
            "eToro API HTTP error for symbol=%s action=%s: status=%s body=%s",
            symbol.upper(),
            normalized_action,
            exc.code,
            error_body or "<empty>",
        )
        return {"status": "error", "reason": f"eToro API HTTP error {exc.code}"}
    except URLError as exc:
        logger.error(
            "eToro API connection error for symbol=%s action=%s: %s",
            symbol.upper(),
            normalized_action,
            exc.reason,
        )
        return {"status": "error", "reason": f"eToro API connection error: {exc.reason}"}

    _record_execution(symbol, normalized_action)
    return {
        "status": "executed",
        "endpoint": endpoint,
        "response": response_body,
    }
