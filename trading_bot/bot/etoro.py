from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _is_env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _state_file_path() -> Path:
    configured_path = os.getenv("ETORO_AUTOTRADE_STATE_FILE", ".etoro_autotrade_state.json")
    return Path(configured_path)


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
    if not base_url or not api_key:
        return {
            "status": "error",
            "reason": "Missing ETORO_API_BASE_URL or ETORO_API_KEY",
        }

    endpoint = f"{base_url}/orders"
    payload = {
        "symbol": symbol.upper(),
        "side": normalized_action,
        "order_type": "market",
        "account_id": os.getenv("ETORO_ACCOUNT_ID"),
        "confidence": confidence,
        "source": "ai-trading-bot",
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=15) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        return {"status": "error", "reason": f"eToro API HTTP error {exc.code}"}
    except URLError as exc:
        return {"status": "error", "reason": f"eToro API connection error: {exc.reason}"}

    _record_execution(symbol, normalized_action)
    return {
        "status": "executed",
        "endpoint": endpoint,
        "response": response_body,
    }
