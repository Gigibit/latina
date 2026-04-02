from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass(frozen=True)
class CryptoMicroVariationConfig:
    proposal_ttl_ms: int = 20000
    max_spread_bps: float = 18.0
    max_volatility_norm: float = 0.045


class CryptoMicroVariationProposalEngine:
    def __init__(self, config: CryptoMicroVariationConfig | None = None) -> None:
        self._config = config or CryptoMicroVariationConfig()

    def evaluate(
        self,
        *,
        symbol: str,
        order_book: dict[str, Any],
        volatility_norm: float,
        now,
    ) -> dict[str, Any] | None:
        spread = float(order_book.get("spread_bps", 999) or 999)
        if spread > self._config.max_spread_bps:
            return None
        if volatility_norm > self._config.max_volatility_norm:
            return None
        return {
            "symbol": symbol,
            "tag": "MICRO",
            "expires_at": now + timedelta(milliseconds=self._config.proposal_ttl_ms),
            "metrics": {
                "spread_bps": round(spread, 4),
                "volatility_norm": round(volatility_norm, 6),
            },
        }
