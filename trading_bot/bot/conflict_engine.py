from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConflictResult:
    conflict_flags: list[str]
    adjusted_confidence: float
    explanation_summary: str


class ConflictEngine:
    def evaluate(
        self,
        *,
        technical_score: float,
        breakout_quality: float,
        microstructure_score: float,
        portfolio_risk_score: float,
        exposure_score: float,
        drawdown_score: float,
        sentiment_score: float,
        attention_score: float,
        fresh_market_data: bool,
        price_move_pct: float,
    ) -> ConflictResult:
        flags: list[str] = []

        dominant_strength = (technical_score + breakout_quality + microstructure_score) / 3
        risk_pressure = (1 - portfolio_risk_score + 1 - exposure_score + 1 - drawdown_score) / 3

        if dominant_strength >= 0.67 and sentiment_score >= 0.72:
            flags.append("technical_bullish_vs_sentiment_euphoria")
        if dominant_strength >= 0.67 and risk_pressure >= 0.55:
            flags.append("strong_setup_vs_high_portfolio_exposure")
        if breakout_quality >= 0.75 and abs(price_move_pct) >= 1.5:
            flags.append("breakout_vs_overextension")
        if attention_score >= 0.75 and sentiment_score >= 0.70:
            flags.append("crowding_risk_attention_cluster")
        if not fresh_market_data:
            flags.append("stale_market_data")
        if abs(price_move_pct) >= 2.0:
            flags.append("proposal_invalidated_material_price_move")

        adjusted = 0.75 - (len(flags) * 0.09)
        adjusted = max(0.15, min(0.95, adjusted))
        if risk_pressure >= 0.7:
            adjusted = min(adjusted, 0.35)

        return ConflictResult(
            conflict_flags=flags,
            adjusted_confidence=round(adjusted, 4),
            explanation_summary=(
                f"dominant={dominant_strength:.2f}, risk_pressure={risk_pressure:.2f}, "
                f"sentiment={sentiment_score:.2f}, attention={attention_score:.2f}"
            ),
        )


def conflict_payload(result: ConflictResult) -> dict[str, Any]:
    return {
        "conflictFlags": result.conflict_flags,
        "adjustedConfidence": result.adjusted_confidence,
        "explanationSummary": result.explanation_summary,
    }
