from __future__ import annotations

from typing import Any


class AttentionEngine:
    def extract(
        self,
        *,
        symbol: str,
        feeds: list[dict[str, Any]],
        watchlists: list[dict[str, Any]],
        curated_lists: list[dict[str, Any]],
    ) -> dict[str, float]:
        lowered_symbol = symbol.upper()
        feed_mentions = sum(
            1 for item in feeds if lowered_symbol in str(item.get("text", "")).upper()
        )
        watchlist_mentions = sum(
            1
            for watchlist in watchlists
            for value in watchlist.get("symbols", [])
            if str(value).upper() == lowered_symbol
        )
        curated_mentions = sum(
            1
            for entry in curated_lists
            for value in entry.get("symbols", [])
            if str(value).upper() == lowered_symbol
        )
        total_attention = feed_mentions + watchlist_mentions + curated_mentions
        attention_score = min(1.0, total_attention / 25)
        crowding_score = min(1.0, (watchlist_mentions + curated_mentions) / 15)
        narrative_velocity = min(1.0, feed_mentions / 12)
        return {
            "attentionScore": round(attention_score, 4),
            "crowdingScore": round(crowding_score, 4),
            "narrativeVelocity": round(narrative_velocity, 4),
        }
