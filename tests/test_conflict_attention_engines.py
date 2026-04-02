from trading_bot.bot.attention_engine import AttentionEngine
from trading_bot.bot.conflict_engine import ConflictEngine


def test_conflict_engine_flags_crowding_and_exposure():
    result = ConflictEngine().evaluate(
        technical_score=0.8,
        breakout_quality=0.8,
        microstructure_score=0.7,
        portfolio_risk_score=0.3,
        exposure_score=0.2,
        drawdown_score=0.2,
        sentiment_score=0.8,
        attention_score=0.8,
        fresh_market_data=True,
        price_move_pct=2.1,
    )
    assert "technical_bullish_vs_sentiment_euphoria" in result.conflict_flags
    assert "strong_setup_vs_high_portfolio_exposure" in result.conflict_flags
    assert result.adjusted_confidence <= 0.35


def test_attention_engine_extracts_scores():
    payload = AttentionEngine().extract(
        symbol="BTCUSDT",
        feeds=[{"text": "BTCUSDT momentum"}, {"text": "No mention"}],
        watchlists=[{"symbols": ["BTCUSDT", "ETHUSDT"]}],
        curated_lists=[{"symbols": ["BTCUSDT"]}],
    )
    assert payload["attentionScore"] > 0
    assert payload["crowdingScore"] > 0
