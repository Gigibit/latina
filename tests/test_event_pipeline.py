from trading_bot.bot.event_pipeline import FeatureSnapshotBuilder, FusionEngine, LLMReasoningEngine


class DummyProviderResult:
    def __init__(self, raw_inputs):
        self.raw_inputs = raw_inputs


def _provider_results():
    return {
        "technical": DummyProviderResult(
            {"trend": "up", "breakout": True, "ma_alignment": "bullish"}
        ),
        "portfolio_risk": DummyProviderResult({"risk_state": "normal"}),
        "market_regime": DummyProviderResult({"regime": "risk_on", "vol_regime": "low"}),
        "social_sentiment": DummyProviderResult({"narrative_velocity": 0.5}),
        "watchlist_interest": DummyProviderResult({"attention_score": 0.7}),
        "curated_interest": DummyProviderResult({"crowding_score": 0.3}),
    }


def test_snapshot_builder_shape():
    builder = FeatureSnapshotBuilder(stale_after_ms=1000)
    snapshot = builder.build(
        symbol="AAPL",
        event_id="evt-1",
        provider_results=_provider_results(),
        fused={"priceMovePct": 0.4},
        account={"drawdown": 0.2},
        positions=[{"symbol": "AAPL", "marketValue": 1000, "quantity": 4}],
    )
    assert snapshot["symbol"] == "AAPL"
    assert "technical" in snapshot
    assert "microstructure" in snapshot
    assert "freshness" in snapshot


def test_fusion_engine_risk_veto():
    fusion = FusionEngine()
    result = fusion.combine(
        base_score=0.8,
        llm_result={"confidence_adjustment": 0.2, "uncertainty": "low", "conflicts": []},
        risk_veto=True,
        is_stale=False,
    )
    assert result["proposalScore"] == 0.0


def test_llm_reasoner_neutral_on_noise(monkeypatch):
    engine = LLMReasoningEngine(api_key="test")
    snapshot = {
        "microstructure": {"shortTermVol": 0.0},
    }
    result = engine.interpret(symbol="AAPL", snapshot=snapshot, deterministic_score=0.5)
    assert result["confidence_adjustment"] == 0.0
    assert result["action_bias"] == "neutral"
