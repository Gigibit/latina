import logging

from django.test import Client


def _client() -> Client:
    return Client(HTTP_HOST="localhost")


def test_dashboard_view_logs_invalid_auto_detection_value(monkeypatch, caplog):
    client = _client()
    monkeypatch.setenv("AUTO_DETECTION_SYMBOL_NUMBER", "invalid")
    caplog.set_level(logging.ERROR, logger="trading_bot.bot.views")

    response = client.get("/")

    assert response.status_code == 200
    assert "Invalid AUTO_DETECTION_SYMBOL_NUMBER=invalid. Falling back to 7." in caplog.text


def test_trading_suggestion_view_logs_unknown_session_errors(caplog):
    client = _client()
    caplog.set_level(logging.ERROR, logger="trading_bot.bot.views")

    response = client.get("/api/suggestion/", {"status": "true", "session_id": "missing"})

    assert response.status_code == 404
    assert "reason=unknown_research_session session_id=missing" in caplog.text


def test_best_projection_view_logs_errors(monkeypatch, caplog):
    client = _client()
    caplog.set_level(logging.ERROR, logger="trading_bot.bot.views")
    monkeypatch.setattr(
        "trading_bot.bot.views.get_best_candidates",
        lambda limit, user_risk_profile: {"candidates": []},
    )

    response = client.get("/api/projections/", {"limit": 5, "risk": "medium"})

    assert response.status_code == 400
    assert (
        "Response best_projection_view status=400 error=No candidates available for projection."
        in caplog.text
    )
