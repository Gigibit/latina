import json

from django.test import RequestFactory

from trading_bot.bot import views


class DummyRuntime:
    def __init__(self):
        self._payload = {
            "sessionId": "sid-1",
            "status": "RUNNING",
            "recentLogs": [{"message": "ok"}],
            "portfolioSummary": {"cash": 100},
            "cash": 100,
            "openPositions": 2,
            "unrealizedPnL": 1,
            "realizedPnL": 2,
            "drawdown": 0.1,
            "riskLevel": "medium",
            "heartbeatAt": None,
            "workerHealthy": True,
            "latestSyncAt": None,
            "latestAnalysisAt": None,
            "lastError": "",
        }

    def start(self):
        class S:
            session_id = "sid-1"

        return S()

    def stop(self):
        return None

    def session_payload(self):
        return self._payload

    def proposals_payload(self):
        return [{"proposalId": "p1", "status": "pending_user"}]

    def approve(self, proposal_id: str):
        return {"proposalId": proposal_id, "status": "approved"}

    def reject(self, proposal_id: str):
        return {"proposalId": proposal_id, "status": "rejected"}


def test_agent_session_view(monkeypatch):
    monkeypatch.setattr(views, "agent_runtime", DummyRuntime())
    request = RequestFactory().get("/api/agent/session")
    response = views.agent_session_view(request)
    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["sessionId"] == "sid-1"


def test_agent_approve_view(monkeypatch):
    monkeypatch.setattr(views, "agent_runtime", DummyRuntime())
    request = RequestFactory().post("/api/agent/proposals/p1/approve")
    response = views.agent_approve_view(request, proposal_id="p1")
    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["status"] == "approved"


def test_crypto_agent_session_view(monkeypatch):
    monkeypatch.setattr(views, "crypto_agent_runtime", DummyRuntime())
    request = RequestFactory().get("/api/agent-crypto/session")
    response = views.crypto_agent_session_view(request)
    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["sessionId"] == "sid-1"


def test_agent_proposals_view_logs_service_and_count(monkeypatch, caplog):
    monkeypatch.setattr(views, "agent_runtime", DummyRuntime())
    caplog.set_level("INFO", logger="trading_bot.bot.views")

    request = RequestFactory().get("/api/agent/proposals")
    response = views.agent_proposals_view(request)

    assert response.status_code == 200
    assert "Request agent_proposals_view service=agent_runtime" in caplog.text
    assert (
        "Response agent_proposals_view status=200 service=agent_runtime "
        "session_id=None proposals_count=1"
    ) in caplog.text
