from __future__ import annotations

import uuid

from django.db import models


class AgentSession(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    status = models.CharField(max_length=32)
    started_at = models.DateTimeField(null=True, blank=True)
    stopped_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    latest_sync_at = models.DateTimeField(null=True, blank=True)
    latest_analysis_at = models.DateTimeField(null=True, blank=True)
    worker_healthy = models.BooleanField(default=False)
    last_error = models.TextField(blank=True)
    metadata = models.JSONField(default=dict)
    capability_registry = models.JSONField(default=dict)
    streaming_connected = models.BooleanField(default=False)
    last_feed_sync_at = models.DateTimeField(null=True, blank=True)
    last_watchlist_sync_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class AgentProposal(models.Model):
    proposal_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    session = models.ForeignKey(AgentSession, on_delete=models.CASCADE, related_name="proposals")
    symbol = models.CharField(max_length=16)
    action = models.CharField(max_length=32)
    size = models.FloatField(default=0.0)
    rationale = models.TextField()
    risk_summary = models.TextField()
    explanation = models.TextField()
    confidence = models.FloatField(default=0.0)
    expected_impact = models.TextField()
    why_now = models.TextField(default="")
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    status = models.CharField(max_length=24, default="drafted")
    snapshot_hash = models.CharField(max_length=128)
    provider_scores = models.JSONField(default=dict)
    signal_freshness = models.JSONField(default=dict)
    conflict_flags = models.JSONField(default=list)
    invalidation_reason = models.TextField(blank=True)
    sentiment_summary = models.JSONField(default=dict)


class AgentApproval(models.Model):
    proposal = models.ForeignKey(AgentProposal, on_delete=models.CASCADE, related_name="approvals")
    approved = models.BooleanField()
    approved_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict)


class AgentLog(models.Model):
    session = models.ForeignKey(AgentSession, on_delete=models.CASCADE, related_name="logs")
    proposal = models.ForeignKey(AgentProposal, on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    level = models.CharField(max_length=16)
    category = models.CharField(max_length=32)
    symbol = models.CharField(max_length=16, blank=True)
    message = models.TextField()
    payload = models.JSONField(default=dict)
    ui_line = models.TextField(default="")


class PortfolioSnapshot(models.Model):
    session = models.ForeignKey(AgentSession, on_delete=models.CASCADE, related_name="snapshots")
    timestamp = models.DateTimeField(auto_now_add=True)
    account_summary = models.JSONField(default=dict)
    positions = models.JSONField(default=list)
    open_orders = models.JSONField(default=list)
    portfolio_history = models.JSONField(default=list)
    snapshot_hash = models.CharField(max_length=128)


class ExecutionEvent(models.Model):
    session = models.ForeignKey(
        AgentSession,
        on_delete=models.CASCADE,
        related_name="execution_events",
    )
    proposal = models.ForeignKey(AgentProposal, on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    event_type = models.CharField(max_length=32)
    status = models.CharField(max_length=24)
    payload = models.JSONField(default=dict)
