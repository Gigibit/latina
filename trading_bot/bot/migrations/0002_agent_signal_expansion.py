from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("bot", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="agentsession",
            name="capability_registry",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="agentsession",
            name="streaming_connected",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="agentsession",
            name="last_feed_sync_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="agentsession",
            name="last_watchlist_sync_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="provider_scores",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="signal_freshness",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="conflict_flags",
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="invalidation_reason",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="sentiment_summary",
            field=models.JSONField(default=dict),
        ),
    ]
