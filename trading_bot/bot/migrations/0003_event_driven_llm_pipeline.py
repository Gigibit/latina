import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("bot", "0002_agent_signal_expansion"),
    ]

    operations = [
        migrations.AddField(
            model_name="agentproposal",
            name="confidence_adjustment",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="interpretation",
            field=models.TextField(default=""),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="llm_conflicts",
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="risk_note",
            field=models.TextField(default=""),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="snapshot_hash_key",
            field=models.CharField(default="", max_length=128),
        ),
        migrations.AddField(
            model_name="agentproposal",
            name="uncertainty",
            field=models.CharField(default="medium", max_length=16),
        ),
        migrations.CreateModel(
            name="SymbolFeatureSnapshot",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("symbol", models.CharField(max_length=16)),
                ("timestamp", models.DateTimeField(auto_now=True)),
                ("event_id", models.CharField(max_length=128)),
                ("snapshot_hash", models.CharField(max_length=128)),
                ("payload", models.JSONField(default=dict)),
                (
                    "session",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="feature_snapshots",
                        to="bot.agentsession",
                    ),
                ),
            ],
            options={"unique_together": {("session", "symbol")}},
        ),
    ]
