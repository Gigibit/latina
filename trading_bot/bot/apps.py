import os

from django.apps import AppConfig


class BotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "trading_bot.bot"

    def ready(self) -> None:
        enabled = os.getenv("AGENT_REVIEW_TRADER_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return

        from trading_bot.bot.agent_runtime import load_agent_config

        try:
            load_agent_config()
        except Exception as exc:
            raise RuntimeError(f"Agent config validation failed: {exc}") from exc
