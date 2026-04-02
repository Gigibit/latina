import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trading_bot.settings")
os.environ.setdefault("AGENT_REVIEW_TRADER_ENABLED", "false")
os.environ.setdefault("ETORO_API_KEY", "test-etoro-key")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("MAX_AGENT_LOSS", "1000")
django.setup()
