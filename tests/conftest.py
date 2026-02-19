import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trading_bot.settings")
django.setup()
