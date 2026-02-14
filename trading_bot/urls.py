from django.urls import path

from trading_bot.bot.views import trading_suggestion_view

urlpatterns = [
    path("api/suggestion/", trading_suggestion_view, name="trading-suggestion"),
]
