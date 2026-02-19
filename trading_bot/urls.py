from django.urls import path

from trading_bot.bot.views import (
    best_candidates_view,
    dashboard_view,
    market_monitor_view,
    trading_suggestion_view,
)

urlpatterns = [
    path("", dashboard_view, name="dashboard"),
    path("api/suggestion/", trading_suggestion_view, name="trading-suggestion"),
    path("api/candidates/", best_candidates_view, name="best-candidates"),
    path("api/market-monitor/", market_monitor_view, name="market-monitor"),
]
