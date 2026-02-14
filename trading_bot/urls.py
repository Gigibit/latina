from django.urls import path

from trading_bot.bot.views import best_candidates_view, trading_suggestion_view

urlpatterns = [
    path("api/suggestion/", trading_suggestion_view, name="trading-suggestion"),
    path("api/candidates/", best_candidates_view, name="best-candidates"),
]
