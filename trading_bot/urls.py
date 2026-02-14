from django.urls import path

from trading_bot.bot.views import chat_page_view, chat_suggestion_view, trading_suggestion_view

urlpatterns = [
    path("chat/", chat_page_view, name="chat-page"),
    path("api/chat/", chat_suggestion_view, name="chat-suggestion"),
    path("api/suggestion/", trading_suggestion_view, name="trading-suggestion"),
]
