from __future__ import annotations

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from trading_bot.bot.service import generate_suggestion


@require_GET
def trading_suggestion_view(request):
    symbol = request.GET.get("symbol", "AAPL")
    risk_profile = request.GET.get("risk", "medium")

    try:
        payload = generate_suggestion(symbol=symbol, user_risk_profile=risk_profile)
        return JsonResponse(payload)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
