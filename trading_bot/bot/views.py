from __future__ import annotations

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from trading_bot.bot.service import generate_suggestion, get_best_candidates


@require_GET
def trading_suggestion_view(request):
    symbols_raw = request.GET.get("symbols")
    symbol = request.GET.get("symbol", "AAPL")
    risk_profile = request.GET.get("risk", "medium")

    try:
        if symbols_raw:
            symbols = [item.strip() for item in symbols_raw.split(",") if item.strip()]
            payload = {
                "risk_profile": risk_profile,
                "results": [
                    generate_suggestion(symbol=current_symbol, user_risk_profile=risk_profile)
                    for current_symbol in symbols
                ],
            }
        else:
            payload = generate_suggestion(symbol=symbol, user_risk_profile=risk_profile)
        return JsonResponse(payload)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)


@require_GET
def best_candidates_view(request):
    limit = int(request.GET.get("limit", "5"))
    risk_profile = request.GET.get("risk", "medium")

    try:
        payload = get_best_candidates(limit=limit, user_risk_profile=risk_profile)
        return JsonResponse(payload)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
