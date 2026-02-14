from __future__ import annotations

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from trading_bot.bot.service import generate_suggestion, generate_weekly_chat_suggestion


@require_GET
def chat_page_view(request):
    return render(request, "chat.html")


@require_GET
def trading_suggestion_view(request):
    symbol = request.GET.get("symbol", "AAPL")
    risk_profile = request.GET.get("risk", "medium")

    try:
        payload = generate_suggestion(symbol=symbol, user_risk_profile=risk_profile)
        return JsonResponse(payload)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)


@csrf_exempt
@require_POST
def chat_suggestion_view(request):
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    message = (body.get("message") or "").strip()
    risk_profile = (body.get("risk") or "medium").strip().lower()

    if not message:
        return JsonResponse({"error": "Field 'message' is required."}, status=400)

    try:
        payload = generate_weekly_chat_suggestion(message=message, user_risk_profile=risk_profile)
        return JsonResponse(payload)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except RuntimeError as exc:
        return JsonResponse({"error": str(exc)}, status=502)
    except Exception as exc:
        return JsonResponse({"error": f"Unexpected error: {exc}"}, status=500)
