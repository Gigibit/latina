from __future__ import annotations

import os

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from trading_bot.bot.research_session import research_sessions
from trading_bot.bot.service import generate_suggestion, get_best_candidates, get_market_monitor


def _auto_detection_symbol_number() -> int:
    raw_value = os.getenv("AUTO_DETECTION_SYMBOL_NUMBER", "7").strip()
    if raw_value.lower() in {"false", "off", "no"}:
        return 0
    return max(int(raw_value), 0)


@require_GET
def dashboard_view(request):
    auto_detection_symbol_number = _auto_detection_symbol_number()
    return render(
        request,
        "bot/dashboard.html",
        {
            "auto_detection_symbol_number": auto_detection_symbol_number,
            "manual_symbol_input_enabled": auto_detection_symbol_number == 0,
        },
    )


@require_GET
def dashboard_view(request):
    return render(request, "bot/dashboard.html")

@require_GET
def trading_suggestion_view(request):
    async_mode = request.GET.get("async", "true").lower() in {"1", "true", "yes", "on"}
    status_only = request.GET.get("status", "false").lower() in {"1", "true", "yes", "on"}
    session_id = request.GET.get("session_id")
    symbols_raw = request.GET.get("symbols")
    symbol = request.GET.get("symbol", "AAPL")
    risk_profile = request.GET.get("risk", "medium")
    symbols = [item.strip().upper() for item in (symbols_raw or "").split(",") if item.strip()]

    try:
        if async_mode:
            if not session_id:
                if not request.session.session_key:
                    request.session.create()
                session_id = request.session.session_key

            if status_only:
                job = research_sessions.get(session_id)
                if not job:
                    return JsonResponse({"error": "Unknown research session."}, status=404)
            else:
                job = research_sessions.get_or_create(
                    session_id=session_id,
                    risk_profile=risk_profile,
                    candidate_symbols=symbols,
                )

            return JsonResponse(
                {
                    "session_id": session_id,
                    "status": job.status,
                    "stream_log": job.stream_log,
                    "result": job.result,
                    "error": job.error,
                }
            )

        if symbols:
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


@require_GET
def market_monitor_view(request):
    limit = int(request.GET.get("limit", "5"))
    try:
        payload = get_market_monitor(limit=limit)
        return JsonResponse(payload)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
