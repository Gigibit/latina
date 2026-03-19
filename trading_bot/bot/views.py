from __future__ import annotations

import logging
import os
from datetime import timedelta

from django.http import HttpResponseServerError, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from trading_bot.bot.data_sources import get_candle_history, resolve_candle_size
from trading_bot.bot.research_session import research_sessions
from trading_bot.bot.service import generate_suggestion, get_best_candidates, get_market_monitor

logger = logging.getLogger(__name__)


def _auto_detection_symbol_number() -> int:
    raw_value = os.getenv("AUTO_DETECTION_SYMBOL_NUMBER", "7").strip()
    if raw_value.lower() in {"false", "off", "no"}:
        return 0
    try:
        return max(int(raw_value), 0)
    except ValueError:
        logger.error(
            "Invalid AUTO_DETECTION_SYMBOL_NUMBER=%s. Falling back to 7.",
            raw_value,
        )
        return 7


@require_GET
def dashboard_view(request):
    try:
        auto_detection_symbol_number = _auto_detection_symbol_number()
        return render(
            request,
            "bot/dashboard.html",
            {
                "auto_detection_symbol_number": auto_detection_symbol_number,
                "manual_symbol_input_enabled": auto_detection_symbol_number == 0,
            },
        )
    except Exception as exc:
        logger.error("Response dashboard_view status=500 error=%s", exc, exc_info=True)
        return HttpResponseServerError("Unable to render dashboard.")


@require_GET
def trading_suggestion_view(request):
    async_mode = request.GET.get("async", "true").lower() in {"1", "true", "yes", "on"}
    status_only = request.GET.get("status", "false").lower() in {"1", "true", "yes", "on"}
    session_id = request.GET.get("session_id")
    symbols_raw = request.GET.get("symbols")
    symbol = request.GET.get("symbol", "AAPL")
    risk_profile = request.GET.get("risk", "medium")
    symbols = [item.strip().upper() for item in (symbols_raw or "").split(",") if item.strip()]

    logger.info(
        "Request trading_suggestion_view async=%s status_only=%s session_id=%s "
        "symbol=%s symbols=%s risk=%s",
        async_mode,
        status_only,
        session_id,
        symbol,
        symbols,
        risk_profile,
    )

    try:
        if async_mode:
            if not session_id:
                if not request.session.session_key:
                    request.session.create()
                session_id = request.session.session_key

            if status_only:
                job = research_sessions.get(session_id)
                if not job:
                    logger.error(
                        "Response trading_suggestion_view status=404 "
                        "reason=unknown_research_session session_id=%s",
                        session_id,
                    )
                    return JsonResponse({"error": "Unknown research session."}, status=404)
            else:
                job = research_sessions.get_or_create(
                    session_id=session_id,
                    risk_profile=risk_profile,
                    candidate_symbols=symbols,
                )

            payload = {
                "session_id": session_id,
                "status": job.status,
                "stream_log": job.stream_log,
                "result": job.result,
                "error": job.error,
            }
            logger.info(
                "Response trading_suggestion_view status=200 async=true session_id=%s "
                "job_status=%s stream_entries=%s has_result=%s has_error=%s",
                session_id,
                job.status,
                len(job.stream_log),
                bool(job.result),
                bool(job.error),
            )
            return JsonResponse(payload)

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
        logger.info(
            "Response trading_suggestion_view status=200 async=false mode=%s risk=%s",
            "multi-symbol" if symbols else "single-symbol",
            risk_profile,
        )
        return JsonResponse(payload)
    except Exception as exc:
        logger.error("Response trading_suggestion_view status=400 error=%s", exc, exc_info=True)
        return JsonResponse({"error": str(exc)}, status=400)


@require_GET
def best_candidates_view(request):
    limit = int(request.GET.get("limit", "5"))
    risk_profile = request.GET.get("risk", "medium")

    logger.info(
        "Request best_candidates_view limit=%s risk=%s",
        limit,
        risk_profile,
    )

    try:
        payload = get_best_candidates(limit=limit, user_risk_profile=risk_profile)
        logger.info(
            "Response best_candidates_view status=200 candidates=%s",
            len(payload.get("candidates", [])),
        )
        return JsonResponse(payload)
    except Exception as exc:
        logger.error("Response best_candidates_view status=400 error=%s", exc, exc_info=True)
        return JsonResponse({"error": str(exc)}, status=400)


@require_GET
def market_monitor_view(request):
    limit = int(request.GET.get("limit", "5"))
    logger.info("Request market_monitor_view limit=%s", limit)
    try:
        payload = get_market_monitor(limit=limit)
        logger.info(
            "Response market_monitor_view status=200 alerts=%s news_count=%s",
            len(payload.get("alerts", [])),
            payload.get("news_count", 0),
        )
        return JsonResponse(payload)
    except Exception as exc:
        logger.error("Response market_monitor_view status=400 error=%s", exc, exc_info=True)
        return JsonResponse({"error": str(exc)}, status=400)


@require_GET
def best_projection_view(request):
    risk_profile = request.GET.get("risk", "medium")
    limit = int(request.GET.get("limit", "5"))

    try:
        candidates_payload = get_best_candidates(limit=limit, user_risk_profile=risk_profile)
        candidates = candidates_payload.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates available for projection.")

        best_candidate = candidates[0]
        symbol = best_candidate["symbol"]
        suggestion = generate_suggestion(symbol=symbol, user_risk_profile=risk_profile)

        candle_size = suggestion.get("candle_size") or os.getenv("CANDLE_SIZE", "1d")
        granularity_size = (
            suggestion.get("candle_rag_granularity_size")
            or os.getenv("CANDLE_RAG_GRANULARITY_SIZE")
            or candle_size
        )

        granularity_interval, candle_days = resolve_candle_size(candle_size)
        _, granularity_days = resolve_candle_size(granularity_size)
        granularity_steps = max(int(candle_days / granularity_days), 1)

        history = get_candle_history(
            symbol=symbol,
            candle_size=granularity_size,
            lookback_candles=max(80, granularity_steps * 5),
        )
        recent_history = history.tail(max(32, granularity_steps * 2)).copy()
        historical_closes = [round(float(value), 4) for value in recent_history["Close"].tolist()]

        last_close = float(history["Close"].iloc[-1])
        buy_probability = float(suggestion.get("buy_probability", 0.5))
        sell_probability = float(suggestion.get("sell_probability", 0.5))
        edge = buy_probability - sell_probability
        expected_change_pct = max(min(edge * 6, 3.5), -3.5)

        target_close = last_close * (1 + (expected_change_pct / 100))
        predicted = []
        for index in range(1, granularity_steps + 1):
            progress = index / granularity_steps
            wave = 0.25 * (1 if index % 2 else -1)
            interpolated = last_close + ((target_close - last_close) * progress)
            predicted_close = interpolated + (abs(target_close - last_close) * wave * 0.1)
            predicted.append(round(predicted_close, 4))

        index_last = recent_history.index[-1]
        last_timestamp = (
            index_last.to_pydatetime() if hasattr(index_last, "to_pydatetime") else index_last
        )
        spacing = timedelta(hours=1) if granularity_interval == "1h" else timedelta(days=1)
        predicted_timestamps = [
            (last_timestamp + (spacing * step)).isoformat()
            for step in range(1, granularity_steps + 1)
        ]

        return JsonResponse(
            {
                "risk_profile": risk_profile,
                "best_candidate": best_candidate,
                "candidate_name": symbol,
                "candle_size": candle_size,
                "candle_rag_granularity_size": granularity_size,
                "granularity_steps_for_next_candle": granularity_steps,
                "historical_granularity_closes": historical_closes,
                "predicted_granularity_closes": predicted,
                "predicted_timestamps": predicted_timestamps,
                "buy_probability": round(buy_probability, 4),
                "sell_probability": round(sell_probability, 4),
                "decision": suggestion.get("decision", {}),
            }
        )
    except Exception as exc:
        logger.error("Response best_projection_view status=400 error=%s", exc, exc_info=True)
        return JsonResponse({"error": str(exc)}, status=400)
