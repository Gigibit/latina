from __future__ import annotations

import logging
import os
from datetime import date, timedelta

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from trading_bot.bot.data_sources import get_candle_history, resolve_candle_size
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.research_session import research_sessions
from trading_bot.bot.service import generate_suggestion, get_best_candidates, get_market_monitor

logger = logging.getLogger(__name__)


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
                        "reason=unknown_research_session session_id=%s "
                        "message=Unknown research session.",
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
        logger.error(
            "Response trading_suggestion_view status=400 "
            "message=Failed to generate trading suggestion error=%s",
            exc,
        )
        logger.exception("Response trading_suggestion_view status=400 error=%s", exc)
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
        logger.error(
            "Response best_candidates_view status=400 "
            "message=Failed to compute best candidates error=%s",
            exc,
        )
        logger.exception("Response best_candidates_view status=400 error=%s", exc)
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
        logger.error(
            "Response market_monitor_view status=400 "
            "message=Failed to build market monitor payload error=%s",
            exc,
        )
        logger.exception("Response market_monitor_view status=400 error=%s", exc)
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
        logger.error(
            "Response best_projection_view status=400 "
            "message=Failed to compute best projection payload error=%s",
            exc,
        )
        logger.exception("Response best_projection_view status=400 error=%s", exc)
        return JsonResponse({"error": str(exc)}, status=400)


@require_GET
def candle_playground_view(request):
    symbol = request.GET.get("symbol", "").strip().upper()
    requested_date = request.GET.get("date", "").strip()
    requested_size = request.GET.get("size", "1d").strip()
    requested_history_limit = request.GET.get("history_limit", "10000").strip()
    size_mapping = {
        "1h": "1h",
        "1d": "1d",
        "1w": "7d",
        "1M": "1M",
    }

    logger.info(
        "Request candle_playground_view symbol=%s date=%s size=%s history_limit=%s",
        symbol,
        requested_date,
        requested_size,
        requested_history_limit,
    )

    try:
        if not symbol:
            raise ValueError("symbol is required.")
        if not requested_date:
            raise ValueError("date is required (YYYY-MM-DD).")
        if requested_size not in size_mapping:
            raise ValueError("size must be one of: 1h, 1d, 1w, 1M.")
        try:
            history_limit = int(requested_history_limit)
        except ValueError as exc:
            raise ValueError("history_limit must be an integer between 1 and 10000.") from exc
        if history_limit < 1 or history_limit > 10000:
            raise ValueError("history_limit must be between 1 and 10000.")

        start_date = date.fromisoformat(requested_date)
        if start_date > date.today():
            raise ValueError("date cannot be in the future.")

        normalized_size = size_mapping[requested_size]
        _, candle_span_days = resolve_candle_size(normalized_size)
        distance_days = max((date.today() - start_date).days, 1)
        lookback_candles = min(max(int(distance_days / candle_span_days) + 5, 10), history_limit)

        history = get_candle_history(
            symbol=symbol,
            candle_size=normalized_size,
            lookback_candles=lookback_candles,
        )

        history_rows: list[tuple[date, object]] = []
        for index, row in history.iterrows():
            candle_date = index.date() if hasattr(index, "date") else index
            history_rows.append((candle_date, row))

        if not history_rows:
            logger.error(
                "candle_playground_view history rows are empty after fetch symbol=%s "
                "requested_date=%s size=%s lookback_candles=%s",
                symbol,
                requested_date,
                requested_size,
                lookback_candles,
            )
            raise ValueError("No candle data available from the selected date.")

        earliest_available_date = min(candle_date for candle_date, _ in history_rows)
        latest_available_date = max(candle_date for candle_date, _ in history_rows)
        effective_start_date = start_date
        if start_date > latest_available_date:
            logger.error(
                "candle_playground_view requested date out of range symbol=%s "
                "requested_date=%s latest_available_date=%s earliest_available_date=%s "
                "action=use_earliest_available_date",
                symbol,
                start_date.isoformat(),
                latest_available_date.isoformat(),
                earliest_available_date.isoformat(),
            )
            effective_start_date = earliest_available_date

        positive_candles = 0
        negative_candles = 0
        sequence: list[str] = []
        for candle_date, row in history_rows:
            if candle_date < effective_start_date:
                continue

            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row.get("High", max(open_price, close_price)))
            low_price = float(row.get("Low", min(open_price, close_price)))
            normalized_range_probability = min(
                abs(high_price - low_price) / max(abs(high_price), 1e-9),
                1.0,
            )
            if close_price >= open_price:
                positive_candles += 1
                sequence.append(f"G( probability={normalized_range_probability:.4f} )")
            else:
                negative_candles += 1
                sequence.append(f"R( probability={normalized_range_probability:.4f} )")

        total_candles = positive_candles + negative_candles
        if total_candles == 0:
            logger.error(
                "candle_playground_view no candles after filtering symbol=%s "
                "requested_date=%s effective_start_date=%s earliest_available_date=%s "
                "latest_available_date=%s",
                symbol,
                start_date.isoformat(),
                effective_start_date.isoformat(),
                earliest_available_date.isoformat(),
                latest_available_date.isoformat(),
            )
            raise ValueError("No candle data available from the selected date.")

        sentiment = "positive" if positive_candles >= negative_candles else "negative"
        sequence_text = " ".join(sequence)
        llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        llm_api_key = os.getenv("OPENAI_API_KEY")
        llm_decider = LLMDecider(provider=llm_provider, model=llm_model, api_key=llm_api_key)
        prediction = llm_decider.predict_next_candle_character(sequence_text)

        payload = {
            "symbol": symbol,
            "date": requested_date,
            "size": requested_size,
            "sentiment": sentiment,
            "sequence": sequence_text,
            "prediction": prediction,
            "candles_count": total_candles,
        }
        logger.info(
            "Response candle_playground_view status=200 symbol=%s candles=%s sentiment=%s",
            symbol,
            total_candles,
            sentiment,
        )
        return JsonResponse(payload)
    except Exception as exc:
        logger.error(
            "Response candle_playground_view status=400 symbol=%s date=%s "
            "size=%s history_limit=%s error=%s",
            symbol,
            requested_date,
            requested_size,
            requested_history_limit,
            exc,
        )
        logger.exception("candle_playground_view failed")
        return JsonResponse({"error": str(exc)}, status=400)
