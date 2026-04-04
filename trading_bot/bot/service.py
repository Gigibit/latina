from __future__ import annotations

import logging
import os
from statistics import mean

from trading_bot.bot.candidate_universe import build_candidate_universe
from trading_bot.bot.data_sources import (
    average_macro_delta,
    compute_technical_indicators,
    fetch_crypto_market_analysis,
    fetch_fundamental_metrics,
    fetch_macro_indicators,
    fetch_market_news,
    fetch_trending_symbols,
    fetch_x_sentiment_scores,
    get_candle_history,
    get_market_snapshot,
    resolve_candle_size,
)
from trading_bot.bot.etoro import execute_etoro_action
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.retrieval import FaissEmbeddingRetriever

DEFAULT_CHUNKIZATION_MODE = "DEFAULT"
INTROSPECTIVE_CANDLE_CHUNKIZATION_MODE = "INTROSPECTIVE_CANDLE"

logger = logging.getLogger(__name__)

def _resolve_chunkization_mode() -> str:
    mode = os.getenv("CHUNKIZATION_MODE", DEFAULT_CHUNKIZATION_MODE).strip().upper()
    allowed_modes = {DEFAULT_CHUNKIZATION_MODE, INTROSPECTIVE_CANDLE_CHUNKIZATION_MODE}
    if mode not in allowed_modes:
        raise ValueError(
            "CHUNKIZATION_MODE must be one of: "
            f"{DEFAULT_CHUNKIZATION_MODE}, {INTROSPECTIVE_CANDLE_CHUNKIZATION_MODE}"
        )
    return mode


def _resolve_rag_granularity_size(candle_size: str) -> tuple[str | None, int | None]:
    chunkization_mode = _resolve_chunkization_mode()
    if chunkization_mode != INTROSPECTIVE_CANDLE_CHUNKIZATION_MODE:
        return None, None

    granularity_size = os.getenv("CANDLE_RAG_GRANULARITY_SIZE")
    if not granularity_size:
        raise ValueError(
            "CANDLE_RAG_GRANULARITY_SIZE is mandatory when "
            "CHUNKIZATION_MODE=INTROSPECTIVE_CANDLE"
        )

    _, candle_days = resolve_candle_size(candle_size)
    _, granularity_days = resolve_candle_size(granularity_size)

    if granularity_days >= candle_days:
        raise ValueError("CANDLE_RAG_GRANULARITY_SIZE must be strictly less than CANDLE_SIZE")

    elements_per_candle = candle_days / granularity_days
    if int(elements_per_candle) != elements_per_candle:
        raise ValueError(
            "CANDLE_RAG_GRANULARITY_SIZE must divide CANDLE_SIZE exactly "
            "for INTROSPECTIVE_CANDLE mode"
        )

    return granularity_size, int(elements_per_candle)


def _build_behavior_samples(
    symbol: str,
    candle_size: str,
    rag_inference_number: int,
    rag_granularity_size: str | None = None,
):
    effective_candle_size = rag_granularity_size or candle_size
    _, window_size = resolve_candle_size(effective_candle_size)
    window_size = max(int(window_size), 1)

    lookback_candles = max(rag_inference_number + window_size + 60, 120)
    history = get_candle_history(
        symbol=symbol,
        candle_size=effective_candle_size,
        lookback_candles=lookback_candles,
    )

    closes = history["Close"].astype(float).tolist()
    volumes = history["Volume"].astype(float).tolist()
    candle_dates = [index.date() for index in history.index]

    sentiment_enabled = _is_env_flag_enabled("TWITTER_SENTIMENT_ANALYSYS_ENABLED", True)
    sentiment_weight = float(os.getenv("TWITTER_SENTIMENT_ANALYSYS_WEIGHT", "0.15"))
    sentiment_by_day = (
        fetch_x_sentiment_scores(symbol=symbol, days=candle_dates) if sentiment_enabled else {}
    )
    sentiment_scores = [float(sentiment_by_day.get(day, 0.0)) for day in candle_dates]

    min_required = rag_inference_number + (window_size * 2) + 2
    if len(closes) < min_required:
        raise ValueError(
            f"Not enough candles for {symbol}. Required at least {min_required}, "
            f"found {len(closes)}"
        )

    history_window_count = max(rag_inference_number - window_size, 1)
    start_index = max(0, len(closes) - history_window_count - (window_size + 1))
    end_index = len(closes) - (window_size + 1)

    samples: list[dict] = []
    for idx in range(start_index, end_index):
        price_slice = closes[idx : idx + window_size]
        volume_slice = volumes[idx : idx + window_size]
        if len(price_slice) < window_size:
            continue

        next_close = closes[idx + window_size]
        pct_change = ((next_close / price_slice[-1]) - 1) * 100
        action = "BUY" if pct_change >= 0 else "SELL"

        mean_price = sum(price_slice) / window_size
        volatility = max(price_slice) - min(price_slice)
        mean_volume = sum(volume_slice) / window_size
        mean_sentiment = sum(sentiment_scores[idx : idx + window_size]) / window_size
        weighted_sentiment = mean_sentiment * sentiment_weight

        samples.append(
            {
                "text": (
                    f"{symbol.upper()} behavior window ending at candle {idx + window_size}: "
                    f"mean_price={mean_price:.2f}, volatility={volatility:.2f}, "
                    f"mean_volume={mean_volume:.2f}, "
                    f"x_sentiment={mean_sentiment:.3f}, "
                    f"weighted_x_sentiment={weighted_sentiment:.3f}, "
                    f"next_{effective_candle_size}_move={pct_change:.2f}% ({action})."
                ),
                "action": action,
                "next_pct_change": pct_change,
                "x_sentiment": round(mean_sentiment, 4),
                "weighted_x_sentiment": round(weighted_sentiment, 4),
                "window_prices": price_slice,
            }
        )

    query_length = rag_inference_number + window_size
    query_closes = closes[-query_length:]
    query_volumes = volumes[-query_length:]
    query_trend = ((query_closes[-1] / query_closes[0]) - 1) * 100
    query_volatility = max(query_closes) - min(query_closes)
    query_avg_volume = sum(query_volumes) / max(len(query_volumes), 1)
    query_sentiment_scores = sentiment_scores[-query_length:]
    query_avg_sentiment = sum(query_sentiment_scores) / max(len(query_sentiment_scores), 1)
    query_weighted_sentiment = query_avg_sentiment * sentiment_weight

    query = (
        f"{symbol.upper()} current sequence over {query_length} candles: "
        f"trend={query_trend:.2f}%, volatility={query_volatility:.2f}, "
        f"avg_volume={query_avg_volume:.2f}, "
        f"x_sentiment={query_avg_sentiment:.3f}, "
        f"weighted_x_sentiment={query_weighted_sentiment:.3f}. Find similar behavior."
    )

    sentiment_metadata = {
        "enabled": sentiment_enabled,
        "weight": sentiment_weight,
        "avg_query_sentiment": round(query_avg_sentiment, 4),
        "weighted_avg_query_sentiment": round(query_weighted_sentiment, 4),
    }

    return samples, query, sentiment_metadata, query_closes


def _select_nearest_behaviors(
    retriever: FaissEmbeddingRetriever,
    query: str,
    samples: list[dict],
    query_prices: list[float],
    chunkization_mode: str,
    candle_chunk_elements: int | None,
) -> list:
    corpus = [sample["text"] for sample in samples]
    if chunkization_mode == DEFAULT_CHUNKIZATION_MODE:
        return retriever.top_k(query=query, corpus=corpus, k=min(7, len(corpus)))

    if not candle_chunk_elements:
        raise ValueError("candle_chunk_elements is required for INTROSPECTIVE_CANDLE mode")

    nearest_chunks = retriever.top_k(query=query, corpus=corpus, k=min(20, len(corpus)))

    query_inner = query_prices[-candle_chunk_elements:]
    if len(query_inner) < candle_chunk_elements:
        return nearest_chunks[: min(7, len(nearest_chunks))]

    ranked_by_inner_time: list[tuple[float, object]] = []
    for chunk in nearest_chunks:
        if chunk.index is None:
            continue
        sample_prices = samples[chunk.index].get("window_prices", [])
        sample_inner = sample_prices[-candle_chunk_elements:]
        if len(sample_inner) < candle_chunk_elements:
            continue

        distance = sum((a - b) ** 2 for a, b in zip(query_inner, sample_inner, strict=False))
        inner_similarity = 1 / (1 + distance)
        combined_score = (max(float(chunk.score), 0.0) * 0.35) + (inner_similarity * 0.65)
        chunk.score = combined_score
        ranked_by_inner_time.append((combined_score, chunk))

    ranked_by_inner_time.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked_by_inner_time[:7]]


def _is_env_flag_enabled(name: str, default: bool = True) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _weighted_behavior_probability(nearest_behaviors, samples: list[dict]) -> tuple[float, float]:
    weighted_buy = 0.0
    total_weight = 0.0

    for chunk in nearest_behaviors:
        if chunk.index is None:
            continue
        action = samples[chunk.index]["action"]
        similarity_weight = max(float(getattr(chunk, "score", 0.0)), 0.0)
        weight = similarity_weight + 1e-6
        total_weight += weight
        if action == "BUY":
            weighted_buy += weight

    if total_weight <= 0:
        return 0.5, 1.0

    buy_probability = weighted_buy / total_weight
    uncertainty = 1 - min(total_weight / max(len(nearest_behaviors), 1), 1.0)
    return buy_probability, uncertainty


def generate_suggestion(symbol: str, user_risk_profile: str = "medium") -> dict:
    candle_size = os.getenv("CANDLE_SIZE", "1d")
    chunkization_mode = _resolve_chunkization_mode()
    rag_granularity_size, candle_chunk_elements = _resolve_rag_granularity_size(candle_size)

    raw_rag_inference_number = os.getenv(
        "CANDLES_RAG_INFERENCE_NUMER", os.getenv("CANDLES_RAG_INFERENCE_NUMBER", "64")
    )
    rag_inference_number = int(raw_rag_inference_number)
    if rag_inference_number <= 1:
        raise ValueError("CANDLES_RAG_INFERENCE_NUMER must be greater than 1")

    samples, query, sentiment_metadata, query_prices = _build_behavior_samples(
        symbol,
        candle_size,
        rag_inference_number,
        rag_granularity_size=rag_granularity_size,
    )

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = FaissEmbeddingRetriever(embedding_model)

    nearest_behaviors = _select_nearest_behaviors(
        retriever,
        query,
        samples,
        query_prices,
        chunkization_mode,
        candle_chunk_elements,
    )

    buy_probability, retrieval_uncertainty = _weighted_behavior_probability(
        nearest_behaviors, samples
    )

    technical = compute_technical_indicators(symbol)
    fundamentals = fetch_fundamental_metrics(symbol)
    macro_indicators = fetch_macro_indicators()
    macro_delta = average_macro_delta(macro_indicators)

    indicator_bias = 0.0
    indicator_bias += 0.06 if technical.sma_20 > technical.sma_50 else -0.06
    if technical.rsi_14 < 30:
        indicator_bias += 0.04
    elif technical.rsi_14 > 70:
        indicator_bias -= 0.04
    indicator_bias += 0.03 if technical.macd > technical.macd_signal else -0.03

    fundamental_bias = 0.0
    if fundamentals.pe_ratio is not None and fundamentals.pe_ratio < 25:
        fundamental_bias += 0.03
    if fundamentals.debt_to_equity is not None and fundamentals.debt_to_equity > 180:
        fundamental_bias -= 0.03

    macro_penalty = min(macro_delta / 100, 0.04)

    uncertainty_penalty = 0.06 if retrieval_uncertainty > 0.35 else 0.0

    adjusted_buy_probability = min(
        0.98,
        max(
            0.02,
            buy_probability
            + indicator_bias
            + fundamental_bias
            - macro_penalty
            - uncertainty_penalty,
        ),
    )

    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        api_key = os.getenv("HUGGINGFACE_API_KEY")

    decider = LLMDecider(provider=provider, model=model, api_key=api_key)

    context_lines = [
        f"- {chunk.text} "
        f"(similarity={chunk.score:.3f}, "
        f"next_action={samples[chunk.index]['action']}, "
        f"next_move={samples[chunk.index]['next_pct_change']:.2f}%)"
        for chunk in nearest_behaviors
        if chunk.index is not None
    ]
    context_blob = "\n".join(context_lines)
    prompt = (
        f"Symbol: {symbol.upper()}\n"
        f"Risk profile: {user_risk_profile}\n"
        f"Candle size: {candle_size}\n"
        "Buy probability from similar historical behaviors: "
        f"{adjusted_buy_probability * 100:.2f}%\n"
        f"Sell probability from similar historical behaviors: "
        f"{(1 - adjusted_buy_probability) * 100:.2f}%\n"
        f"Technical indicators: SMA20={technical.sma_20:.2f}, SMA50={technical.sma_50:.2f}, "
        f"RSI14={technical.rsi_14:.2f}, MACD={technical.macd:.3f}, "
        f"MACD_SIGNAL={technical.macd_signal:.3f}, BB_upper={technical.bollinger_upper:.2f}, "
        f"BB_lower={technical.bollinger_lower:.2f}.\n"
        f"Fundamental metrics: PE={fundamentals.pe_ratio}, EPS={fundamentals.eps}, "
        f"Debt/Equity={fundamentals.debt_to_equity}, MarketCap={fundamentals.market_cap}.\n"
        f"Macro delta absolute mean: {macro_delta:.4f}.\n"
        f"Retrieval uncertainty score (0=low, 1=high): {retrieval_uncertainty:.3f}.\n"
        f"Selected market context:\n{context_blob}\n"
        "Return strict JSON with keys action, confidence, reasoning, risk_notes. "
        f"Predict whether next {candle_size} should be BUY, SELL, or HOLD. "
        "Decision rule: If buy_probability - sell_probability >= 0.12 choose BUY. "
        "If sell_probability - buy_probability >= 0.12 choose SELL. "
        "Else choose HOLD. Apply HOLD only when evidence is mixed or uncertainty is high."
    )

    decision = decider.decide(prompt)
    current_price = float(query_prices[-1]) if query_prices else None
    reference_price = float(technical.sma_20) if technical.sma_20 else None
    etoro_result = execute_etoro_action(
        symbol=symbol,
        action=str(decision.get("action", "HOLD")),
        confidence=decision.get("confidence"),
        current_price=current_price,
        reference_price=reference_price,
    )
    return {
        "symbol": symbol.upper(),
        "risk_profile": user_risk_profile,
        "provider": provider,
        "model": model,
        "candle_size": candle_size,
        "chunkization_mode": chunkization_mode,
        "candle_rag_granularity_size": rag_granularity_size,
        "candle_rag_granularity_elements": candle_chunk_elements,
        "candles_rag_inference_number": rag_inference_number,
        "x_sentiment": sentiment_metadata,
        "technical_indicators": technical.__dict__,
        "fundamental_metrics": fundamentals.__dict__,
        "macro_indicators": [item.__dict__ for item in macro_indicators],
        "buy_probability": round(adjusted_buy_probability, 4),
        "sell_probability": round(1 - adjusted_buy_probability, 4),
        "selected_context": [chunk.__dict__ for chunk in nearest_behaviors],
        "retrieval_uncertainty": round(retrieval_uncertainty, 4),
        "decision": decision,
        "etoro_execution": etoro_result,
    }


def get_best_candidates(limit: int = 5, user_risk_profile: str = "medium") -> dict:
    universe = build_candidate_universe(
        target_size=max(limit * 3, 10),
        min_avg_volume_20d=1.0,
        min_latest_close=0.01,
        fetch_symbols_fn=fetch_trending_symbols,
        snapshot_fn=get_market_snapshot,
    )
    candidates = []

    for row in universe:
        symbol = row["symbol"]
        try:
            snapshot = get_market_snapshot(symbol)
        except ValueError:
            continue

        relative_volume = snapshot.latest_volume / max(snapshot.avg_volume_20d, 1)
        risk_multiplier = {"low": 0.6, "medium": 1.0, "high": 1.35}.get(
            user_risk_profile.lower(), 1.0
        )
        score = (snapshot.pct_change_5d + (relative_volume - 1) * 8) * risk_multiplier

        candidates.append(
            {
                "symbol": snapshot.symbol,
                "latest_close": round(snapshot.latest_close, 2),
                "pct_change_5d": round(snapshot.pct_change_5d, 2),
                "pct_change_20d": round(snapshot.pct_change_20d, 2),
                "relative_volume": round(relative_volume, 2),
                "score": round(score, 2),
                "zone": row["region"],
            }
        )

    top_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]

    llm_evaluations: dict[str, dict] = {}
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if top_candidates and openai_api_key:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_decider = LLMDecider(provider="openai", model=openai_model, api_key=openai_api_key)
        llm_evaluations = openai_decider.evaluate_trending_candidates(top_candidates)

    enriched_candidates = []
    for candidate in top_candidates:
        llm_data = llm_evaluations.get(candidate["symbol"], {})
        llm_score = float(llm_data.get("llm_score", candidate["score"]))
        combined_score = round((candidate["score"] * 0.65) + (llm_score * 0.35), 2)

        buy_probability = round(min(max((combined_score + 100) / 200, 0.0001), 0.9999), 4)
        sell_probability = round(1 - buy_probability, 4)
        probability_delta = buy_probability - sell_probability
        if probability_delta >= 0.12:
            action = "BUY"
        elif probability_delta <= -0.12:
            action = "SELL"
        else:
            action = "HOLD"

        confidence = int(round(max(buy_probability, sell_probability) * 100))
        if action == "BUY":
            reasoning = (
                "Buy probability significantly exceeds sell probability, "
                "indicating a stronger likelihood of price increase."
            )
        elif action == "SELL":
            reasoning = (
                "Sell probability significantly exceeds buy probability, "
                "indicating a stronger likelihood of price decline."
            )
        else:
            reasoning = (
                "Buy and sell probabilities are close, so there is no clear directional edge."
            )

        enriched_candidates.append(
            {
                **candidate,
                "llm_score": round(llm_score, 2),
                "llm_summary": llm_data.get("summary"),
                "combined_score": combined_score,
                "buy_probability": buy_probability,
                "sell_probability": sell_probability,
                "decision": {
                    "action": action,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "risk_notes": (
                        f"{user_risk_profile.capitalize()} risk profile with "
                        "moderate uncertainty on trend continuation."
                    ),
                },
            }
        )

    ranked_candidates = sorted(
        enriched_candidates,
        key=lambda item: item["combined_score"],
        reverse=True,
    )
    return {
        "risk_profile": user_risk_profile,
        "source": "Yahoo Finance trending (US + EU country zones)",
        "ranking_strategy": "quant_score + openai_comparison",
        "candidates": ranked_candidates,
    }


def get_market_monitor(limit: int = 5) -> dict:
    news = fetch_market_news(limit=limit)
    macro = fetch_macro_indicators()
    crypto_market = fetch_crypto_market_analysis(limit=limit)
    macro_volatility = average_macro_delta(macro)
    alerts: list[str] = []

    if macro_volatility > 0.5:
        alerts.append("Elevata volatilità macroeconomica rilevata.")
    if any(abs(item.delta) > 0.25 for item in macro):
        alerts.append("Sono presenti variazioni macro significative nelle ultime rilevazioni.")
    if crypto_market:
        alerts.append(
            "Crypto market analysis sperimentale attiva tramite Binance "
            "(BINANCE_CRYPTO_API_KEY configurata)."
        )
    if not alerts:
        alerts.append("Nessun alert macro significativo al momento.")

    return {
        "alerts": alerts,
        "macro_indicators": [item.__dict__ for item in macro],
        "news": news,
        "news_count": len(news),
        "crypto_market": crypto_market,
        "market_regime": "risk_off" if macro_volatility > 0.5 else "neutral",
        "headline_sentiment_proxy": round(mean([0.0 for _ in news]) if news else 0.0, 4),
    }
