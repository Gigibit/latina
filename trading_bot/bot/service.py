from __future__ import annotations

import os

from trading_bot.bot.data_sources import (
    fetch_trending_symbols,
    fetch_x_sentiment_scores,
    get_candle_history,
    get_market_snapshot,
    resolve_candle_size,
)
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.retrieval import FaissEmbeddingRetriever


def _build_behavior_samples(symbol: str, candle_size: str, rag_inference_number: int):
    _, window_size = resolve_candle_size(candle_size)

    lookback_candles = max(rag_inference_number + window_size + 60, 120)
    history = get_candle_history(
        symbol=symbol,
        candle_size=candle_size,
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
                    f"next_{candle_size}_move={pct_change:.2f}% ({action})."
                ),
                "action": action,
                "next_pct_change": pct_change,
                "x_sentiment": round(mean_sentiment, 4),
                "weighted_x_sentiment": round(weighted_sentiment, 4),
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

    return samples, query, sentiment_metadata


def _is_env_flag_enabled(name: str, default: bool = True) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def generate_suggestion(symbol: str, user_risk_profile: str = "medium") -> dict:
    candle_size = os.getenv("CANDLE_SIZE", "1d")
    raw_rag_inference_number = os.getenv(
        "CANDLES_RAG_INFERENCE_NUMER", os.getenv("CANDLES_RAG_INFERENCE_NUMBER", "64")
    )
    rag_inference_number = int(raw_rag_inference_number)
    if rag_inference_number <= 1:
        raise ValueError("CANDLES_RAG_INFERENCE_NUMER must be greater than 1")

    samples, query, sentiment_metadata = _build_behavior_samples(
        symbol, candle_size, rag_inference_number
    )
    corpus = [sample["text"] for sample in samples]

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = FaissEmbeddingRetriever(embedding_model)

    nearest_behaviors = retriever.top_k(query=query, corpus=corpus, k=min(7, len(corpus)))

    selected = [samples[chunk.index] for chunk in nearest_behaviors if chunk.index is not None]
    buy_count = sum(1 for item in selected if item["action"] == "BUY")
    total_count = max(len(selected), 1)
    buy_probability = buy_count / total_count

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
        f"(similarity={chunk.score:.3f}, next_action={samples[chunk.index]['action']})"
        for chunk in nearest_behaviors
        if chunk.index is not None
    ]
    context_blob = "\n".join(context_lines)
    prompt = (
        f"Symbol: {symbol.upper()}\n"
        f"Risk profile: {user_risk_profile}\n"
        f"Candle size: {candle_size}\n"
        f"Buy probability from similar historical behaviors: {buy_probability * 100:.2f}%\n"
        f"Sell probability from similar historical behaviors: {(1 - buy_probability) * 100:.2f}%\n"
        f"Selected market context:\n{context_blob}\n"
        f"Predict whether next {candle_size} should be BUY or SELL, "
        "include probability and concise risk notes."
    )

    decision = decider.decide(prompt)
    return {
        "symbol": symbol.upper(),
        "risk_profile": user_risk_profile,
        "provider": provider,
        "model": model,
        "candle_size": candle_size,
        "candles_rag_inference_number": rag_inference_number,
        "x_sentiment": sentiment_metadata,
        "buy_probability": round(buy_probability, 4),
        "sell_probability": round(1 - buy_probability, 4),
        "selected_context": [chunk.__dict__ for chunk in nearest_behaviors],
        "decision": decision,
    }


def get_best_candidates(limit: int = 5, user_risk_profile: str = "medium") -> dict:
    symbols = fetch_trending_symbols(limit=max(limit * 3, 10))
    candidates = []

    for symbol in symbols:
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
            }
        )

    top_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]
    return {
        "risk_profile": user_risk_profile,
        "source": "Yahoo Finance trending",
        "candidates": top_candidates,
    }
