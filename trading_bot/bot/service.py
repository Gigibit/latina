from __future__ import annotations

import os
import re
from dataclasses import asdict, is_dataclass

from trading_bot.bot.data_sources import MarketSnapshot, get_market_snapshot
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.retrieval import EmbeddingRetriever

DEFAULT_CANDIDATES = ["AAPL", "MSFT", "NVDA", "SPY", "GOOGL", "AMZN"]


def build_symbol_corpus(symbol: str) -> list[str]:
    lookback_days = int(os.getenv("YAHOO_LOOKBACK_DAYS", "90"))
    snapshot = get_market_snapshot(symbol, lookback_days=lookback_days)
    return snapshot_to_corpus(snapshot)


def snapshot_to_corpus(snapshot: MarketSnapshot) -> list[str]:
    relative_volume = snapshot.latest_volume / max(snapshot.avg_volume_20d, 1)
    return [
        f"{snapshot.symbol} close is {snapshot.latest_close:.2f} USD.",
        f"5-day momentum is {snapshot.pct_change_5d:.2f}%.",
        f"20-day momentum is {snapshot.pct_change_20d:.2f}%.",
        f"Latest volume is {relative_volume:.2f}x the 20-day average.",
        "High positive momentum with unusual volume can suggest trend continuation.",
        "Strong negative momentum with unusual volume can indicate downside pressure.",
        "Low volume conviction may justify HOLD until signal improves.",
    ]


def _build_llm_decider() -> tuple[LLMDecider, str, str]:
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        api_key = os.getenv("HUGGINGFACE_API_KEY")

    decider = LLMDecider(provider=provider, model=model, api_key=api_key)
    return decider, provider, model


def _extract_symbols_from_message(message: str) -> list[str]:
    candidates = [
        item.strip().upper()
        for item in os.getenv("DEFAULT_CANDIDATES", ",".join(DEFAULT_CANDIDATES)).split(",")
        if item.strip()
    ]
    candidate_set = set(candidates)

    dollar_tickers = {match.upper() for match in re.findall(r"\$([A-Za-z]{1,5})", message)}
    stopwords = {
        "WHAT",
        "SHOULD",
        "NEXT",
        "WEEK",
        "THAN",
        "WITH",
        "RISK",
        "BEST",
        "BUY",
        "SELL",
        "HOLD",
        "THE",
        "FOR",
        "AND",
        "YOU",
        "HAVE",
        "YOUR",
        "IS",
        "DO",
        "DOES",
        "DID",
        "CAN",
        "COULD",
        "WILL",
        "WOULD",
        "TO",
        "OF",
        "IN",
        "ON",
        "AT",
        "ME",
        "MY",
        "WE",
    }
    free_tokens = {
        token.upper()
        for token in re.findall(r"\b[A-Za-z]{2,5}\b", message)
        if token.upper() not in stopwords
    }
    known_tokens = {token for token in free_tokens if token in candidate_set}

    explicit = sorted(dollar_tickers | known_tokens | free_tokens)
    return explicit or candidates


def _rank_weekly_candidates(symbols: list[str]) -> list[dict]:
    lookback_days = int(os.getenv("YAHOO_LOOKBACK_DAYS", "90"))
    rows = []
    for symbol in symbols:
        snapshot = get_market_snapshot(symbol, lookback_days=lookback_days)
        volume_ratio = snapshot.latest_volume / max(snapshot.avg_volume_20d, 1)
        score = (snapshot.pct_change_5d * 0.6) + (snapshot.pct_change_20d * 0.4)
        score += max(volume_ratio - 1, -1) * 3

        rows.append(
            {
                "symbol": snapshot.symbol,
                "score": round(score, 3),
                "volume_ratio": round(volume_ratio, 3),
                "snapshot": asdict(snapshot) if is_dataclass(snapshot) else snapshot.__dict__,
                "corpus": snapshot_to_corpus(snapshot),
            }
        )

    return sorted(rows, key=lambda item: item["score"], reverse=True)


def generate_suggestion(symbol: str, user_risk_profile: str = "medium") -> dict:
    corpus = build_symbol_corpus(symbol)

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = EmbeddingRetriever(embedding_model)
    query = f"Find evidence useful for a {user_risk_profile} risk trading decision for {symbol}."
    top_chunks = retriever.top_k(query=query, corpus=corpus, k=4)

    decider, provider, model = _build_llm_decider()

    context_lines = [f"- {chunk.text} (similarity={chunk.score:.3f})" for chunk in top_chunks]
    context_blob = "\n".join(context_lines)
    prompt = (
        f"Symbol: {symbol.upper()}\n"
        f"Risk profile: {user_risk_profile}\n"
        f"Selected market context:\n{context_blob}\n"
        "Return JSON with action, confidence, reasoning, risk_notes for next-week horizon."
    )

    decision = decider.decide(prompt)
    return {
        "symbol": symbol.upper(),
        "risk_profile": user_risk_profile,
        "provider": provider,
        "model": model,
        "selected_context": [chunk.__dict__ for chunk in top_chunks],
        "decision": decision,
    }


def generate_weekly_chat_suggestion(message: str, user_risk_profile: str = "medium") -> dict:
    symbols = _extract_symbols_from_message(message)
    ranked = _rank_weekly_candidates(symbols)
    top_ranked = ranked[: min(4, len(ranked))]

    combined_corpus: list[str] = []
    for item in top_ranked:
        combined_corpus.extend(item["corpus"])

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = EmbeddingRetriever(embedding_model)
    query = f"Question: {message} | horizon: next week | risk: {user_risk_profile}"
    top_chunks = retriever.top_k(query=query, corpus=combined_corpus, k=6)

    decider, provider, model = _build_llm_decider()
    context_lines = [f"- {chunk.text} (similarity={chunk.score:.3f})" for chunk in top_chunks]
    context_blob = "\n".join(context_lines)
    ranking_blob = "\n".join(
        f"- {item['symbol']}: score={item['score']} volume_ratio={item['volume_ratio']}"
        for item in top_ranked
    )

    prompt = (
        "You are helping with a weekly trading suggestion.\n"
        "Return JSON keys: answer, top_pick, alternatives, reasoning, risk_notes, confidence.\n"
        f"User message: {message}\n"
        f"Risk profile: {user_risk_profile}\n"
        f"Pre-ranked candidates:\n{ranking_blob}\n"
        f"Retrieved context:\n{context_blob}\n"
        "Do not promise profits."
    )

    decision = decider.decide(prompt)
    answer = decision.get("answer") or (
        f"Top pick for next week: {decision.get('top_pick', top_ranked[0]['symbol'])}"
    )

    return {
        "message": message,
        "risk_profile": user_risk_profile,
        "provider": provider,
        "model": model,
        "ranked_candidates": [
            {"symbol": item["symbol"], "score": item["score"], "volume_ratio": item["volume_ratio"]}
            for item in ranked
        ],
        "selected_context": [chunk.__dict__ for chunk in top_chunks],
        "decision": decision,
        "answer": answer,
    }
