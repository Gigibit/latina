from __future__ import annotations

import os

from trading_bot.bot.data_sources import get_market_snapshot
from trading_bot.bot.llm import LLMDecider
from trading_bot.bot.retrieval import EmbeddingRetriever


def build_symbol_corpus(symbol: str) -> list[str]:
    lookback_days = int(os.getenv("YAHOO_LOOKBACK_DAYS", "90"))
    snapshot = get_market_snapshot(symbol, lookback_days=lookback_days)
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


def generate_suggestion(symbol: str, user_risk_profile: str = "medium") -> dict:
    corpus = build_symbol_corpus(symbol)

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = EmbeddingRetriever(embedding_model)
    query = f"Find evidence useful for a {user_risk_profile} risk trading decision for {symbol}."
    top_chunks = retriever.top_k(query=query, corpus=corpus, k=4)

    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        api_key = os.getenv("HUGGINGFACE_API_KEY")

    decider = LLMDecider(provider=provider, model=model, api_key=api_key)

    context_lines = [f"- {chunk.text} (similarity={chunk.score:.3f})" for chunk in top_chunks]
    context_blob = "\n".join(context_lines)
    prompt = (
        f"Symbol: {symbol.upper()}\n"
        f"Risk profile: {user_risk_profile}\n"
        f"Selected market context:\n{context_blob}\n"
        "Return a directional decision based only on this context."
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
