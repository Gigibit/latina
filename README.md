# AI Trading Suggestion Bot (Django)

Simple Django API that provides **BUY / SELL / HOLD** suggestions using:
- Multi-source market data (Yahoo Finance/Stooq candles, FRED macro data, market RSS news)
- Technical analysis (SMA, RSI, MACD, Bollinger Bands)
- Fundamental metrics (P/E, EPS, debt/equity, market cap)
- Local embeddings retrieval (`sentence-transformers`)
- Decision LLM provider selectable via `.env` (`openai` or `huggingface`)
- Optional eToro order automation for BUY/SELL/HOLD with per-action toggles and cooldown window

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy env file:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` with your API keys and preferred models/provider.

## Run

```bash
python manage.py runserver
```

## Endpoints

- `GET /api/suggestion/?risk=medium` (async by default; symbol selection is system-managed via web scraping + sentiment)
- `GET /api/candidates/?limit=5&risk=medium`
- `GET /api/market-monitor/?limit=5`

Example response (`/api/suggestion/`, once async research is completed):

```json
{
  "symbol": "AAPL",
  "risk_profile": "medium",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "buy_probability": 0.67,
  "technical_indicators": {
    "sma_20": 212.3,
    "sma_50": 205.1,
    "rsi_14": 57.2,
    "macd": 1.51,
    "macd_signal": 1.08,
    "bollinger_upper": 219.7,
    "bollinger_lower": 198.8
  },
  "fundamental_metrics": {
    "pe_ratio": 24.1,
    "eps": 6.43,
    "debt_to_equity": 145.2,
    "market_cap": 3450000000000
  },
  "decision": {
    "action": "HOLD",
    "confidence": 62,
    "reasoning": "...",
    "risk_notes": "..."
  },
  "etoro_execution": {
    "status": "skipped",
    "reason": "HOLD action enabled: no order sent"
  }
}
```

## Notes

- This is for educational use only, not financial advice.
- If dependencies are missing, the service returns clear install guidance errors.


## eToro automation

Set these `.env` variables to enable order execution:

- `ETORO_AUTOTRADE_ENABLED=true` to allow automation
- `ETORO_ENABLE_BUY_ACTION`, `ETORO_ENABLE_SELL_ACTION`, `ETORO_ENABLE_HOLD_ACTION` to enable/disable each action
- `ETORO_NEXT_SUGGESTION_WAIT_SECONDS` to wait before sending the same `symbol + action` again
- `ETORO_API_BASE_URL`, `ETORO_API_KEY`, `ETORO_ACCOUNT_ID` for API integration
- `ETORO_AUTOTRADE_STATE_FILE` to store cooldown state

When action is `HOLD`, the API call is skipped and cooldown still gets updated if HOLD is enabled.


### Async research sessions

- Suggestion requests now run asynchronously and are bound to a session id.
- Poll status with `GET /api/suggestion/?async=true&status=true&session_id=<id>`.
- `stream_log` is preserved and updated while research is running so humans can follow progress.
- The engine auto-selects symbols from scraped market trends + sentiment analysis; the UI no longer accepts manual symbol input.
- `SYMBOL_ACTION_ACCEPTANCE_THRESHOLD` enforces a minimum confidence. If confidence is below threshold, action is forced to `HOLD`.
- `RETRY_BACKOFFF_ENABLED` toggles retry + exponential backoff for rate-limited trending-symbol scraping (default: `true`).
- `MARKETS_DATA_PROVIDER` selects candle provider: `yfinance` (default) or `stooq`.
