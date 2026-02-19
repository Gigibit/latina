# AI Trading Suggestion Bot (Django)

Simple Django API that provides **BUY / SELL / HOLD** suggestions using:
- Multi-source market data (Yahoo Finance, FRED macro data, market RSS news)
- Technical analysis (SMA, RSI, MACD, Bollinger Bands)
- Fundamental metrics (P/E, EPS, debt/equity, market cap)
- Local embeddings retrieval (`sentence-transformers`)
- Decision LLM provider selectable via `.env` (`openai` or `huggingface`)

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

- `GET /api/suggestion/?symbol=AAPL&risk=medium`
- `GET /api/candidates/?limit=5&risk=medium`
- `GET /api/market-monitor/?limit=5`

Example response (`/api/suggestion/`):

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
  }
}
```

## Notes

- This is for educational use only, not financial advice.
- If dependencies are missing, the service returns clear install guidance errors.
