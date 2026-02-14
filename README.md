# AI Trading Suggestion Bot (Django)

Simple Django API that provides **BUY / SELL / HOLD** suggestions using:
- Yahoo Finance market data (`yfinance`)
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

## Endpoint

`GET /api/suggestion/?symbol=AAPL&risk=medium`

Example response:

```json
{
  "symbol": "AAPL",
  "risk_profile": "medium",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "selected_context": [
    {"text": "...", "score": 0.79}
  ],
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
