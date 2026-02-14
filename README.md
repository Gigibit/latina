# AI Trading Suggestion Bot (Django)

Django service with:
- Yahoo Finance market data (`yfinance`)
- Local embeddings retrieval (`sentence-transformers`)
- LLM decision provider selectable via `.env` (`openai` or `huggingface`)
- ChatGPT-like web UI for weekly suggestions

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

- UI: `GET /chat/`
- API (single symbol): `GET /api/suggestion/?symbol=AAPL&risk=medium`
- API (chat weekly): `POST /api/chat/`

Example chat request:

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message":"what suggestion do you have for the next week?","risk":"medium"}'
```

## Notes

- Educational use only. Not financial advice.
- If dependencies are missing, the service returns clear install guidance errors.

### Web scraping candidates

- If no explicit ticker is provided in the chat message, the service can scrape Yahoo Finance Most Active and use those symbols as candidate assets.
- Configure with `.env`: `USE_WEB_CANDIDATES=true` and `WEB_CANDIDATES_LIMIT=10`.
- If scraping fails, it falls back to `DEFAULT_CANDIDATES`.
