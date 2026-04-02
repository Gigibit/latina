# AI Trading Suggestion Bot (Django)

Simple Django API that provides **BUY / SELL / HOLD** suggestions using:
- Multi-source market data (Yahoo Finance/Stooq candles, FRED macro data, market RSS news)
- Experimental crypto market analysis via Binance (enabled only when `BINANCE_CRYPTO_API_KEY` is set)
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
- `GET /api/market-monitor/?limit=5` (includes optional experimental `crypto_market` block)

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
- `TRENDING_CANDIDATES_SEARCH_NUMBER` controls Yahoo Finance trending query size via `?count=` (default: `10`).
- `MARKETS_DATA_PROVIDER` selects candle/fundamentals provider: `yfinance` (default), `stooq`, `alpha_vantage`, or `massive` (Massive docs: https://massive.com/docs).
- `ALPHA_VANTAGE_API_KEY` is required when `MARKETS_DATA_PROVIDER=alpha_vantage`.
- `MASSIVE_API_KEY` is required when `MARKETS_DATA_PROVIDER=massive`.

- `BINANCE_CRYPTO_API_KEY` enables experimental crypto market analysis in `/api/market-monitor/`.
- `CRYPTO_MARKET_PROVIDER` currently accepts only `binance` (default: `binance`).
- `CRYPTO_MARKET_SYMBOLS` comma-separated Binance symbols to evaluate (default: `BTCUSDT,ETHUSDT,BNBUSDT`).

## Agent Review Trader (human-in-the-loop)

New tab **Agent Review Trader** provides a strict approval workflow:

- Start/Stop agent runtime (`POST /api/agent/start`, `POST /api/agent/stop`)
- Poll session and health (`GET /api/agent/session`, `GET /api/agent/health`)
- Review pending decisions (`GET /api/agent/proposals`)
- Explicit approval only via user click (`POST /api/agent/proposals/:id/approve`)
- Explicit rejection (`POST /api/agent/proposals/:id/reject`)
- Logs and portfolio snapshots (`GET /api/agent/logs`, `GET /api/agent/portfolio`)

### Required env vars for Agent Review Trader

- `ETORO_API_KEY`
- `OPENAI_API_KEY`
- `MAX_AGENT_LOSS`

Optional:

- `AGENT_LOOP_INTERVAL_MS`
- `APPROVAL_TIMEOUT_MS`
- `MAX_OPEN_PROPOSALS`
- `MAX_POSITION_SIZE`
- `MAX_DAILY_TRADES`
- `PROPOSAL_INVALIDATION_PCT` (default `1.5`)

### Capability registry and feature flags

At startup the agent builds a runtime capability registry and stores it in session payload:

- `supportsFeeds`
- `supportsSocialAnalytics`
- `supportsCuratedLists`
- `supportsWatchlists`
- `supportsMarketMonitorStreaming`
- `supportsAgentPortfolios`
- `supportsDemoTrading`
- `supportsRealTrading`

Each capability can be force-enabled/disabled through env flags:

- `ETORO_CAP_SUPPORTS_FEEDS`
- `ETORO_CAP_SUPPORTS_SOCIAL_ANALYTICS`
- `ETORO_CAP_SUPPORTS_CURATED_LISTS`
- `ETORO_CAP_SUPPORTS_WATCHLISTS`
- `ETORO_CAP_SUPPORTS_MARKET_MONITOR_STREAMING`
- `ETORO_CAP_SUPPORTS_AGENT_PORTFOLIOS`
- `ETORO_CAP_SUPPORTS_DEMO_TRADING`
- `ETORO_CAP_SUPPORTS_REAL_TRADING`

### Multi-signal architecture

The Agent Review Trader now uses modular signal providers:

1. `TechnicalSignalProvider` (dominant signal)
2. `PortfolioRiskProvider`
3. `MarketRegimeProvider`
4. `FeedSentimentProvider` (auxiliary only)
5. `WatchlistAttentionProvider` (auxiliary only)
6. `CuratedListProvider` (auxiliary only)

Each provider returns:

- `score`
- `confidence`
- `freshness`
- `rationale`
- `raw_inputs`
- `warnings`

The fusion engine applies weighted synthesis:

- `technical_score = 0.60`
- `portfolio_risk_score = 0.20`
- `market_regime_score = 0.10`
- `social_sentiment_score = 0.05`
- `watchlist_interest_score = 0.05`

Safety rules enforced:

- no action without explicit approval
- sentiment never dominates technical analysis
- risk can veto opportunities (`REQUIRE_REVIEW`)
- stale or materially moved market can invalidate pending proposals
- duplicate pending proposals are suppressed

### Proposal types

Supported proposal semantics:

- `BUY_CANDIDATE`
- `SELL_CANDIDATE`
- `REDUCE_CANDIDATE`
- `CLOSE_CANDIDATE`
- `HOLD`
- `WATCH`
- `INVALIDATE_PENDING`
- `REQUIRE_REVIEW`

### OpenAI synthesis layer

OpenAI is used for explainability only:

- recommendation synthesis across heterogeneous providers
- “why now” summary
- conflict explanation
- uncertainty signaling

OpenAI never sends orders directly.

### Agent session and proposal API expansion

`GET /api/agent/session` now includes:

- capability flags (`capabilities`)
- `streamingConnected`
- `lastFeedSync`
- `lastWatchlistSync`

`GET /api/agent/proposals` now includes per proposal:

- `providerScores`
- `signalFreshness`
- `conflictFlags`
- `invalidationReason`
- `sentimentSummary`

### Data persistence

The runtime stores:

- `agent_sessions`
- `agent_logs`
- `agent_proposals`
- `agent_approvals`
- `portfolio_snapshots`
- `execution_events`

Run migrations after pulling changes:

```bash
python manage.py migrate
```
