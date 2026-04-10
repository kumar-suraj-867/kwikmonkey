# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NIFTY 50 Options Trading Dashboard — a real-time Indian equity options analytics platform with backtesting and paper trading. Built with Streamlit + Fyers API v3. Supports NIFTY 50 and SENSEX indices.

## Running the App

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run dashboard.py

# Auth (first run) — follow the in-app OAuth flow, or run standalone:
python auth.py
```

Streamlit serves on port 8501. Dev container auto-starts the server on attach.

Python version is pinned to 3.12 (`.python-version`) for Streamlit Cloud compatibility — fyers_apiv3 fails on 3.14.

## Architecture

The app is a single Streamlit process with four tabs (Live Dashboard, Backtest, Paper Trading, Data Migration) all rendered from `dashboard.py` (~2500 lines).

### Module layers

**UI layer** — `dashboard.py` (main entry + live dashboard), `backtest_ui.py`, `paper_trading_ui.py`

**API/Auth** — `auth.py` (Fyers OAuth2, token persistence), `data_fetcher.py` (quotes, option chains), `history_fetcher.py` (historical candles)

**Data persistence** — `data_store.py` (dual-backend: SQLite local / PostgreSQL cloud, auto-detected from `DATABASE_URL`), `data_collector.py` (background daemon saving option chain snapshots), `migrate_to_pg.py` (one-time SQLite-to-PostgreSQL migration)

**Analytics** — `metrics.py` (Black-Scholes pricing, IV solver, Greeks, PCR, Max Pain), `price_action.py` (trend detection via EMAs, support/resistance, entry signals)

**Backtesting** — `backtest_engine.py` (hybrid: uses locally-collected option data when available, falls back to Black-Scholes synthetic pricing from spot candles)

**Paper Trading** — `paper_trading.py` (position lifecycle, account tracking, DB persistence for trades/account)

### Config/secrets flow

`config.py` reads from Streamlit secrets first, then `.env` / `os.environ`. Token priority: session state > Streamlit secrets > env vars > `.fyers_token` file.

### Database

`data_store.py` abstracts SQLite vs PostgreSQL behind a `_connect()` context manager and `_PgConn` wrapper that translates `?` placeholders to `%s` and casts numpy types. PostgreSQL uses connection pooling (psycopg2 SimpleConnectionPool, max 5) with `sslmode=require`. IPv4 resolution is applied for Supabase URLs (IPv6 workaround).

Tables: `option_snapshots`, `spot_snapshots` (in data_store.py); `paper_trades`, `paper_account` (in paper_trading.py).

### Multi-index support

`INDEX_PROFILES` dict in `config.py` defines per-index parameters (underlying symbol, lot size, strike step, expiry weekday). Currently NIFTY 50 and SENSEX. To add an index, add an entry to this dict.

## Key Design Decisions

- **Hybrid backtesting**: Fyers API has no historical data for expired options. The engine uses locally-collected snapshots when available, falling back to BS model. Data source is tracked per trade ("db", "bs_model", "mixed").
- **Bulk inserts**: PostgreSQL path uses `psycopg2.extras.execute_values` (100x faster than row-by-row `executemany`).
- **Auto-refresh**: Uses `st.fragment` for non-blocking dashboard updates at configurable intervals.
- **No test suite**: The project has no automated tests. Validation is done by running the Streamlit app.

## Environment Variables

Required: `FYERS_APP_ID`, `FYERS_SECRET_KEY`, `FYERS_REDIRECT_URI`

Optional: `DATABASE_URL` (PostgreSQL for cloud), `FYERS_ACCESS_TOKEN` (pre-existing token), `RISK_FREE_RATE` (default 0.07), `DEFAULT_STRIKE_COUNT` (default 15), `REFRESH_INTERVAL_SEC` (default 30)

## Deployment

- **Local**: `.env` file + SQLite (`market_data.db`, git-ignored)
- **Streamlit Cloud**: Streamlit secrets + PostgreSQL via `DATABASE_URL`, Python 3.12 pinned
- **Dev Container**: Python 3.11 image, auto-installs requirements, auto-starts Streamlit
