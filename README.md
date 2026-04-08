# NIFTY 50 Options Trading Dashboard

Real-time NIFTY 50 options analytics dashboard with integrated strategy backtesting, built on Fyers API and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

### Live Dashboard
- **Option Chain** — Real-time option chain with bid/ask spread, OI, volume
- **Greeks** — Delta, Gamma, Theta, Vega, IV via Black-Scholes model
- **OI Analysis** — PCR (OI & Volume), Max Pain, Support/Resistance from OI walls
- **Market Intelligence**
  - Market State — PCR regime, ATM IV, IV skew, trend signal
  - Key Levels — OI-based support/resistance with price positioning
  - OI Activity — Fresh buildup, unwinding, long/short classification
  - Trade Suggestions — Strategy-specific strike picks scored by OI + liquidity + spread
  - Trap Detector — Bull/bear trap detection from price-OI divergence
- **Auto-refresh** — Configurable refresh interval (default 5s)

### Strategy Backtester
- **Strategies** — Iron Condor, Bull Call Spread, Bear Put Spread, Long Straddle
- **Hybrid data** — Uses locally collected option data (SQLite) when available, falls back to Black-Scholes synthetic pricing from NIFTY spot candles
- **Realistic execution** — Configurable slippage (pts) and brokerage (per lot)
- **Metrics** — Win rate, P&L, max drawdown, profit factor, expectancy, Sharpe ratio, Sortino ratio
- **Visualisation** — Equity curve, P&L distribution, trade-by-trade log

### Data Collection
- **Background collector** — Saves live option chain snapshots to SQLite at configurable intervals (1-5 min)
- **Build your own dataset** — Over time, backtests use real recorded OI, prices, and IV instead of synthetic BS prices

## Architecture

```
Fyers API
    |
    v
+-------------------+     +------------------+
| data_fetcher.py   |---->| data_collector.py|---> SQLite (market_data.db)
| history_fetcher.py|     +------------------+         |
+-------------------+                                  v
    |                                         +------------------+
    v                                         | backtest_engine  |
+-------------------+                         | (hybrid: DB + BS)|
| dashboard.py      |                         +------------------+
| (Streamlit UI)    |                                  |
|  - Live Dashboard |                                  v
|  - Backtest Tab   |<----- backtest_ui.py <-----------+
+-------------------+
    |
    v
+-------------------+
| metrics.py        |
| (BS, Greeks, OI)  |
+-------------------+
```

## Setup

### 1. Prerequisites

- Python 3.10+
- [Fyers API account](https://myapi.fyers.in/dashboard) with App ID and Secret Key

### 2. Clone and install

```bash
git clone https://github.com/<your-username>/nifty-options-dashboard.git
cd nifty-options-dashboard

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your Fyers credentials:

```
FYERS_APP_ID=your_app_id-100
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=https://trade.fyers.in/api-login/redirect-uri/abc123
```

### 4. Run

```bash
streamlit run dashboard.py
```

On first run, you'll be redirected to Fyers login. After authentication, the token is saved locally for subsequent sessions.

## Usage

### Live Dashboard Tab

Opens by default. Shows real-time option chain, Greeks, OI analysis, and market intelligence. Auto-refreshes every 5 seconds.

### Backtest Tab

1. **Start data collection** (recommended) — Click "Start Collecting" to save live snapshots. The longer you collect, the better your backtests.
2. **Configure** — Select strategy, date range, strike offsets, risk parameters
3. **Run** — Click "Run Backtest". Results show:
   - Summary metrics (P&L, win rate, Sharpe, etc.)
   - Equity curve and P&L distribution charts
   - Trade-by-trade log with entry/exit details
   - Data source indicator (Local DB vs BS Model)

### Backtest Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Strategy | Iron Condor, Bull Call, Bear Put, Long Straddle | Iron Condor |
| CE/PE Offset | Strike distance from ATM (points) | 200 |
| Wing Width | Distance between sold and bought strikes (IC only) | 50 |
| Stop-loss % | Max loss as % of risk | 50% |
| Target % | Profit target as % of credit/debit | 50% |
| IV Assumption | Annualised IV for BS model fallback | 15% |
| Slippage | Points added/subtracted per trade | 1 pt |
| Brokerage | Cost per lot per trade | 20/lot |

## Project Structure

```
td/
├── auth.py              # Fyers OAuth2 authentication
├── config.py            # Configuration & environment variables
├── dashboard.py         # Main Streamlit app (live dashboard + tabs)
├── data_fetcher.py      # Fyers API wrapper (quotes, option chain)
├── history_fetcher.py   # Fyers historical candle data
├── metrics.py           # Black-Scholes, Greeks, PCR, Max Pain
├── data_store.py        # SQLite storage for option chain snapshots
├── data_collector.py    # Background data collection thread
├── backtest_engine.py   # Strategy simulation (hybrid DB + BS)
├── backtest_ui.py       # Backtest tab UI
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── .gitignore
```

## How the Backtester Works

The backtester uses a **hybrid approach** because Fyers API does not provide historical data for expired option contracts:

1. **Local DB (preferred)** — If you've been collecting data, the engine uses real recorded LTP, OI, and IV for accurate simulation
2. **Black-Scholes fallback** — When local data is unavailable, it fetches NIFTY spot candles from Fyers and computes synthetic option prices using the BS model with your IV assumption

As you collect more data over time, backtests automatically become more accurate.

## Disclaimer

This tool is for **educational and research purposes only**. It is not financial advice. Options trading involves significant risk. Past performance (including backtested results) does not guarantee future results. Always do your own research and consult a qualified financial advisor before trading.

## License

[MIT License](LICENSE)
