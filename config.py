import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit Cloud secrets first, then .env / os.environ."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# Fyers API credentials
FYERS_APP_ID = _get_secret("FYERS_APP_ID")
FYERS_SECRET_KEY = _get_secret("FYERS_SECRET_KEY")
FYERS_REDIRECT_URI = _get_secret(
    "FYERS_REDIRECT_URI",
    "https://trade.fyers.in/api-login/redirect-uri/abc123",
)

# Market settings
RISK_FREE_RATE = float(_get_secret("RISK_FREE_RATE", "0.07"))
DEFAULT_STRIKE_COUNT = int(_get_secret("DEFAULT_STRIKE_COUNT", "15"))
REFRESH_INTERVAL_SEC = float(_get_secret("REFRESH_INTERVAL_SEC", "1"))
LIVE_DATA_STALE_SEC = int(_get_secret("LIVE_DATA_STALE_SEC", "10"))
CHAIN_REST_REFRESH_SEC = int(_get_secret("CHAIN_REST_REFRESH_SEC", "60"))

# Symbols
NIFTY_UNDERLYING = "NSE:NIFTY50-INDEX"
NIFTY_OPTIONS_SYMBOL = "NSE:NIFTY50-INDEX"
VIX_SYMBOL = "NSE:INDIAVIX-INDEX"

# NIFTY option parameters
NIFTY_LOT_SIZE = 65
NIFTY_STRIKE_STEP = 50

# Index profiles for multi-index support
INDEX_PROFILES = {
    "NIFTY 50": {
        "name": "NIFTY 50",
        "underlying": "NSE:NIFTY50-INDEX",
        "options_symbol": "NSE:NIFTY50-INDEX",
        "futures_prefix": "NSE:NIFTY",
        "lot_size": 65,
        "strike_step": 50,
        "expiry_weekday": 3,  # Thursday
    },
    "SENSEX": {
        "name": "SENSEX",
        "underlying": "BSE:SENSEX-INDEX",
        "options_symbol": "BSE:SENSEX-INDEX",
        "futures_prefix": "BSE:SENSEX",
        "lot_size": 10,
        "strike_step": 100,
        "expiry_weekday": 4,  # Friday
    },
}

# Backtesting
HISTORY_API_DELAY_SEC = 0.15

# Token file path
TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".fyers_token")
