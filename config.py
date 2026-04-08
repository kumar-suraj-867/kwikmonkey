import os
from dotenv import load_dotenv

load_dotenv()

# Fyers API credentials
FYERS_APP_ID = os.getenv("FYERS_APP_ID", "")
FYERS_SECRET_KEY = os.getenv("FYERS_SECRET_KEY", "")
FYERS_REDIRECT_URI = os.getenv(
    "FYERS_REDIRECT_URI",
    "https://trade.fyers.in/api-login/redirect-uri/abc123",
)

# Market settings
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.07"))
DEFAULT_STRIKE_COUNT = int(os.getenv("DEFAULT_STRIKE_COUNT", "15"))
REFRESH_INTERVAL_SEC = int(os.getenv("REFRESH_INTERVAL_SEC", "5"))

# Symbols
NIFTY_UNDERLYING = "NSE:NIFTY50-INDEX"
NIFTY_OPTIONS_SYMBOL = "NSE:NIFTY50-INDEX"

# NIFTY option parameters
NIFTY_LOT_SIZE = 25
NIFTY_STRIKE_STEP = 50

# Backtesting
HISTORY_API_DELAY_SEC = 0.15

# Token file path
TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".fyers_token")
