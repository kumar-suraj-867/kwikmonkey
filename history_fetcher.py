"""Fetch historical OHLCV candles from Fyers API for backtesting."""

import time

import pandas as pd
from fyers_apiv3 import fyersModel

import config


class HistoryFetcher:
    """Wraps fyers.history() with error handling, rate-limiting, and retries."""

    def __init__(self, fyers_model: fyersModel.FyersModel,
                 underlying: str = None):
        self.fyers = fyers_model
        self.underlying = underlying or config.NIFTY_UNDERLYING
        self.last_error = None

    _RESOLUTION_MAP = {
        "Day": "D", "day": "D", "1D": "D", "D": "D",
        "1": "1", "5": "5", "15": "15", "30": "30",
        "60": "60", "120": "120", "240": "240",
    }

    _MAX_RETRIES = 3
    _BASE_DELAY = 1.0  # seconds between calls
    _RETRY_DELAY = 3.0  # seconds to wait on rate limit before retry

    def get_candles(self, symbol: str, from_date: str, to_date: str,
                    resolution: str = "15") -> pd.DataFrame:
        """Fetch OHLCV candles for any symbol.

        Includes rate-limit delay and automatic retry on 'request limit reached'.
        """
        fyers_res = self._RESOLUTION_MAP.get(resolution, resolution)

        params = {
            "symbol": symbol,
            "resolution": fyers_res,
            "date_format": "1",
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "1",
        }

        self.last_error = None

        for attempt in range(self._MAX_RETRIES):
            # Rate-limit: wait between every API call
            time.sleep(self._BASE_DELAY)

            try:
                resp = self.fyers.history(data=params)
            except Exception as e:
                self.last_error = f"Exception for {symbol}: {e}"
                return pd.DataFrame()

            if resp.get("s") == "ok":
                candles = resp.get("candles", [])
                if not candles:
                    self.last_error = f"{symbol}: API returned ok but 0 candles"
                    return pd.DataFrame()

                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                return df

            # Check if rate limited
            msg = str(resp.get("message", ""))
            if "request limit" in msg.lower() or "rate limit" in msg.lower():
                wait = self._RETRY_DELAY * (attempt + 1)
                time.sleep(wait)
                continue  # retry

            # Other error — don't retry
            self.last_error = f"{symbol} [{from_date} → {to_date}]: {resp.get('s')} — {msg}"
            return pd.DataFrame()

        # All retries exhausted
        self.last_error = f"{symbol} [{from_date} → {to_date}]: rate limit — retries exhausted"
        return pd.DataFrame()

    def get_spot_candles(self, from_date: str, to_date: str,
                         resolution: str = "15",
                         underlying: str = None) -> pd.DataFrame:
        """Fetch index spot candles."""
        symbol = underlying or self.underlying
        return self.get_candles(symbol, from_date, to_date, resolution)

    def get_option_candles(self, symbol: str, from_date: str, to_date: str,
                           resolution: str = "15") -> pd.DataFrame:
        """Fetch candles for an individual option contract."""
        return self.get_candles(symbol, from_date, to_date, resolution)

    def get_vix_candles(self, from_date: str, to_date: str,
                        resolution: str = "15") -> pd.DataFrame:
        """Fetch India VIX candles."""
        import config
        return self.get_candles(config.VIX_SYMBOL, from_date, to_date, resolution)
