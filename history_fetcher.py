"""Fetch historical OHLCV candles from Fyers API for backtesting."""

import time

import pandas as pd
from fyers_apiv3 import fyersModel

import config


class HistoryFetcher:
    """Wraps fyers.history() with error handling and DataFrame output."""

    def __init__(self, fyers_model: fyersModel.FyersModel):
        self.fyers = fyers_model
        self.last_error = None  # stores last API error for debugging

    # Map user-friendly resolution names to Fyers API values
    _RESOLUTION_MAP = {
        "Day": "D", "day": "D", "1D": "D", "D": "D",
        "1": "1", "5": "5", "15": "15", "30": "30",
        "60": "60", "120": "120", "240": "240",
    }

    def get_candles(self, symbol: str, from_date: str, to_date: str,
                    resolution: str = "15") -> pd.DataFrame:
        """Fetch OHLCV candles for any symbol.

        Parameters
        ----------
        symbol : Fyers symbol e.g. "NSE:NIFTY50-INDEX" or "NSE:NIFTY2541024000CE"
        from_date : "yyyy-mm-dd"
        to_date : "yyyy-mm-dd"
        resolution : "1","5","15","30","60","D","Day"

        Returns
        -------
        DataFrame with columns: timestamp, open, high, low, close, volume
        Empty DataFrame on error or no data.
        """
        fyers_res = self._RESOLUTION_MAP.get(resolution, resolution)

        params = {
            "symbol": symbol,
            "resolution": fyers_res,
            "date_format": "1",
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "0",
        }

        self.last_error = None

        try:
            resp = self.fyers.history(data=params)
        except Exception as e:
            self.last_error = f"Exception for {symbol}: {e}"
            return pd.DataFrame()

        if resp.get("s") != "ok":
            self.last_error = f"{symbol} [{from_date} → {to_date}]: {resp.get('s')} — {resp.get('message', resp)}"
            return pd.DataFrame()

        candles = resp.get("candles", [])
        if not candles:
            self.last_error = f"{symbol}: API returned ok but 0 candles"
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def get_spot_candles(self, from_date: str, to_date: str,
                         resolution: str = "15") -> pd.DataFrame:
        """Fetch NIFTY 50 index candles."""
        return self.get_candles(config.NIFTY_UNDERLYING, from_date, to_date, resolution)

    def get_option_candles(self, symbol: str, from_date: str, to_date: str,
                           resolution: str = "15") -> pd.DataFrame:
        """Fetch candles for an individual option contract with rate-limit delay."""
        time.sleep(config.HISTORY_API_DELAY_SEC)
        return self.get_candles(symbol, from_date, to_date, resolution)
