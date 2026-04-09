"""Fetch option chain and market data from Fyers API."""

import calendar
import re
from datetime import datetime, date

import pandas as pd
from fyers_apiv3 import fyersModel

import config


def _extract_strike(item: dict) -> float:
    """Extract strike price from an option chain item, trying multiple keys
    and falling back to parsing the symbol name."""
    # Try known API key names
    strike = (item.get("strikePrice")
              or item.get("strike_price")
              or item.get("strikeprice")
              or item.get("strike"))
    if strike and float(strike) > 0:
        return float(strike)

    # Fallback: parse from symbol like NSE:NIFTY25D1023500CE
    sym = item.get("symbol", "")
    m = re.search(r'(\d{4,6})(CE|PE)$', sym)
    if m:
        return float(m.group(1))

    return 0.0


class FyersDataFetcher:
    """Wrapper around Fyers API for option chain and spot data."""

    def __init__(self, access_token: str,
                 underlying: str = None, options_symbol: str = None):
        self.fyers = fyersModel.FyersModel(
            client_id=config.FYERS_APP_ID,
            token=access_token,
            is_async=False,
            log_path="",
        )
        self.underlying = underlying or config.NIFTY_UNDERLYING
        self.options_symbol = options_symbol or self.options_symbol

    # ------------------------------------------------------------------
    # Spot / underlying
    # ------------------------------------------------------------------

    def get_spot_quote(self) -> dict:
        """Return spot price info for the active index.

        Returns dict with keys: ltp, change, change_pct, open, high, low, prev_close.
        """
        resp = self.fyers.quotes({"symbols": self.underlying})

        if resp.get("code") != 200 and resp.get("s") != "ok":
            raise RuntimeError(f"Quotes API error: {resp}")

        d = resp["d"][0]["v"]
        return {
            "ltp": d.get("lp", 0),
            "change": d.get("ch", 0),
            "change_pct": d.get("chp", 0),
            "open": d.get("open_price", 0),
            "high": d.get("high_price", 0),
            "low": d.get("low_price", 0),
            "prev_close": d.get("prev_close_price", 0),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

    # ------------------------------------------------------------------
    # Option chain
    # ------------------------------------------------------------------

    def get_expiry_dates(self) -> list[dict]:
        """Return list of expiry dicts with 'date' and 'expiry' (epoch) keys.

        Makes a lightweight call with strikecount=1 and the first available
        expiry to extract the full expiry list from the response.
        """
        # First call without timestamp to get the expiry list from error/data
        resp = self.fyers.optionchain(
            {"symbol": self.options_symbol, "strikecount": 1, "timestamp": ""}
        )
        # The API may return the expiry list even on error code 1
        expiry_data = resp.get("data", {}).get("expiryData", [])
        if not expiry_data:
            raise RuntimeError(f"No expiry data returned: {resp}")
        return expiry_data

    def get_option_chain(self, strike_count: int = None,
                         expiry_ts: str = "") -> pd.DataFrame:
        """Fetch option chain and return a flat DataFrame.

        Parameters
        ----------
        strike_count : number of strikes above and below ATM
        expiry_ts : expiry epoch timestamp string (e.g. '1776074400')

        Returns DataFrame with columns:
            symbol, strike, option_type, ltp, bid, ask, open, high, low,
            prev_close, volume, oi, prev_oi, change, change_pct, iv
        """
        if strike_count is None:
            strike_count = config.DEFAULT_STRIKE_COUNT

        params = {
            "symbol": self.options_symbol,
            "strikecount": strike_count,
            "timestamp": expiry_ts,
        }
        resp = self.fyers.optionchain(params)

        if resp.get("s") == "error":
            raise RuntimeError(f"Option chain API error: {resp}")

        chain = resp.get("data", {}).get("optionsChain", [])
        if not chain:
            return pd.DataFrame()

        rows = []
        for item in chain:
            rows.append({
                "symbol": item.get("symbol", ""),
                "strike": _extract_strike(item),
                "option_type": item.get("option_type", ""),
                "ltp": item.get("ltp", 0),
                "bid": item.get("bid", 0),
                "ask": item.get("ask", 0),
                "open": item.get("open", 0),
                "high": item.get("high", 0),
                "low": item.get("low", 0),
                "prev_close": item.get("prevClose", 0),
                "volume": item.get("volume", 0),
                "oi": item.get("oi", 0),
                "prev_oi": item.get("prevOI", item.get("prev_oi", 0)),
                "change": item.get("change", 0),
                "change_pct": item.get("changePer", item.get("chp", 0)),
                "iv": item.get("iv", 0),
            })

        df = pd.DataFrame(rows)
        df["oi_change"] = df["oi"] - df["prev_oi"]
        df["spread"] = df["ask"] - df["bid"]
        return df

    def get_option_chain_with_expiries(self, strike_count: int = None,
                                        expiry_ts: str = "") -> tuple[pd.DataFrame, list[dict]]:
        """Return (option_chain_df, expiry_list) in one call."""
        if strike_count is None:
            strike_count = config.DEFAULT_STRIKE_COUNT

        params = {
            "symbol": self.options_symbol,
            "strikecount": strike_count,
            "timestamp": expiry_ts,
        }
        resp = self.fyers.optionchain(params)

        if resp.get("code") != 200 and resp.get("s") != "ok":
            raise RuntimeError(f"Option chain API error: {resp}")

        data = resp.get("data", {})
        expiry_data = data.get("expiryData", [])
        chain = data.get("optionsChain", [])

        if not chain:
            return pd.DataFrame(), expiry_data

        rows = []
        for item in chain:
            rows.append({
                "symbol": item.get("symbol", ""),
                "strike": _extract_strike(item),
                "option_type": item.get("option_type", ""),
                "ltp": item.get("ltp", 0),
                "bid": item.get("bid", 0),
                "ask": item.get("ask", 0),
                "open": item.get("open", 0),
                "high": item.get("high", 0),
                "low": item.get("low", 0),
                "prev_close": item.get("prevClose", 0),
                "volume": item.get("volume", 0),
                "oi": item.get("oi", 0),
                "prev_oi": item.get("prevOI", item.get("prev_oi", 0)),
                "change": item.get("change", 0),
                "change_pct": item.get("changePer", item.get("chp", 0)),
                "iv": item.get("iv", 0),
            })

        df = pd.DataFrame(rows)
        df["oi_change"] = df["oi"] - df["prev_oi"]
        df["spread"] = df["ask"] - df["bid"]
        return df, expiry_data

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def get_vix_quote(self) -> dict:
        """Return India VIX quote."""
        resp = self.fyers.quotes({"symbols": config.VIX_SYMBOL})
        if resp.get("code") != 200 and resp.get("s") != "ok":
            raise RuntimeError(f"VIX quotes error: {resp}")
        d = resp["d"][0]["v"]
        return {
            "ltp": d.get("lp", 0),
            "change": d.get("ch", 0),
            "change_pct": d.get("chp", 0),
            "open": d.get("open_price", 0),
            "high": d.get("high_price", 0),
            "low": d.get("low_price", 0),
            "prev_close": d.get("prev_close_price", 0),
        }

    # ------------------------------------------------------------------
    # Futures
    # ------------------------------------------------------------------

    @staticmethod
    def _futures_symbol(prefix: str, dt: date = None) -> str:
        """Build futures symbol like NSE:NIFTY26APRFUT."""
        dt = dt or date.today()
        yy = dt.strftime("%y")
        mmm = dt.strftime("%b").upper()
        return f"{prefix}{yy}{mmm}FUT"

    def get_futures_quote(self, futures_prefix: str = None) -> dict | None:
        """Return near-month futures quote with OI.

        Tries current month; if that fails (expired), tries next month.
        """
        prefix = futures_prefix or "NSE:NIFTY"
        today = date.today()
        symbols_to_try = [
            self._futures_symbol(prefix, today),
        ]
        # Add next month
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        symbols_to_try.append(self._futures_symbol(prefix, next_month))

        for sym in symbols_to_try:
            try:
                resp = self.fyers.quotes({"symbols": sym})
                if resp.get("s") != "ok" and resp.get("code") != 200:
                    continue
                d = resp["d"][0]["v"]
                oi = d.get("open_interest", d.get("oi", 0))
                ltp = d.get("lp", 0)
                if ltp <= 0:
                    continue
                return {
                    "symbol": sym,
                    "ltp": ltp,
                    "change": d.get("ch", 0),
                    "change_pct": d.get("chp", 0),
                    "open": d.get("open_price", 0),
                    "high": d.get("high_price", 0),
                    "low": d.get("low_price", 0),
                    "prev_close": d.get("prev_close_price", 0),
                    "volume": d.get("volume", 0),
                    "oi": oi,
                }
            except Exception:
                continue
        return None
