"""Real-time market data provider using Fyers WebSocket.

Streams live ticks (LTP, bid/ask, volume) for the spot index and all
option-chain symbols.  A periodic REST refresh supplies IV and OI
(not available via WebSocket).

Thread safety: the WebSocket callback runs on an internal fyers thread.
All shared state is guarded by ``threading.Lock``.  The Streamlit
fragment reads via ``get_latest()`` and never touches internals directly.
"""

import logging
import threading
from datetime import datetime

import pandas as pd

from fyers_apiv3.FyersWebsocket.data_ws import FyersDataSocket

import config

log = logging.getLogger(__name__)


class LiveDataProvider:
    """Bridges Fyers WebSocket ticks into a thread-safe snapshot store."""

    def __init__(
        self,
        access_token: str,
        underlying: str,
        options_symbol: str,
        strike_count: int,
        expiry_ts: str,
        index_name: str,
        futures_prefix: str | None = None,
    ):
        self._access_token = access_token
        self._underlying = underlying
        self._options_symbol = options_symbol
        self._strike_count = strike_count
        self._expiry_ts = expiry_ts
        self._index_name = index_name
        self._futures_prefix = futures_prefix or "NSE:NIFTY"

        # Thread-safe shared state
        self._lock = threading.Lock()
        self._tick_data: dict[str, dict] = {}   # symbol -> latest tick fields
        self._base_chain: pd.DataFrame | None = None  # last REST chain (has IV, OI)
        self._expiry_list: list[dict] = []
        self._spot: dict | None = None
        self._last_tick_time: datetime | None = None
        self._last_error: str | None = None
        self._connected = False

        # Built lazily on start()
        self._data_socket: FyersDataSocket | None = None
        self._subscribed_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, fetcher) -> None:
        """Do initial REST fetch, then connect WebSocket and subscribe."""
        # Initial REST fetch — gives us IV, OI, and the symbol list
        chain_df, expiry_list = fetcher.get_option_chain_with_expiries(
            strike_count=self._strike_count,
            expiry_ts=self._expiry_ts,
        )
        spot = fetcher.get_spot_quote()

        with self._lock:
            self._base_chain = chain_df
            self._expiry_list = expiry_list
            self._spot = spot
            self._last_tick_time = datetime.now()

        # Build symbol list for subscription
        option_symbols = chain_df["symbol"].tolist() if not chain_df.empty else []
        all_symbols = [self._underlying, config.VIX_SYMBOL] + option_symbols

        # Add futures symbol
        from data_fetcher import FyersDataFetcher
        futures_sym = FyersDataFetcher._futures_symbol(self._futures_prefix)
        all_symbols.append(futures_sym)

        # Remove empty strings and deduplicate
        all_symbols = list({s for s in all_symbols if s})

        # Connect WebSocket
        # Reset singleton so we get a clean instance
        FyersDataSocket._instance = None
        self._data_socket = FyersDataSocket(
            access_token=self._access_token,
            litemode=False,
            reconnect=True,
            reconnect_retry=50,
            on_message=self._on_message,
            on_error=self._on_error,
            on_connect=lambda: self._on_connect(all_symbols),
            on_close=self._on_close,
            log_path="",
        )
        self._data_socket.connect()

    def stop(self) -> None:
        """Close WebSocket connection and clean up."""
        if self._data_socket is not None:
            try:
                self._data_socket.close_connection()
            except Exception as e:
                log.warning("Error closing WebSocket: %s", e)
            FyersDataSocket._instance = None
            self._data_socket = None
        with self._lock:
            self._connected = False
            self._subscribed_symbols.clear()

    # ------------------------------------------------------------------
    # WebSocket callbacks (run on fyers internal thread)
    # ------------------------------------------------------------------

    def _on_connect(self, symbols: list[str]) -> None:
        """Called when WebSocket connection is established."""
        with self._lock:
            self._connected = True
            self._last_error = None

        # Subscribe to all symbols
        if symbols and self._data_socket:
            self._data_socket.subscribe(symbols, data_type="SymbolUpdate")
            with self._lock:
                self._subscribed_symbols = set(symbols)

    def _on_message(self, msg: dict) -> None:
        """Called for each tick — update shared state under lock."""
        symbol = msg.get("symbol", "")
        if not symbol:
            return

        now = datetime.now()
        with self._lock:
            # Store raw tick
            self._tick_data[symbol] = msg
            self._last_tick_time = now

            # Update spot dict if this is the underlying index
            if symbol == self._underlying:
                self._spot = {
                    "ltp": msg.get("ltp", 0),
                    "change": msg.get("ch", 0),
                    "change_pct": msg.get("chp", 0),
                    "open": msg.get("open_price", 0),
                    "high": msg.get("high_price", 0),
                    "low": msg.get("low_price", 0),
                    "prev_close": msg.get("prev_close_price", 0),
                    "timestamp": now.strftime("%H:%M:%S"),
                }

    def _on_error(self, msg) -> None:
        """Called on WebSocket error."""
        with self._lock:
            self._last_error = f"{datetime.now():%H:%M:%S} — {msg}"
        log.warning("WebSocket error: %s", msg)

    def _on_close(self, msg) -> None:
        """Called when WebSocket disconnects."""
        with self._lock:
            self._connected = False
        log.info("WebSocket closed: %s", msg)

    # ------------------------------------------------------------------
    # Public read API (called from Streamlit fragment)
    # ------------------------------------------------------------------

    def get_latest(self) -> dict | None:
        """Return merged chain + spot snapshot. Thread-safe.

        Overlays real-time tick data (LTP, bid/ask, volume) onto the
        base chain from the last REST fetch (which has IV and OI).
        """
        with self._lock:
            if self._base_chain is None or self._base_chain.empty:
                return None

            df = self._base_chain.copy()
            tick_data = dict(self._tick_data)  # snapshot
            spot = dict(self._spot) if self._spot else None
            expiry_list = list(self._expiry_list)
            fetch_time = self._last_tick_time

        # WebSocket binary protocol returns all values as floats.  Cast every
        # numeric column so df.at assignments never hit an int64 ↔ float clash.
        num_cols = df.select_dtypes(include=["int64", "int32"]).columns
        df[num_cols] = df[num_cols].astype(float)

        # Merge tick data into chain (outside lock for performance)
        for idx, row in df.iterrows():
            sym = row["symbol"]
            if sym in tick_data:
                tick = tick_data[sym]
                df.at[idx, "ltp"] = tick.get("ltp", row["ltp"])
                df.at[idx, "bid"] = tick.get("bid_price", row["bid"])
                df.at[idx, "ask"] = tick.get("ask_price", row["ask"])
                df.at[idx, "volume"] = tick.get("vol_traded_today", row["volume"])
                df.at[idx, "open"] = tick.get("open_price", row["open"])
                df.at[idx, "high"] = tick.get("high_price", row["high"])
                df.at[idx, "low"] = tick.get("low_price", row["low"])
                df.at[idx, "prev_close"] = tick.get("prev_close_price", row["prev_close"])
                df.at[idx, "change"] = tick.get("ch", row["change"])
                df.at[idx, "change_pct"] = tick.get("chp", row["change_pct"])
                # iv and oi stay from base_chain (REST)

        # Recompute derived columns
        df["oi_change"] = df["oi"] - df["prev_oi"]
        df["spread"] = df["ask"] - df["bid"]

        return {
            "spot": spot,
            "chain_df": df,
            "expiry_list": expiry_list,
            "fetch_time": fetch_time,
        }

    # ------------------------------------------------------------------
    # REST refresh (called from fragment every ~60s)
    # ------------------------------------------------------------------

    def refresh_chain(self, fetcher) -> None:
        """Re-fetch full chain via REST for IV/OI updates and new strikes.

        Also adjusts WebSocket subscriptions if the strike list changed
        (e.g. spot moved and new ATM strikes appeared).
        """
        chain_df, expiry_list = fetcher.get_option_chain_with_expiries(
            strike_count=self._strike_count,
            expiry_ts=self._expiry_ts,
        )
        if chain_df.empty:
            return

        new_symbols = set(chain_df["symbol"].tolist())

        with self._lock:
            old_symbols = (
                set(self._base_chain["symbol"].tolist())
                if self._base_chain is not None and not self._base_chain.empty
                else set()
            )
            self._base_chain = chain_df
            self._expiry_list = expiry_list

        # Adjust subscriptions for changed strikes
        removed = old_symbols - new_symbols
        added = new_symbols - old_symbols

        if self._data_socket and self._data_socket.is_connected():
            if removed:
                try:
                    self._data_socket.unsubscribe(
                        list(removed), data_type="SymbolUpdate"
                    )
                except Exception as e:
                    log.warning("Unsubscribe failed: %s", e)
            if added:
                try:
                    self._data_socket.subscribe(
                        list(added), data_type="SymbolUpdate"
                    )
                except Exception as e:
                    log.warning("Subscribe failed: %s", e)

            with self._lock:
                self._subscribed_symbols = (
                    self._subscribed_symbols - removed
                ) | added

    # ------------------------------------------------------------------
    # Parameter updates (sidebar changes)
    # ------------------------------------------------------------------

    def update_params(
        self,
        strike_count: int,
        expiry_ts: str,
        index_name: str,
        futures_prefix: str | None = None,
    ) -> None:
        """Update fetch parameters. Chain refresh picks up changes on next call."""
        self._strike_count = strike_count
        self._expiry_ts = expiry_ts
        self._index_name = index_name
        if futures_prefix:
            self._futures_prefix = futures_prefix

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return provider status for UI display."""
        with self._lock:
            return {
                "connected": self._connected,
                "subscribed_count": len(self._subscribed_symbols),
                "last_tick": (
                    self._last_tick_time.strftime("%H:%M:%S")
                    if self._last_tick_time
                    else None
                ),
                "last_error": self._last_error,
            }
