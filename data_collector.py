"""Background data collector — saves live option chain snapshots to SQLite.

Runs as a daemon thread alongside the Streamlit dashboard.
Collects option chain + spot data at configurable intervals.
"""

import threading
import time
from datetime import datetime

import data_store
from data_fetcher import FyersDataFetcher


class DataCollector:
    """Periodically fetches and stores option chain + spot data."""

    def __init__(self, fetcher: FyersDataFetcher, interval_sec: int = 60):
        self.fetcher = fetcher
        self.interval_sec = interval_sec
        self._thread = None
        self._stop_event = threading.Event()
        self.last_snapshot_time = None
        self.last_error = None
        self.total_snapshots = 0
        self.is_running = False

    def start(self):
        """Start the background collection thread."""
        if self._thread and self._thread.is_alive():
            return  # already running

        # Ensure DB tables exist
        data_store.init_db()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.is_running = True

    def stop(self):
        """Stop the background collection thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_running = False

    def _run(self):
        """Main collection loop."""
        while not self._stop_event.is_set():
            try:
                self._collect_once()
            except Exception as e:
                self.last_error = f"{datetime.now():%H:%M:%S} — {e}"

            self._stop_event.wait(self.interval_sec)

        self.is_running = False

    def _collect_once(self):
        """Fetch and store one snapshot."""
        # Get spot price
        spot = self.fetcher.get_spot_quote()
        data_store.save_spot_snapshot(spot)

        # Get expiry list, use nearest expiry
        expiry_list = self.fetcher.get_expiry_dates()
        if not expiry_list:
            self.last_error = "No expiry dates available"
            return

        nearest_expiry = expiry_list[0]
        expiry_ts = str(nearest_expiry.get("expiry", ""))
        expiry_date_str = nearest_expiry.get("date", "")

        # Convert dd-mm-yyyy to yyyy-mm-dd for storage
        if expiry_date_str:
            parts = expiry_date_str.split("-")
            if len(parts) == 3:
                expiry_date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"

        # Get option chain for nearest expiry
        df = self.fetcher.get_option_chain(expiry_ts=expiry_ts)

        if not df.empty:
            count = data_store.save_option_snapshot(
                df, spot["ltp"], expiry_date_str
            )
            self.total_snapshots += 1
            self.last_snapshot_time = datetime.now()
            self.last_error = None

    def status(self) -> dict:
        """Return current collector status."""
        return {
            "running": self.is_running,
            "interval_sec": self.interval_sec,
            "total_snapshots": self.total_snapshots,
            "last_snapshot": self.last_snapshot_time.strftime("%H:%M:%S") if self.last_snapshot_time else None,
            "last_error": self.last_error,
        }
