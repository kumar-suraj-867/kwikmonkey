"""Persistent storage for option chain snapshots and spot data.

Supports SQLite (local dev) and PostgreSQL (cloud deployment).
Backend is auto-detected from DATABASE_URL env var or Streamlit secret.
"""

import os
import socket
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from urllib.parse import urlparse, urlunparse

import pandas as pd


# ======================================================================
# Backend detection
# ======================================================================

def _get_secret(key, default=""):
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def _resolve_ipv4(url: str) -> str:
    """Replace hostname in a PostgreSQL URL with its IPv4 address.

    Supabase returns IPv6 addresses which some hosts (Streamlit Cloud)
    cannot connect to.  Resolving to IPv4 avoids the issue.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return url
        ipv4 = socket.getaddrinfo(hostname, None, socket.AF_INET)[0][4][0]
        # Replace hostname (preserving user:pass and port) with the IPv4 address
        netloc = parsed.netloc.replace(hostname, ipv4)
        return urlunparse(parsed._replace(netloc=netloc))
    except (socket.gaierror, IndexError, Exception):
        return url


_DATABASE_URL_RAW = _get_secret("DATABASE_URL")
_DATABASE_URL = _resolve_ipv4(_DATABASE_URL_RAW) if _DATABASE_URL_RAW else ""
USE_POSTGRES = bool(_DATABASE_URL and "postgres" in _DATABASE_URL)

if USE_POSTGRES:
    import psycopg2

# DB file lives next to the app (SQLite only)
DB_PATH = os.path.join(os.path.dirname(__file__), "market_data.db")


# ======================================================================
# PostgreSQL compatibility wrapper
# ======================================================================

class _PgConn:
    """Wraps psycopg2 connection to behave like sqlite3 for our use patterns.

    Translates ? → %s placeholders and provides execute/executemany/executescript.
    """

    def __init__(self, conn):
        self._conn = conn

    @staticmethod
    def _q(sql):
        return sql.replace("?", "%s")

    def execute(self, sql, params=None):
        cur = self._conn.cursor()
        cur.execute(self._q(sql), params or ())
        return cur

    def executemany(self, sql, params_list):
        cur = self._conn.cursor()
        cur.executemany(self._q(sql), params_list)
        return cur

    def executescript(self, sql):
        """Execute multiple statements (no params)."""
        cur = self._conn.cursor()
        cur.execute(sql)
        return cur


# ======================================================================
# Connection management
# ======================================================================

@contextmanager
def _connect(db_path: str = DB_PATH):
    """Context manager yielding a connection wrapper.

    Uses PostgreSQL if DATABASE_URL is set, else SQLite.
    """
    if USE_POSTGRES:
        conn = psycopg2.connect(_DATABASE_URL, sslmode="require")
        try:
            yield _PgConn(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _read_sql(query, conn, params=None, parse_dates=None):
    """pd.read_sql_query wrapper that works with both backends."""
    if isinstance(conn, _PgConn):
        return pd.read_sql_query(
            query.replace("?", "%s"), conn._conn,
            params=params, parse_dates=parse_dates,
        )
    return pd.read_sql_query(
        query, conn, params=params, parse_dates=parse_dates,
    )


# ======================================================================
# Schema
# ======================================================================

_DDL_SQLITE = """
    CREATE TABLE IF NOT EXISTS option_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TIMESTAMP NOT NULL,
        expiry_date TEXT NOT NULL,
        strike REAL NOT NULL,
        option_type TEXT NOT NULL,
        ltp REAL,
        bid REAL,
        ask REAL,
        oi INTEGER,
        prev_oi INTEGER,
        volume INTEGER,
        iv REAL,
        spot_price REAL,
        UNIQUE(ts, strike, option_type, expiry_date)
    );

    CREATE INDEX IF NOT EXISTS idx_snap_ts
        ON option_snapshots(ts);
    CREATE INDEX IF NOT EXISTS idx_snap_strike_type
        ON option_snapshots(strike, option_type);
    CREATE INDEX IF NOT EXISTS idx_snap_expiry
        ON option_snapshots(expiry_date);

    CREATE TABLE IF NOT EXISTS spot_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TIMESTAMP NOT NULL UNIQUE,
        ltp REAL,
        open REAL,
        high REAL,
        low REAL,
        prev_close REAL
    );

    CREATE INDEX IF NOT EXISTS idx_spot_ts
        ON spot_snapshots(ts);
"""

_DDL_POSTGRES = """
    CREATE TABLE IF NOT EXISTS option_snapshots (
        id SERIAL PRIMARY KEY,
        ts TIMESTAMP NOT NULL,
        expiry_date TEXT NOT NULL,
        strike REAL NOT NULL,
        option_type TEXT NOT NULL,
        ltp REAL,
        bid REAL,
        ask REAL,
        oi INTEGER,
        prev_oi INTEGER,
        volume INTEGER,
        iv REAL,
        spot_price REAL,
        UNIQUE(ts, strike, option_type, expiry_date)
    );

    CREATE INDEX IF NOT EXISTS idx_snap_ts
        ON option_snapshots(ts);
    CREATE INDEX IF NOT EXISTS idx_snap_strike_type
        ON option_snapshots(strike, option_type);
    CREATE INDEX IF NOT EXISTS idx_snap_expiry
        ON option_snapshots(expiry_date);

    CREATE TABLE IF NOT EXISTS spot_snapshots (
        id SERIAL PRIMARY KEY,
        ts TIMESTAMP NOT NULL UNIQUE,
        ltp REAL,
        "open" REAL,
        high REAL,
        low REAL,
        prev_close REAL
    );

    CREATE INDEX IF NOT EXISTS idx_spot_ts
        ON spot_snapshots(ts);
"""


def init_db(db_path: str = DB_PATH):
    """Create tables if they don't exist."""
    with _connect(db_path) as conn:
        if USE_POSTGRES:
            conn.executescript(_DDL_POSTGRES)
        else:
            conn.executescript(_DDL_SQLITE)


# ======================================================================
# Write operations
# ======================================================================

def save_option_snapshot(df: pd.DataFrame, spot_price: float,
                         expiry_date_str: str, db_path: str = DB_PATH):
    """Save an option chain DataFrame snapshot to the DB.

    Parameters
    ----------
    df : DataFrame with columns: strike, option_type, ltp, bid, ask, oi, prev_oi, volume, iv
    spot_price : current spot price
    expiry_date_str : expiry date as string (e.g. '2026-04-10')
    """
    if df.empty:
        return 0

    now = datetime.now().replace(microsecond=0)
    rows = []
    for _, row in df.iterrows():
        rows.append((
            now,
            expiry_date_str,
            row.get("strike", 0),
            row.get("option_type", ""),
            row.get("ltp", 0),
            row.get("bid", 0),
            row.get("ask", 0),
            int(row.get("oi", 0)),
            int(row.get("prev_oi", 0)),
            int(row.get("volume", 0)),
            row.get("iv", 0),
            spot_price,
        ))

    with _connect(db_path) as conn:
        conn.executemany("""
            INSERT INTO option_snapshots
                (ts, expiry_date, strike, option_type, ltp, bid, ask,
                 oi, prev_oi, volume, iv, spot_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ts, strike, option_type, expiry_date) DO NOTHING
        """, rows)

    return len(rows)


def save_spot_snapshot(spot: dict, db_path: str = DB_PATH):
    """Save a spot price snapshot.

    spot : dict with keys ltp, open, high, low, prev_close
    """
    now = datetime.now().replace(microsecond=0)
    with _connect(db_path) as conn:
        conn.execute("""
            INSERT INTO spot_snapshots
                (ts, ltp, "open", high, low, prev_close)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (ts) DO NOTHING
        """, (
            now,
            spot.get("ltp", 0),
            spot.get("open", 0),
            spot.get("high", 0),
            spot.get("low", 0),
            spot.get("prev_close", 0),
        ))


# ======================================================================
# Read operations (for backtesting)
# ======================================================================

def get_option_history(strike: float, option_type: str,
                       from_dt: datetime, to_dt: datetime,
                       expiry_date_str: str = None,
                       db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch recorded option snapshots for a specific strike/type."""
    query = """
        SELECT ts, strike, option_type, ltp, bid, ask,
               oi, prev_oi, volume, iv, spot_price
        FROM option_snapshots
        WHERE strike = ? AND option_type = ?
          AND ts BETWEEN ? AND ?
    """
    params = [strike, option_type, from_dt, to_dt]

    if expiry_date_str:
        query += " AND expiry_date = ?"
        params.append(expiry_date_str)

    query += " ORDER BY ts"

    with _connect(db_path) as conn:
        df = _read_sql(query, conn, params=params, parse_dates=["ts"])
    return df


def get_spot_history(from_dt: datetime, to_dt: datetime,
                     db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch recorded spot snapshots."""
    with _connect(db_path) as conn:
        df = _read_sql("""
            SELECT ts, ltp, "open", high, low, prev_close
            FROM spot_snapshots
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts
        """, conn, params=[from_dt, to_dt], parse_dates=["ts"])
    return df


def get_available_strikes(from_dt: datetime, to_dt: datetime,
                          option_type: str = None,
                          db_path: str = DB_PATH) -> list[float]:
    """List all strikes that have recorded data in the date range."""
    query = """
        SELECT DISTINCT strike FROM option_snapshots
        WHERE ts BETWEEN ? AND ?
    """
    params = [from_dt, to_dt]
    if option_type:
        query += " AND option_type = ?"
        params.append(option_type)
    query += " ORDER BY strike"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [r[0] for r in rows]


# ======================================================================
# DB info / stats
# ======================================================================

def get_db_stats(db_path: str = DB_PATH) -> dict:
    """Return summary stats about the stored data."""
    if not USE_POSTGRES and not os.path.exists(db_path):
        return {"exists": False, "option_rows": 0, "spot_rows": 0}

    try:
        with _connect(db_path) as conn:
            opt_count = conn.execute(
                "SELECT COUNT(*) FROM option_snapshots"
            ).fetchone()[0]
            spot_count = conn.execute(
                "SELECT COUNT(*) FROM spot_snapshots"
            ).fetchone()[0]

            opt_range = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM option_snapshots"
            ).fetchone()
            spot_range = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM spot_snapshots"
            ).fetchone()

            opt_days = conn.execute(
                "SELECT COUNT(DISTINCT DATE(ts)) FROM option_snapshots"
            ).fetchone()[0]

        if USE_POSTGRES:
            db_size = "cloud"
        else:
            db_size = round(os.path.getsize(db_path) / (1024 * 1024), 2)

        return {
            "exists": True,
            "option_rows": opt_count,
            "spot_rows": spot_count,
            "option_from": opt_range[0] if opt_range[0] else None,
            "option_to": opt_range[1] if opt_range[1] else None,
            "spot_from": spot_range[0] if spot_range[0] else None,
            "spot_to": spot_range[1] if spot_range[1] else None,
            "trading_days": opt_days,
            "db_size_mb": db_size,
        }
    except Exception:
        return {"exists": False, "option_rows": 0, "spot_rows": 0}
