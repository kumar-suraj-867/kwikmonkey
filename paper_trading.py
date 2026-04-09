"""Paper trading engine: position management, P&L tracking, persistence.

All trades are simulated — no real orders are placed.  Positions are
persisted to SQLite so they survive app restarts.
"""

import uuid
from datetime import datetime

import pandas as pd

import config
from data_store import _connect, _read_sql, DB_PATH, USE_POSTGRES


# ======================================================================
# DB schema
# ======================================================================

_PAPER_DDL = """
    CREATE TABLE IF NOT EXISTS paper_trades (
        id TEXT PRIMARY KEY,
        group_id TEXT NOT NULL,
        strategy TEXT NOT NULL DEFAULT 'Custom',
        entry_time TIMESTAMP NOT NULL,
        expiry_date TEXT NOT NULL,
        strike REAL NOT NULL,
        option_type TEXT NOT NULL,
        action TEXT NOT NULL,
        lots INTEGER NOT NULL,
        lot_size INTEGER NOT NULL DEFAULT 65,
        entry_price REAL NOT NULL,
        exit_time TIMESTAMP,
        exit_price REAL,
        exit_reason TEXT,
        sl_price REAL,
        target_price REAL,
        status TEXT NOT NULL DEFAULT 'OPEN',
        pnl REAL,
        notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_paper_status
        ON paper_trades(status);
    CREATE INDEX IF NOT EXISTS idx_paper_group
        ON paper_trades(group_id);

    CREATE TABLE IF NOT EXISTS paper_account (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        initial_capital REAL NOT NULL,
        realized_pnl REAL NOT NULL DEFAULT 0,
        total_trades INTEGER NOT NULL DEFAULT 0,
        winning_trades INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP NOT NULL
    );
"""


_paper_initialized = False

def init_paper_tables(db_path: str = DB_PATH):
    """Create paper trading tables if they don't exist. No-op after first call."""
    global _paper_initialized
    if _paper_initialized:
        return
    with _connect(db_path) as conn:
        conn.executescript(_PAPER_DDL)
        # Migration: add lot_size column to existing DBs
        if USE_POSTGRES:
            conn.execute(
                "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS lot_size INTEGER NOT NULL DEFAULT 65"
            )
            conn.execute("ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY")
            conn.execute("ALTER TABLE paper_account ENABLE ROW LEVEL SECURITY")
        else:
            try:
                conn.execute("SELECT lot_size FROM paper_trades LIMIT 1")
            except Exception:
                conn.execute(
                    "ALTER TABLE paper_trades ADD COLUMN lot_size INTEGER NOT NULL DEFAULT 65"
                )
    _paper_initialized = True


# ======================================================================
# Account
# ======================================================================

def get_or_create_account(initial_capital: float = 500000,
                          db_path: str = DB_PATH) -> dict:
    """Get account info, creating it if first run."""
    init_paper_tables(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        if row:
            return {
                "initial_capital": row[1],
                "realized_pnl": row[2],
                "total_trades": row[3],
                "winning_trades": row[4],
                "created_at": row[5],
            }
        conn.execute("""
            INSERT INTO paper_account (id, initial_capital, realized_pnl,
                                       total_trades, winning_trades, created_at)
            VALUES (1, ?, 0, 0, 0, ?)
        """, (initial_capital, datetime.now()))
        return {
            "initial_capital": initial_capital,
            "realized_pnl": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "created_at": datetime.now(),
        }


def reset_account(initial_capital: float = 500000,
                  db_path: str = DB_PATH):
    """Wipe all paper trades and reset account."""
    init_paper_tables(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM paper_account")
        conn.execute("""
            INSERT INTO paper_account (id, initial_capital, realized_pnl,
                                       total_trades, winning_trades, created_at)
            VALUES (1, ?, 0, 0, 0, ?)
        """, (initial_capital, datetime.now()))


# ======================================================================
# Place / close orders
# ======================================================================

def place_order(expiry_date: str, strike: float, option_type: str,
                action: str, lots: int, entry_price: float,
                sl_price: float = None, target_price: float = None,
                strategy: str = "Custom", group_id: str = None,
                notes: str = "", lot_size: int = None,
                db_path: str = DB_PATH) -> str:
    """Place a paper trade. Returns trade id."""
    init_paper_tables(db_path)
    if lot_size is None:
        lot_size = config.NIFTY_LOT_SIZE
    trade_id = uuid.uuid4().hex[:12]
    if not group_id:
        group_id = trade_id

    # Cast numpy types to native Python (psycopg2 can't serialize np.float64)
    strike = float(strike)
    entry_price = float(entry_price)
    lots = int(lots)
    lot_size = int(lot_size)
    sl_price = float(sl_price) if sl_price is not None else None
    target_price = float(target_price) if target_price is not None else None

    with _connect(db_path) as conn:
        conn.execute("""
            INSERT INTO paper_trades
                (id, group_id, strategy, entry_time, expiry_date, strike,
                 option_type, action, lots, lot_size, entry_price,
                 sl_price, target_price, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (trade_id, group_id, strategy, datetime.now(), expiry_date,
              strike, option_type, action, lots, lot_size, entry_price,
              sl_price, target_price, notes))
    return trade_id


def close_position(trade_id: str, exit_price: float,
                   exit_reason: str = "manual",
                   db_path: str = DB_PATH) -> float:
    """Close an open trade. Returns realized P&L."""
    exit_price = float(exit_price)
    init_paper_tables(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT action, lots, entry_price, lot_size FROM paper_trades WHERE id = ? AND status = 'OPEN'",
            (trade_id,)
        ).fetchone()
        if not row:
            return 0

        action, lots, entry_price, lot_size = row
        direction = 1 if action == "BUY" else -1
        pnl = float((exit_price - entry_price) * direction * lots * lot_size)

        conn.execute("""
            UPDATE paper_trades
            SET exit_time = ?, exit_price = ?, exit_reason = ?,
                status = 'CLOSED', pnl = ?
            WHERE id = ?
        """, (datetime.now(), exit_price, exit_reason, round(pnl, 2), trade_id))

        # Update account
        conn.execute("""
            UPDATE paper_account
            SET realized_pnl = realized_pnl + ?,
                total_trades = total_trades + 1,
                winning_trades = winning_trades + ?
            WHERE id = 1
        """, (round(pnl, 2), 1 if pnl > 0 else 0))

    return round(pnl, 2)


def close_group(group_id: str, chain_df: pd.DataFrame,
                exit_reason: str = "manual",
                db_path: str = DB_PATH) -> float:
    """Close all open legs of a strategy group. Returns total P&L."""
    trades = get_open_trades(db_path)
    group_trades = trades[trades["group_id"] == group_id]
    total_pnl = 0
    for _, t in group_trades.iterrows():
        price = _find_live_price(t["strike"], t["option_type"], chain_df)
        if price > 0:
            total_pnl += close_position(t["id"], price, exit_reason, db_path)
    return total_pnl


# ======================================================================
# Queries
# ======================================================================

def get_open_trades(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return all open paper trades."""
    init_paper_tables(db_path)
    with _connect(db_path) as conn:
        df = _read_sql(
            "SELECT * FROM paper_trades WHERE status = 'OPEN' ORDER BY entry_time DESC",
            conn, parse_dates=["entry_time"],
        )
    return df


def get_closed_trades(limit: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    """Return recent closed paper trades."""
    init_paper_tables(db_path)
    with _connect(db_path) as conn:
        df = _read_sql(
            "SELECT * FROM paper_trades WHERE status = 'CLOSED' ORDER BY exit_time DESC LIMIT ?",
            conn, params=[limit], parse_dates=["entry_time", "exit_time"],
        )
    return df


def get_all_trades(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return all paper trades (open + closed)."""
    init_paper_tables(db_path)
    with _connect(db_path) as conn:
        df = _read_sql(
            "SELECT * FROM paper_trades ORDER BY entry_time DESC",
            conn, parse_dates=["entry_time", "exit_time"],
        )
    return df


# ======================================================================
# Live P&L computation
# ======================================================================

def compute_live_pnl(open_trades: pd.DataFrame,
                     chain_df: pd.DataFrame) -> pd.DataFrame:
    """Add live_price, unrealized_pnl columns to open trades using current chain."""
    if open_trades.empty or chain_df.empty:
        open_trades["live_price"] = 0.0
        open_trades["unrealized_pnl"] = 0.0
        return open_trades

    df = open_trades.copy()
    live_prices = []
    pnls = []

    for _, t in df.iterrows():
        price = _find_live_price(t["strike"], t["option_type"], chain_df)
        live_prices.append(price)

        lot_size = t.get("lot_size", config.NIFTY_LOT_SIZE)
        direction = 1 if t["action"] == "BUY" else -1
        pnl = (price - t["entry_price"]) * direction * t["lots"] * lot_size
        pnls.append(round(pnl, 2))

    df["live_price"] = live_prices
    df["unrealized_pnl"] = pnls
    return df


def check_sl_target(open_trades: pd.DataFrame,
                    chain_df: pd.DataFrame,
                    db_path: str = DB_PATH) -> list[str]:
    """Check if any open trades hit SL or target. Auto-close and return messages."""
    if open_trades.empty or chain_df.empty:
        return []

    messages = []
    for _, t in open_trades.iterrows():
        price = _find_live_price(t["strike"], t["option_type"], chain_df)
        if price <= 0:
            continue

        direction = 1 if t["action"] == "BUY" else -1
        pnl_per_unit = (price - t["entry_price"]) * direction

        # SL check
        if t.get("sl_price") and t["sl_price"] > 0:
            if t["action"] == "BUY" and price <= t["sl_price"]:
                realized = close_position(t["id"], price, "sl_hit", db_path)
                messages.append(f"SL hit: {t['option_type']} {t['strike']:.0f} {t['action']} @ {price:.2f} | P&L: {realized:+,.0f}")
            elif t["action"] == "SELL" and price >= t["sl_price"]:
                realized = close_position(t["id"], price, "sl_hit", db_path)
                messages.append(f"SL hit: {t['option_type']} {t['strike']:.0f} {t['action']} @ {price:.2f} | P&L: {realized:+,.0f}")

        # Target check
        if t.get("target_price") and t["target_price"] > 0:
            if t["action"] == "BUY" and price >= t["target_price"]:
                realized = close_position(t["id"], price, "target_hit", db_path)
                messages.append(f"Target hit: {t['option_type']} {t['strike']:.0f} {t['action']} @ {price:.2f} | P&L: {realized:+,.0f}")
            elif t["action"] == "SELL" and price <= t["target_price"]:
                realized = close_position(t["id"], price, "target_hit", db_path)
                messages.append(f"Target hit: {t['option_type']} {t['strike']:.0f} {t['action']} @ {price:.2f} | P&L: {realized:+,.0f}")

    return messages


# ======================================================================
# Helpers
# ======================================================================

def _find_live_price(strike: float, option_type: str,
                     chain_df: pd.DataFrame) -> float:
    """Look up current LTP for a strike/type from the live chain."""
    match = chain_df[
        (chain_df["strike"] == strike) & (chain_df["option_type"] == option_type)
    ]
    if not match.empty:
        return match.iloc[0]["ltp"]
    return 0.0


def compute_portfolio_stats(account: dict, open_trades: pd.DataFrame,
                            chain_df: pd.DataFrame) -> dict:
    """Compute portfolio-level statistics.

    Capital tracking:
    - BUY orders: premium paid is deducted from available cash
    - SELL orders: premium received is added to cash, but margin is blocked
    - Net value = cash + current market value of all open positions
    """
    capital = account["initial_capital"]
    realized = account["realized_pnl"]
    total_trades = account["total_trades"]
    wins = account["winning_trades"]

    # Start with cash = initial capital + realized P&L from closed trades
    cash = capital + realized

    # Unrealized P&L and capital deployed
    premium_paid = 0.0      # total paid for BUY positions
    premium_received = 0.0  # total received for SELL positions
    margin_blocked = 0.0    # SPAN margin estimate for SELL positions
    unrealized = 0.0

    if not open_trades.empty:
        # Use per-trade lot_size (stored at order time)
        if "lot_size" not in open_trades.columns:
            open_trades = open_trades.copy()
            open_trades["lot_size"] = config.NIFTY_LOT_SIZE

        buys = open_trades[open_trades["action"] == "BUY"]
        sells = open_trades[open_trades["action"] == "SELL"]

        if not buys.empty:
            premium_paid = (buys["entry_price"] * buys["lots"] * buys["lot_size"]).sum()
        if not sells.empty:
            premium_received = (sells["entry_price"] * sells["lots"] * sells["lot_size"]).sum()
            margin_blocked = (sells["entry_price"] * sells["lots"] * sells["lot_size"] * 4).sum()

        if not chain_df.empty:
            enriched = compute_live_pnl(open_trades, chain_df)
            unrealized = enriched["unrealized_pnl"].sum()

    # Cash after accounting for open orders
    # BUY: premium leaves the account
    # SELL: premium enters but margin is blocked
    cash_after_orders = cash - premium_paid + premium_received

    # Net value = remaining cash + market value of positions (via unrealized P&L)
    # unrealized already captures (current_value - entry_cost) for all positions
    net_value = cash + unrealized  # simpler: initial + realized + unrealized
    deployed = premium_paid + margin_blocked
    available = max(cash_after_orders - margin_blocked, 0)

    return {
        "initial_capital": capital,
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unrealized, 2),
        "net_value": round(net_value, 2),
        "premium_paid": round(premium_paid, 2),
        "premium_received": round(premium_received, 2),
        "margin_blocked": round(margin_blocked, 2),
        "capital_deployed": round(deployed, 2),
        "available_capital": round(available, 2),
        "total_trades": total_trades,
        "winning_trades": wins,
        "win_rate": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
        "open_positions": len(open_trades),
    }
