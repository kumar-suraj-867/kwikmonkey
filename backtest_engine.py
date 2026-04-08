"""Backtesting engine for NIFTY option strategies.

Hybrid approach:
  1. Uses locally collected option chain data (SQLite) when available
  2. Falls back to Black-Scholes synthetic pricing from NIFTY spot candles
  3. Adds realistic execution costs (slippage + brokerage)
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import config
import data_store
from metrics import bs_price


# ======================================================================
# Helpers
# ======================================================================

def determine_atm_strike(spot_price: float, step: int = None) -> int:
    """Round spot price to nearest strike step."""
    if step is None:
        step = config.NIFTY_STRIKE_STEP
    return int(round(spot_price / step) * step)


# ======================================================================
# Expiry generation
# ======================================================================

def generate_weekly_expiries(start: date, end: date) -> list[date]:
    """Generate all Thursdays (NIFTY weekly expiry) between start and end."""
    expiries = []
    d = start
    while d.weekday() != 3:  # Thursday
        d += timedelta(days=1)
    while d <= end:
        expiries.append(d)
        d += timedelta(weeks=1)
    return expiries


def entry_date_for_expiry(expiry: date, days_before: int = 4) -> date:
    """Calculate entry date (business day) N days before expiry."""
    d = expiry - timedelta(days=days_before)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ======================================================================
# Strategy leg construction
# ======================================================================

@dataclass
class Leg:
    action: str       # "BUY" or "SELL"
    option_type: str  # "CE" or "PE"
    strike_offset: int  # offset from ATM in points

    @property
    def direction(self) -> int:
        return 1 if self.action == "BUY" else -1


def construct_strategy_legs(strategy: str, ce_offset: int = 200,
                            pe_offset: int = 200,
                            wing_width: int = 50) -> list[Leg]:
    """Return list of Leg definitions for a strategy."""
    if strategy == "Iron Condor":
        return [
            Leg("SELL", "CE", ce_offset),
            Leg("BUY",  "CE", ce_offset + wing_width),
            Leg("SELL", "PE", -pe_offset),
            Leg("BUY",  "PE", -(pe_offset + wing_width)),
        ]
    elif strategy == "Bull Call Spread":
        return [
            Leg("BUY",  "CE", 0),
            Leg("SELL", "CE", ce_offset),
        ]
    elif strategy == "Bear Put Spread":
        return [
            Leg("BUY",  "PE", 0),
            Leg("SELL", "PE", -pe_offset),
        ]
    elif strategy == "Long Straddle":
        return [
            Leg("BUY", "CE", 0),
            Leg("BUY", "PE", 0),
        ]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ======================================================================
# Data source abstraction
# ======================================================================

def _get_option_prices_from_db(strike: float, option_type: str,
                               entry_dt: date, expiry: date) -> pd.DataFrame | None:
    """Try to get recorded option prices from local DB.

    Returns DataFrame with columns: ts, ltp, oi, iv, spot_price
    or None if insufficient data.
    """
    from_dt = datetime.combine(entry_dt, datetime.min.time())
    to_dt = datetime.combine(expiry, datetime.max.time().replace(microsecond=0))
    expiry_str = expiry.strftime("%Y-%m-%d")

    df = data_store.get_option_history(strike, option_type, from_dt, to_dt, expiry_str)
    if df.empty or len(df) < 2:
        return None
    return df


def _compute_bs_prices(spot_candles: pd.DataFrame, strike: int,
                       option_type: str, expiry: date,
                       iv: float, r: float) -> pd.DataFrame:
    """Compute synthetic option prices using Black-Scholes from spot candles.

    Returns DataFrame with columns: ts, ltp (BS price), spot_price
    """
    expiry_dt = datetime.combine(expiry, datetime.min.time()).replace(hour=15, minute=30)

    results = []
    for _, row in spot_candles.iterrows():
        ts = row["timestamp"]
        spot = row["close"]
        T = max((expiry_dt - ts).total_seconds() / (365.25 * 86400), 0)

        if T <= 0:
            price = max(spot - strike, 0) if option_type == "CE" else max(strike - spot, 0)
        else:
            price = bs_price(spot, strike, T, r, iv, option_type)

        results.append({"ts": ts, "ltp": price, "spot_price": spot})

    return pd.DataFrame(results)


# ======================================================================
# Trade simulation
# ======================================================================

@dataclass
class TradeResult:
    expiry: date
    entry_date: date
    atm_strike: int
    legs_detail: list[dict]
    total_pnl: float
    exit_reason: str          # "target", "stoploss", "expiry", "skipped"
    data_source: str = ""     # "db", "bs_model", "mixed"
    peak_pnl: float = 0
    trough_pnl: float = 0


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    trades: list[TradeResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def simulate_single_expiry(spot_candles: pd.DataFrame, expiry: date,
                           atm_strike: int, legs: list[Leg], lot_size: int,
                           stop_loss_pct: float, target_pct: float,
                           entry_dt: date, iv: float,
                           slippage_pts: float = 0, brokerage_per_lot: float = 0,
                           r: float = None) -> TradeResult:
    """Simulate one strategy instance for a single expiry.

    Tries local DB data first, falls back to BS model.
    """
    if r is None:
        r = config.RISK_FREE_RATE

    # Filter spot candles to entry period
    candles = spot_candles[spot_candles["timestamp"].dt.date >= entry_dt].copy()
    if candles.empty:
        return TradeResult(
            expiry=expiry, entry_date=entry_dt, atm_strike=atm_strike,
            legs_detail=[{"error": "no spot candles for entry period"}],
            total_pnl=0, exit_reason="skipped",
        )
    candles = candles.reset_index(drop=True)

    # Build price series for each leg
    leg_info = []
    data_sources = set()

    for leg in legs:
        strike = atm_strike + leg.strike_offset

        # Try DB first
        db_data = _get_option_prices_from_db(strike, leg.option_type, entry_dt, expiry)
        if db_data is not None and len(db_data) >= 2:
            prices = db_data.rename(columns={"ts": "timestamp"})
            data_sources.add("db")
        else:
            # Fall back to BS model
            prices = _compute_bs_prices(candles, strike, leg.option_type, expiry, iv, r)
            prices = prices.rename(columns={"ts": "timestamp"})
            data_sources.add("bs_model")

        if prices.empty:
            return TradeResult(
                expiry=expiry, entry_date=entry_dt, atm_strike=atm_strike,
                legs_detail=[{"error": f"no price data for {leg.option_type} {strike}"}],
                total_pnl=0, exit_reason="skipped",
            )

        leg_info.append({
            "leg": leg,
            "strike": strike,
            "prices": prices,
            "entry_price": prices.iloc[0]["ltp"],
        })

    # Determine data source label
    if data_sources == {"db"}:
        source = "db"
    elif data_sources == {"bs_model"}:
        source = "bs_model"
    else:
        source = "mixed"

    # Apply slippage to entry prices
    for li in leg_info:
        if li["leg"].action == "BUY":
            li["entry_price"] += slippage_pts  # pay more
        else:
            li["entry_price"] -= slippage_pts  # receive less

    # Net entry credit/debit
    net_entry = sum(li["entry_price"] * li["leg"].direction for li in leg_info)

    # Max risk for SL/TP thresholds
    has_sell = any(leg.action == "SELL" for leg in legs)
    has_buy = any(leg.action == "BUY" for leg in legs)
    if has_sell and has_buy:
        ce_strikes = [atm_strike + l.strike_offset for l in legs if l.option_type == "CE"]
        pe_strikes = [atm_strike + l.strike_offset for l in legs if l.option_type == "PE"]
        ce_width = (max(ce_strikes) - min(ce_strikes)) if len(ce_strikes) > 1 else 0
        pe_width = (max(pe_strikes) - min(pe_strikes)) if len(pe_strikes) > 1 else 0
        max_width = max(ce_width, pe_width)
        max_risk = max_width - abs(net_entry) if max_width > 0 else abs(net_entry)
    else:
        max_risk = abs(net_entry)

    max_risk_amount = max_risk * lot_size
    sl_threshold = -max_risk_amount * stop_loss_pct / 100
    tp_threshold = abs(net_entry) * lot_size * target_pct / 100

    # Use the shortest price series length as reference
    n_candles = min(len(li["prices"]) for li in leg_info)

    peak_pnl = 0.0
    trough_pnl = 0.0
    exit_reason = "expiry"
    exit_idx = n_candles - 1

    for i in range(n_candles):
        pnl = 0.0
        for li in leg_info:
            current_price = li["prices"].iloc[i]["ltp"]
            leg_pnl = (current_price - li["entry_price"]) * li["leg"].direction * lot_size
            pnl += leg_pnl

        peak_pnl = max(peak_pnl, pnl)
        trough_pnl = min(trough_pnl, pnl)

        if sl_threshold < 0 and pnl <= sl_threshold:
            exit_reason = "stoploss"
            exit_idx = i
            break
        if tp_threshold > 0 and pnl >= tp_threshold:
            exit_reason = "target"
            exit_idx = i
            break

    # Final P&L at exit
    total_pnl = 0.0
    legs_detail = []
    for li in leg_info:
        idx = min(exit_idx, len(li["prices"]) - 1)
        exit_price = li["prices"].iloc[idx]["ltp"]

        # Apply slippage to exit
        if li["leg"].action == "BUY":
            exit_price -= slippage_pts  # sell for less
        else:
            exit_price += slippage_pts  # buy back for more

        leg_pnl = (exit_price - li["entry_price"]) * li["leg"].direction * lot_size
        total_pnl += leg_pnl

        legs_detail.append({
            "strike": li["strike"],
            "action": li["leg"].action,
            "type": li["leg"].option_type,
            "entry_price": round(li["entry_price"], 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(leg_pnl, 2),
        })

    # Deduct brokerage (per lot, per leg, entry + exit = 2 trades per leg)
    total_brokerage = brokerage_per_lot * len(legs) * 2
    total_pnl -= total_brokerage

    return TradeResult(
        expiry=expiry,
        entry_date=entry_dt,
        atm_strike=atm_strike,
        legs_detail=legs_detail,
        total_pnl=round(total_pnl, 2),
        exit_reason=exit_reason,
        data_source=source,
        peak_pnl=round(peak_pnl, 2),
        trough_pnl=round(trough_pnl, 2),
    )


# ======================================================================
# Full backtest orchestrator
# ======================================================================

def run_backtest(history_fetcher, strategy_name: str,
                 start_date: date, end_date: date,
                 ce_offset: int = 200, pe_offset: int = 200,
                 wing_width: int = 50, lot_size: int = None,
                 stop_loss_pct: float = 50, target_pct: float = 50,
                 days_before_expiry: int = 4,
                 resolution: str = "15",
                 iv: float = 0.15,
                 slippage_pts: float = 1.0,
                 brokerage_per_lot: float = 20.0,
                 progress_callback=None) -> BacktestResult:
    """Run a full backtest across multiple weekly expiries.

    Hybrid approach:
      - Tries local DB for actual recorded option prices
      - Falls back to BS model using NIFTY spot candles from Fyers
      - Applies slippage and brokerage for realistic simulation
    """
    if lot_size is None:
        lot_size = config.NIFTY_LOT_SIZE

    # Ensure DB is initialised
    data_store.init_db()

    legs_template = construct_strategy_legs(strategy_name, ce_offset, pe_offset, wing_width)

    expanded_end = end_date + timedelta(days=7)
    expiries = generate_weekly_expiries(start_date, expanded_end)

    result = BacktestResult(
        strategy_name=strategy_name,
        params={
            "ce_offset": ce_offset, "pe_offset": pe_offset,
            "wing_width": wing_width, "lot_size": lot_size,
            "stop_loss_pct": stop_loss_pct, "target_pct": target_pct,
            "days_before_expiry": days_before_expiry,
            "resolution": resolution, "iv": f"{iv*100:.0f}%",
            "slippage": f"{slippage_pts} pts",
            "brokerage": f"₹{brokerage_per_lot}/lot",
            "start_date": str(start_date), "end_date": str(end_date),
        },
    )

    cumulative_pnl = 0.0

    if not expiries:
        result.warnings.append(f"No weekly expiries found between {start_date} and {expanded_end}")

    for i, expiry in enumerate(expiries):
        if progress_callback:
            progress_callback(i, len(expiries), f"Processing expiry {expiry}")

        entry_dt = entry_date_for_expiry(expiry, days_before_expiry)

        # Fetch spot candles from Fyers
        window_start = entry_dt - timedelta(days=5)
        spot_candles = history_fetcher.get_spot_candles(
            window_start.strftime("%Y-%m-%d"),
            expiry.strftime("%Y-%m-%d"),
            resolution,
        )

        if spot_candles.empty:
            err = history_fetcher.last_error or "unknown"
            result.warnings.append(f"No spot data near {entry_dt} for expiry {expiry}: {err}")
            continue

        # Find actual trading day for entry
        on_or_after = spot_candles[spot_candles["timestamp"].dt.date >= entry_dt]
        if not on_or_after.empty:
            actual_entry_dt = on_or_after.iloc[0]["timestamp"].date()
        else:
            actual_entry_dt = spot_candles.iloc[-1]["timestamp"].date()

        # ATM from spot at entry
        entry_candles = spot_candles[spot_candles["timestamp"].dt.date == actual_entry_dt]
        if entry_candles.empty:
            entry_candles = on_or_after if not on_or_after.empty else spot_candles
        spot_at_entry = entry_candles.iloc[0]["open"]
        atm_strike = determine_atm_strike(spot_at_entry)

        trade = simulate_single_expiry(
            spot_candles, expiry, atm_strike, legs_template,
            lot_size, stop_loss_pct, target_pct, actual_entry_dt, iv,
            slippage_pts, brokerage_per_lot,
        )

        result.trades.append(trade)

        if trade.exit_reason == "skipped":
            detail = trade.legs_detail[0] if trade.legs_detail else {}
            err_msg = detail.get("error", "unknown")
            result.warnings.append(f"Expiry {expiry}: skipped — {err_msg}")
        else:
            cumulative_pnl += trade.total_pnl
        result.equity_curve.append(cumulative_pnl)

    if progress_callback:
        progress_callback(len(expiries), len(expiries), "Done")

    result.summary = calculate_summary(result.trades)
    return result


# ======================================================================
# Summary statistics
# ======================================================================

def calculate_summary(trades: list[TradeResult]) -> dict:
    """Compute backtest summary stats from trade list."""
    valid = [t for t in trades if t.exit_reason != "skipped"]

    if not valid:
        return {
            "total_trades": 0, "executed_trades": 0,
            "skipped": len(trades), "total_pnl": 0,
            "win_rate": 0, "avg_win": 0, "avg_loss": 0,
            "max_drawdown": 0, "profit_factor": 0, "expectancy": 0,
            "sharpe_ratio": 0, "sortino_ratio": 0,
        }

    pnls = [t.total_pnl for t in valid]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(winners) / len(valid) * 100 if valid else 0
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from cumulative P&L
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * abs(avg_loss))

    # Sharpe ratio (annualised, assuming weekly trades)
    pnl_arr = np.array(pnls)
    mean_ret = pnl_arr.mean()
    std_ret = pnl_arr.std()
    sharpe = (mean_ret / std_ret * np.sqrt(52)) if std_ret > 0 else 0  # 52 weeks/year

    # Sortino ratio (only downside deviation)
    downside = pnl_arr[pnl_arr < 0]
    downside_std = downside.std() if len(downside) > 1 else 0
    sortino = (mean_ret / downside_std * np.sqrt(52)) if downside_std > 0 else 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in valid:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Data source breakdown
    data_sources = {}
    for t in valid:
        src = t.data_source or "unknown"
        data_sources[src] = data_sources.get(src, 0) + 1

    return {
        "total_trades": len(trades),
        "executed_trades": len(valid),
        "skipped": len(trades) - len(valid),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 1),
        "winners": len(winners),
        "losers": len(losers),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "exit_reasons": exit_reasons,
        "data_sources": data_sources,
    }
