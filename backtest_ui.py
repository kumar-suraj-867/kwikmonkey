"""Streamlit UI for the Backtest tab."""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data_store
from backtest_engine import (
    BacktestResult,
    run_backtest,
)
from data_fetcher import FyersDataFetcher
from data_collector import DataCollector
from history_fetcher import HistoryFetcher


def render_backtest_tab(fetcher: FyersDataFetcher):
    """Main entry point for the Backtest tab."""
    st.header("📈 Strategy Backtester")

    history = HistoryFetcher(fetcher.fyers)

    # ------------------------------------------------------------------
    # Data collection controls
    # ------------------------------------------------------------------
    _render_data_collection(fetcher)

    st.divider()

    # ------------------------------------------------------------------
    # Input controls
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("Backtest Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            strategy = st.selectbox(
                "Strategy",
                ["Iron Condor", "Bull Call Spread", "Bear Put Spread", "Long Straddle"],
            )
        with col2:
            start_date = st.date_input(
                "Start date",
                value=date.today() - timedelta(days=90),
            )
        with col3:
            end_date = st.date_input(
                "End date",
                value=date.today() - timedelta(days=1),
            )

        # Strategy-specific parameters
        st.markdown("**Strike parameters**")
        if strategy == "Iron Condor":
            c1, c2, c3 = st.columns(3)
            ce_offset = c1.number_input("CE offset from ATM (pts)", value=200, step=50, min_value=50)
            pe_offset = c2.number_input("PE offset from ATM (pts)", value=200, step=50, min_value=50)
            wing_width = c3.number_input("Wing width (pts)", value=50, step=50, min_value=50)
        elif strategy == "Bull Call Spread":
            ce_offset = st.number_input("Sell CE offset from ATM (pts)", value=200, step=50, min_value=50)
            pe_offset = 0
            wing_width = 0
        elif strategy == "Bear Put Spread":
            pe_offset = st.number_input("Sell PE offset from ATM (pts)", value=200, step=50, min_value=50)
            ce_offset = 0
            wing_width = 0
        else:  # Long Straddle
            ce_offset = 0
            pe_offset = 0
            wing_width = 0

        st.markdown("**Risk management & model**")
        c1, c2, c3, c4 = st.columns(4)
        stop_loss_pct = c1.number_input("Stop-loss %", value=50.0, step=5.0, min_value=10.0, max_value=100.0)
        target_pct = c2.number_input("Target %", value=50.0, step=5.0, min_value=10.0, max_value=200.0)
        lot_size = c3.number_input("Lot size", value=config.NIFTY_LOT_SIZE, step=25, min_value=25)
        days_before = c4.number_input("Entry days before expiry", value=4, step=1, min_value=1, max_value=7)

        st.markdown("**Execution & pricing**")
        c1, c2, c3, c4 = st.columns(4)
        iv_pct = c1.number_input("IV assumption %", value=15.0, step=1.0, min_value=5.0, max_value=50.0,
                                 help="Used when local DB data is unavailable (BS model fallback)")
        slippage = c2.number_input("Slippage (pts)", value=1.0, step=0.5, min_value=0.0, max_value=10.0,
                                   help="Added to buy price, subtracted from sell price")
        brokerage = c3.number_input("Brokerage (₹/lot)", value=20.0, step=5.0, min_value=0.0,
                                    help="Per lot per trade (entry + exit)")
        resolution = c4.selectbox("Candle resolution", ["15", "30", "60", "D"], index=0)

    # ------------------------------------------------------------------
    # Run button
    # ------------------------------------------------------------------
    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    if not run_btn:
        st.info(
            "Uses **local DB data** when available, falls back to **Black-Scholes model** "
            "from NIFTY spot candles. Start collecting data above to build your dataset."
        )
        if "backtest_result" in st.session_state:
            # Clear stale results from old engine version
            old = st.session_state["backtest_result"]
            if old.trades and not hasattr(old.trades[0], "data_source"):
                del st.session_state["backtest_result"]
            else:
                _render_results(old)
        return

    # ------------------------------------------------------------------
    # Execute backtest
    # ------------------------------------------------------------------
    progress_bar = st.progress(0, text="Starting backtest...")

    def on_progress(current, total, message):
        pct = current / total if total > 0 else 0
        progress_bar.progress(pct, text=f"{message} ({current}/{total})")

    result = run_backtest(
        history_fetcher=history,
        strategy_name=strategy,
        start_date=start_date,
        end_date=end_date,
        ce_offset=ce_offset,
        pe_offset=pe_offset,
        wing_width=wing_width,
        lot_size=lot_size,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
        days_before_expiry=days_before,
        resolution=resolution,
        iv=iv_pct / 100,
        slippage_pts=slippage,
        brokerage_per_lot=brokerage,
        progress_callback=on_progress,
    )

    progress_bar.empty()
    st.session_state["backtest_result"] = result
    _render_results(result)


# ======================================================================
# Data collection section
# ======================================================================

def _render_data_collection(fetcher: FyersDataFetcher):
    """Render data collection controls and DB stats."""
    with st.container(border=True):
        st.subheader("📦 Data Collection")
        st.caption("Collect live option chain snapshots to build your own historical dataset")

        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            interval = st.selectbox(
                "Collection interval",
                [60, 120, 180, 300],
                format_func=lambda x: f"{x//60} min" if x >= 60 else f"{x} sec",
                index=0,
            )

        with col2:
            st.write("")  # spacer
            st.write("")
            # Collector instance persisted in session state
            if "data_collector" not in st.session_state:
                st.session_state["data_collector"] = DataCollector(fetcher, interval)

            collector = st.session_state["data_collector"]
            collector.interval_sec = interval

            if collector.is_running:
                if st.button("⏹ Stop Collecting", type="secondary"):
                    collector.stop()
                    st.rerun()
            else:
                if st.button("▶ Start Collecting", type="primary"):
                    collector.start()
                    st.rerun()

        with col3:
            status = collector.status()
            if status["running"]:
                st.success(f"Collecting every {interval//60} min")
                if status["last_snapshot"]:
                    st.caption(f"Last snapshot: {status['last_snapshot']} | Total: {status['total_snapshots']}")
                if status["last_error"]:
                    st.caption(f"Last error: {status['last_error']}")
            else:
                st.info("Not collecting")

        # DB stats
        stats = data_store.get_db_stats()
        if stats.get("exists") and stats["option_rows"] > 0:
            st.markdown("**Stored data:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Option snapshots", f"{stats['option_rows']:,}")
            c2.metric("Spot snapshots", f"{stats['spot_rows']:,}")
            c3.metric("Trading days", stats["trading_days"])
            c4.metric("DB size", f"{stats['db_size_mb']} MB")
            if stats["option_from"]:
                st.caption(f"Range: {stats['option_from']} → {stats['option_to']}")
        else:
            st.caption("No data collected yet. Start collecting to build your backtest dataset.")


# ======================================================================
# Results rendering
# ======================================================================

def _render_results(result: BacktestResult):
    """Render full backtest results."""
    st.divider()
    s = result.summary

    if s.get("executed_trades", 0) == 0:
        st.warning("No trades were executed. Check date range and data availability.")
        if result.warnings:
            with st.expander("Warnings"):
                for w in result.warnings:
                    st.caption(w)
        return

    # ------------------------------------------------------------------
    # Data source indicator
    # ------------------------------------------------------------------
    sources = s.get("data_sources", {})
    if sources:
        source_parts = []
        for src, count in sources.items():
            label = "Local DB" if src == "db" else "BS Model" if src == "bs_model" else "Mixed"
            source_parts.append(f"{label}: {count}")
        st.caption(f"Data source: {' | '.join(source_parts)}")

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    st.subheader(f"Results — {result.strategy_name}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pnl_color = "normal" if s["total_pnl"] >= 0 else "inverse"
    c1.metric("Total P&L", f"₹{s['total_pnl']:,.0f}", delta_color=pnl_color)
    c2.metric("Win Rate", f"{s['win_rate']}%")
    c3.metric("Trades", f"{s['executed_trades']} / {s['total_trades']}")
    c4.metric("Profit Factor", f"{s['profit_factor']:.2f}")
    c5.metric("Max Drawdown", f"₹{s['max_drawdown']:,.0f}")
    c6.metric("Expectancy", f"₹{s['expectancy']:,.0f}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Avg Win", f"₹{s['avg_win']:,.0f}")
    col2.metric("Avg Loss", f"₹{s['avg_loss']:,.0f}")
    col3.metric("Best Trade", f"₹{s['best_trade']:,.0f}")
    col4.metric("Worst Trade", f"₹{s['worst_trade']:,.0f}")
    col5.metric("Sharpe Ratio", f"{s.get('sharpe_ratio', 0):.2f}")
    col6.metric("Sortino Ratio", f"{s.get('sortino_ratio', 0):.2f}")

    # Exit reason breakdown
    if s.get("exit_reasons"):
        cols = st.columns(len(s["exit_reasons"]))
        for i, (reason, count) in enumerate(s["exit_reasons"].items()):
            cols[i].metric(f"Exit: {reason}", count)

    st.divider()

    # ------------------------------------------------------------------
    # Equity curve & P&L distribution
    # ------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Equity Curve**")
        valid_trades = [t for t in result.trades if t.exit_reason != "skipped"]
        if valid_trades and result.equity_curve:
            eq_dates = [t.expiry for t in result.trades]
            eq_values = result.equity_curve

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_dates, y=eq_values,
                mode="lines+markers",
                fill="tozeroy",
                line=dict(color="#22c55e" if eq_values[-1] >= 0 else "#ef4444"),
                marker=dict(size=6),
                name="Cumulative P&L",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                xaxis_title="Expiry Date",
                yaxis_title="Cumulative P&L (₹)",
                template="plotly_dark",
                height=400,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**P&L Distribution**")
        pnls = [t.total_pnl for t in valid_trades]
        if pnls:
            colors = ["#22c55e" if p >= 0 else "#ef4444" for p in pnls]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[t.expiry.strftime("%d-%b") for t in valid_trades],
                y=pnls,
                marker_color=colors,
                name="P&L",
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(
                xaxis_title="Expiry",
                yaxis_title="P&L (₹)",
                template="plotly_dark",
                height=400,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Trade log table
    # ------------------------------------------------------------------
    st.markdown("**Trade Log**")

    rows = []
    for t in result.trades:
        legs_str = ""
        entry_total = 0
        exit_total = 0
        if t.exit_reason != "skipped" and t.legs_detail:
            leg_parts = []
            for ld in t.legs_detail:
                if "error" not in ld:
                    leg_parts.append(f"{ld['action']} {ld['type']} {ld['strike']}")
                    entry_total += ld["entry_price"] * (1 if ld["action"] == "BUY" else -1)
                    exit_total += ld["exit_price"] * (1 if ld["action"] == "BUY" else -1)
            legs_str = " | ".join(leg_parts)

        rows.append({
            "Expiry": t.expiry,
            "Entry": t.entry_date,
            "ATM": t.atm_strike,
            "Legs": legs_str,
            "Net Entry": round(entry_total, 2),
            "Net Exit": round(exit_total, 2),
            "P&L (₹)": t.total_pnl,
            "Peak": t.peak_pnl,
            "Trough": t.trough_pnl,
            "Exit": t.exit_reason,
            "Source": getattr(t, "data_source", ""),
        })

    trade_df = pd.DataFrame(rows)
    st.dataframe(trade_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Warnings & params
    # ------------------------------------------------------------------
    if result.warnings:
        with st.expander(f"Warnings ({len(result.warnings)})"):
            for w in result.warnings:
                st.caption(w)

    with st.expander("Backtest parameters"):
        params_df = pd.DataFrame([result.params]).T
        params_df.columns = ["Value"]
        st.dataframe(params_df, use_container_width=True)
