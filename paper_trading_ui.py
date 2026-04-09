"""Streamlit UI for the Paper Trading tab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from data_fetcher import FyersDataFetcher
from paper_trading import (
    get_or_create_account,
    reset_account,
    place_order,
    close_position,
    close_group,
    get_open_trades,
    get_closed_trades,
    compute_live_pnl,
    check_sl_target,
    compute_portfolio_stats,
)


def _active_profile() -> dict:
    """Get active index profile from session state."""
    return st.session_state.get(
        "_index_profile", config.INDEX_PROFILES["NIFTY 50"]
    )


def render_paper_trading_tab(fetcher: FyersDataFetcher):
    """Main entry point for the Paper Trading tab."""
    st.header("📝 Paper Trading")

    account = get_or_create_account()

    # Reuse data already fetched by the Live Dashboard tab (shared cache)
    spot = st.session_state.get("_spot")
    chain_df = st.session_state.get("_enriched_chain", pd.DataFrame())
    expiry_list = st.session_state.get("_expiry_data", [])

    # Fallback: fetch only if dashboard hasn't populated the cache yet
    if spot is None or chain_df.empty:
        try:
            spot = fetcher.get_spot_quote()
            chain_df, expiry_list = fetcher.get_option_chain_with_expiries()
        except Exception as e:
            st.error(f"Cannot fetch market data: {e}")
            return

    if chain_df.empty:
        st.warning("No option chain data available.")
        return

    spot_ltp = spot["ltp"] if isinstance(spot, dict) else spot

    # Auto SL/Target check on every refresh
    open_trades = get_open_trades()
    sl_tgt_msgs = check_sl_target(open_trades, chain_df)
    if sl_tgt_msgs:
        for msg in sl_tgt_msgs:
            st.toast(msg, icon="🔔")
        open_trades = get_open_trades()  # refresh after closes
        account = get_or_create_account()

    # Portfolio metrics
    stats = compute_portfolio_stats(account, open_trades, chain_df)
    _render_portfolio_header(stats, spot_ltp, spot)

    st.divider()

    # Order form + Quick strategies
    col_order, col_quick = st.columns([3, 2])
    with col_order:
        _render_order_form(chain_df, expiry_list, spot_ltp)
    with col_quick:
        _render_quick_strategies(chain_df, expiry_list, spot_ltp)

    st.divider()

    # Open positions
    _render_open_positions(open_trades, chain_df)

    st.divider()

    # Trade history + equity curve
    _render_trade_history()

    # Reset button (in expander to avoid accidental clicks)
    with st.expander("Account Settings"):
        c1, c2 = st.columns(2)
        new_capital = c1.number_input("Initial Capital", value=int(account["initial_capital"]),
                                      step=100000, min_value=100000)
        if c2.button("Reset Account", type="secondary",
                     help="Wipe all trades and start fresh"):
            reset_account(new_capital)
            st.rerun()


# ======================================================================
# Portfolio header
# ======================================================================

def _render_portfolio_header(stats: dict, spot_ltp: float, spot: dict):
    """Render portfolio summary metrics with capital breakdown."""
    # Row 1: Key P&L metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    net_change = stats["realized_pnl"] + stats["unrealized_pnl"]
    net_pct = (net_change / stats["initial_capital"] * 100) if stats["initial_capital"] > 0 else 0

    c1.metric("Net Value", f"₹{stats['net_value']:,.0f}",
              delta=f"{net_change:+,.0f} ({net_pct:+.1f}%)")
    c2.metric("Unrealized P&L", f"₹{stats['unrealized_pnl']:,.0f}",
              delta_color="normal" if stats["unrealized_pnl"] >= 0 else "inverse")
    c3.metric("Realized P&L", f"₹{stats['realized_pnl']:,.0f}",
              delta_color="normal" if stats["realized_pnl"] >= 0 else "inverse")
    c4.metric("Win Rate", f"{stats['win_rate']}%" if stats["total_trades"] > 0 else "—",
              delta=f"{stats['winning_trades']}/{stats['total_trades']}")
    c5.metric("Open Positions", stats["open_positions"])
    c6.metric(f"{_active_profile()['name']} Spot", f"₹{spot_ltp:,.2f}",
              delta=f"{spot.get('change', 0):+.2f} ({spot.get('change_pct', 0):+.2f}%)")

    # Row 2: Capital breakdown
    if stats["open_positions"] > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Deployed", f"₹{stats['capital_deployed']:,.0f}",
                  help="Premium paid for buys + margin blocked for sells")
        c2.metric("Premium Paid", f"₹{stats['premium_paid']:,.0f}",
                  help="Total premium paid for BUY positions")
        c3.metric("Premium Received", f"₹{stats['premium_received']:,.0f}",
                  help="Total premium received for SELL positions")
        c4.metric("Available Capital", f"₹{stats['available_capital']:,.0f}",
                  help="Cash remaining after open orders")


# ======================================================================
# Order entry form
# ======================================================================

def _render_order_form(chain_df: pd.DataFrame, expiry_list: list, spot: float):
    """Single-leg order entry form."""
    st.subheader("Place Order")

    # Expiry selector
    expiry_labels = []
    expiry_map = {}
    for exp in expiry_list:
        label = exp.get("date", str(exp.get("expiry", "")))
        expiry_labels.append(label)
        expiry_map[label] = exp
    if not expiry_labels:
        st.warning("No expiries available")
        return

    c1, c2, c3 = st.columns(3)
    selected_expiry = c1.selectbox("Expiry", expiry_labels, key="pt_expiry")
    option_type = c2.radio("Type", ["CE", "PE"], horizontal=True, key="pt_type")
    action = c3.radio("Action", ["BUY", "SELL"], horizontal=True, key="pt_action")

    # Strike selector — show nearby ATM strikes
    atm = round(spot / _active_profile()["strike_step"]) * _active_profile()["strike_step"]
    available_strikes = sorted(chain_df[chain_df["option_type"] == option_type]["strike"].unique())
    if not available_strikes:
        st.warning("No strikes available")
        return

    # Default to ATM
    default_idx = 0
    for i, s in enumerate(available_strikes):
        if s >= atm:
            default_idx = i
            break

    c1, c2 = st.columns(2)
    strike = c1.selectbox(
        "Strike", available_strikes, index=default_idx, key="pt_strike",
        format_func=lambda s: f"₹{s:,.0f}" + (" (ATM)" if s == atm else
                              f" (ITM {abs(s - atm):.0f})" if (option_type == "CE" and s < atm) or (option_type == "PE" and s > atm) else
                              f" (OTM +{abs(s - atm):.0f})")
    )
    lots = c2.number_input("Lots", value=1, min_value=1, max_value=100, step=1, key="pt_lots")

    # Show live price for selected strike
    match = chain_df[(chain_df["strike"] == strike) & (chain_df["option_type"] == option_type)]
    if not match.empty:
        row = match.iloc[0]
        ltp = row["ltp"]
        bid = row.get("bid", 0)
        ask = row.get("ask", 0)
        iv = row.get("iv", 0)
        oi = row.get("oi", 0)

        st.markdown(
            f"**LTP:** ₹{ltp:.2f} &nbsp;|&nbsp; "
            f"**Bid:** ₹{bid:.2f} &nbsp;|&nbsp; "
            f"**Ask:** ₹{ask:.2f} &nbsp;|&nbsp; "
            f"**IV:** {iv:.1f}% &nbsp;|&nbsp; "
            f"**OI:** {oi:,.0f}"
        )

        # Premium / margin
        active_lot_size = _active_profile()["lot_size"]
        premium = ltp * lots * active_lot_size
        if action == "BUY":
            st.caption(f"Premium: ₹{premium:,.0f} ({lots} lot × {active_lot_size} × ₹{ltp:.2f})")
        else:
            margin_est = premium * 4  # rough SPAN margin
            st.caption(f"Est. margin: ₹{margin_est:,.0f} | Premium received: ₹{premium:,.0f}")
    else:
        ltp = 0
        st.warning("No live price for selected strike")

    # SL / Target
    c1, c2 = st.columns(2)
    use_sl = c1.checkbox("Set Stop Loss", key="pt_use_sl")
    use_tgt = c2.checkbox("Set Target", key="pt_use_tgt")

    sl_price = 0.0
    target_price = 0.0
    if use_sl:
        if action == "BUY":
            sl_default = round(ltp * 0.5, 2) if ltp > 0 else 0
            sl_price = st.number_input("SL Price (option drops to)", value=sl_default,
                                       step=1.0, min_value=0.0, key="pt_sl")
        else:
            sl_default = round(ltp * 1.5, 2) if ltp > 0 else 0
            sl_price = st.number_input("SL Price (option rises to)", value=sl_default,
                                       step=1.0, min_value=0.0, key="pt_sl")
    if use_tgt:
        if action == "BUY":
            tgt_default = round(ltp * 1.5, 2) if ltp > 0 else 0
            target_price = st.number_input("Target Price (option rises to)", value=tgt_default,
                                           step=1.0, min_value=0.0, key="pt_tgt")
        else:
            tgt_default = round(ltp * 0.5, 2) if ltp > 0 else 0
            target_price = st.number_input("Target Price (option drops to)", value=tgt_default,
                                           step=1.0, min_value=0.0, key="pt_tgt")

    # Place order button
    if st.button(f"{'🟢 BUY' if action == 'BUY' else '🔴 SELL'} {lots} lot {option_type} {strike:.0f}",
                 type="primary", key="pt_place"):
        if ltp <= 0:
            st.error("Cannot place order — no live price available")
        else:
            trade_id = place_order(
                expiry_date=selected_expiry,
                strike=strike,
                option_type=option_type,
                action=action,
                lots=lots,
                entry_price=ltp,
                sl_price=sl_price if use_sl else None,
                target_price=target_price if use_tgt else None,
                lot_size=_active_profile()["lot_size"],
            )
            st.success(f"Order placed: {action} {lots}L {option_type} {strike:.0f} @ ₹{ltp:.2f}")
            st.rerun()


# ======================================================================
# Quick strategy shortcuts
# ======================================================================

def _render_quick_strategies(chain_df: pd.DataFrame, expiry_list: list, spot: float):
    """One-click strategy entry."""
    st.subheader("Quick Strategies")

    atm = round(spot / _active_profile()["strike_step"]) * _active_profile()["strike_step"]

    expiry_labels = [exp.get("date", str(exp.get("expiry", ""))) for exp in expiry_list]
    if not expiry_labels:
        return

    selected_expiry = expiry_labels[0]  # nearest expiry

    c1, c2 = st.columns(2)
    lots = c1.number_input("Lots", value=1, min_value=1, max_value=50, step=1, key="qs_lots")
    offset = c2.number_input("OTM offset (pts)", value=200, step=50, min_value=50, key="qs_offset")

    def _get_price(strike, opt_type):
        m = chain_df[(chain_df["strike"] == strike) & (chain_df["option_type"] == opt_type)]
        return m.iloc[0]["ltp"] if not m.empty else 0

    strategies = {
        "Long Straddle": [
            ("BUY", "CE", atm),
            ("BUY", "PE", atm),
        ],
        "Short Straddle": [
            ("SELL", "CE", atm),
            ("SELL", "PE", atm),
        ],
        "Long Strangle": [
            ("BUY", "CE", atm + offset),
            ("BUY", "PE", atm - offset),
        ],
        "Short Strangle": [
            ("SELL", "CE", atm + offset),
            ("SELL", "PE", atm - offset),
        ],
        "Bull Call Spread": [
            ("BUY", "CE", atm),
            ("SELL", "CE", atm + offset),
        ],
        "Bear Put Spread": [
            ("BUY", "PE", atm),
            ("SELL", "PE", atm - offset),
        ],
        "Iron Condor": [
            ("SELL", "CE", atm + offset),
            ("BUY", "CE", atm + offset + _active_profile()["strike_step"]),
            ("SELL", "PE", atm - offset),
            ("BUY", "PE", atm - offset - _active_profile()["strike_step"]),
        ],
    }

    for name, legs in strategies.items():
        # Compute net premium
        net = 0
        all_priced = True
        leg_strs = []
        for action, opt_type, strike in legs:
            price = _get_price(strike, opt_type)
            if price <= 0:
                all_priced = False
                break
            sign = 1 if action == "BUY" else -1
            net += price * sign
            leg_strs.append(f"{action[0]} {opt_type}{strike:.0f}@{price:.1f}")

        if not all_priced:
            continue

        net_total = net * lots * _active_profile()["lot_size"]
        label = f"{name}: {' + '.join(leg_strs)}"
        cost_label = f"{'Debit' if net > 0 else 'Credit'} ₹{abs(net_total):,.0f}"

        if st.button(f"{name} ({cost_label})", key=f"qs_{name}", use_container_width=True):
            import uuid
            group_id = uuid.uuid4().hex[:12]
            for action, opt_type, strike in legs:
                price = _get_price(strike, opt_type)
                place_order(
                    expiry_date=selected_expiry,
                    strike=strike,
                    option_type=opt_type,
                    action=action,
                    lots=lots,
                    entry_price=price,
                    strategy=name,
                    group_id=group_id,
                    lot_size=_active_profile()["lot_size"],
                )
            st.success(f"{name} placed: {len(legs)} legs @ net {cost_label}")
            st.rerun()


# ======================================================================
# Open positions
# ======================================================================

def _render_open_positions(open_trades: pd.DataFrame, chain_df: pd.DataFrame):
    """Show open positions with live P&L and detailed position info."""
    st.subheader(f"Open Positions ({len(open_trades)})")

    if open_trades.empty:
        st.info("No open positions. Place an order above to get started.")
        return

    from datetime import datetime

    enriched = compute_live_pnl(open_trades, chain_df)
    now = datetime.now()

    # Group by strategy group
    groups = enriched.groupby("group_id")

    for group_id, group_df in groups:
        strategy = group_df.iloc[0]["strategy"]
        total_pnl = group_df["unrealized_pnl"].sum()
        pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        num_legs = len(group_df)

        # Net premium for the strategy group
        net_entry = 0.0
        net_current = 0.0
        for _, t in group_df.iterrows():
            sign = 1 if t["action"] == "BUY" else -1
            trade_lot_size = t.get("lot_size", _active_profile()["lot_size"])
            qty = t["lots"] * trade_lot_size
            net_entry += t["entry_price"] * sign * qty
            net_current += t["live_price"] * sign * qty

        entry_time = group_df["entry_time"].min()
        if pd.notna(entry_time):
            holding = now - entry_time
            hours = holding.total_seconds() / 3600
            if hours < 1:
                hold_str = f"{int(holding.total_seconds() / 60)}m"
            elif hours < 24:
                hold_str = f"{hours:.1f}h"
            else:
                hold_str = f"{holding.days}d {int(hours % 24)}h"
            time_str = entry_time.strftime("%d-%b %H:%M")
        else:
            hold_str = "—"
            time_str = "—"

        # ROI %
        cost_basis = abs(net_entry) if net_entry != 0 else 1
        roi_pct = total_pnl / cost_basis * 100

        with st.container(border=True):
            # Header row: strategy, P&L, ROI, holding time
            st.markdown(
                f"**{strategy}** ({num_legs} leg{'s' if num_legs > 1 else ''}) "
                f"&nbsp;|&nbsp; P&L: <span style='color:{pnl_color}'>₹{total_pnl:+,.0f}</span> "
                f"&nbsp;|&nbsp; ROI: <span style='color:{pnl_color}'>{roi_pct:+.1f}%</span> "
                f"&nbsp;|&nbsp; Holding: {hold_str} "
                f"&nbsp;|&nbsp; Entry: {time_str}",
                unsafe_allow_html=True,
            )

            # Strategy-level metrics
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.caption(f"Net Entry: ₹{net_entry:+,.0f}")
            sc2.caption(f"Net Current: ₹{net_current:+,.0f}")
            sc3.caption(f"Expiry: {group_df.iloc[0].get('expiry_date', '—')}")
            sc4.caption(f"Qty: {int(group_df['lots'].sum())} lot(s)")

            # Per-leg detail table
            rows = []
            for _, t in group_df.iterrows():
                direction_icon = "🟢" if t["action"] == "BUY" else "🔴"
                pnl_val = t["unrealized_pnl"]

                # % change from entry
                if t["entry_price"] > 0:
                    chg_pct = (t["live_price"] - t["entry_price"]) / t["entry_price"] * 100
                else:
                    chg_pct = 0

                # Greeks from enriched chain (if available)
                m = chain_df[
                    (chain_df["strike"] == t["strike"]) &
                    (chain_df["option_type"] == t["option_type"])
                ]
                delta = m.iloc[0].get("delta", 0) if not m.empty and "delta" in chain_df.columns else 0
                iv = m.iloc[0].get("iv", 0) if not m.empty and "iv" in chain_df.columns else 0

                row = {
                    "": direction_icon,
                    "Action": t["action"],
                    "Type": t["option_type"],
                    "Strike": f"₹{t['strike']:,.0f}",
                    "Lots": t["lots"],
                    "Entry": f"₹{t['entry_price']:.2f}",
                    "LTP": f"₹{t['live_price']:.2f}",
                    "Chg%": f"{chg_pct:+.1f}%",
                    "P&L": f"₹{pnl_val:+,.0f}",
                    "Delta": f"{delta:.3f}" if delta else "—",
                    "IV%": f"{iv:.1f}" if iv else "—",
                    "SL": f"₹{t['sl_price']:.2f}" if t.get("sl_price") and t["sl_price"] > 0 else "—",
                    "Target": f"₹{t['target_price']:.2f}" if t.get("target_price") and t["target_price"] > 0 else "—",
                }
                rows.append(row)

            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Close buttons
            col1, col2 = st.columns([1, 4])
            if num_legs > 1:
                if col1.button("Close All Legs", key=f"close_grp_{group_id}", type="secondary"):
                    close_group(group_id, chain_df)
                    st.rerun()
            else:
                t = group_df.iloc[0]
                if col1.button("Close", key=f"close_{t['id']}", type="secondary"):
                    price = chain_df[
                        (chain_df["strike"] == t["strike"]) &
                        (chain_df["option_type"] == t["option_type"])
                    ]
                    exit_price = price.iloc[0]["ltp"] if not price.empty else t["live_price"]
                    close_position(t["id"], exit_price)
                    st.rerun()


# ======================================================================
# Trade history
# ======================================================================

def _render_trade_history():
    """Show closed trades and equity curve."""
    closed = get_closed_trades(limit=100)

    st.subheader(f"Trade History ({len(closed)} trades)")

    if closed.empty:
        st.info("No closed trades yet.")
        return

    # Equity curve
    sorted_trades = closed.sort_values("exit_time")
    cumulative = sorted_trades["pnl"].cumsum()

    if len(cumulative) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_trades["exit_time"],
            y=cumulative.values,
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#22c55e" if cumulative.iloc[-1] >= 0 else "#ef4444"),
            marker=dict(size=5),
            name="Cumulative P&L",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Cumulative P&L (₹)",
            template="plotly_dark",
            height=300,
            margin=dict(t=20, b=40, l=60, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, key="paper_equity")

    # Stats row
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total P&L", f"₹{closed['pnl'].sum():,.0f}")
    c2.metric("Avg Win", f"₹{wins['pnl'].mean():,.0f}" if not wins.empty else "—")
    c3.metric("Avg Loss", f"₹{losses['pnl'].mean():,.0f}" if not losses.empty else "—")
    pf = abs(wins["pnl"].sum() / losses["pnl"].sum()) if not losses.empty and losses["pnl"].sum() != 0 else 0
    c4.metric("Profit Factor", f"{pf:.2f}" if pf > 0 else "—")

    # Trade table
    display_cols = []
    rows = []
    for _, t in closed.iterrows():
        direction = 1 if t["action"] == "BUY" else -1
        rows.append({
            "Exit Time": t["exit_time"].strftime("%d-%b %H:%M") if pd.notna(t["exit_time"]) else "",
            "Strategy": t["strategy"],
            "Type": t["option_type"],
            "Strike": f"₹{t['strike']:,.0f}",
            "Action": t["action"],
            "Lots": t["lots"],
            "Entry ₹": f"{t['entry_price']:.2f}",
            "Exit ₹": f"{t['exit_price']:.2f}" if t.get("exit_price") else "—",
            "P&L": f"₹{t['pnl']:+,.0f}" if t.get("pnl") else "—",
            "Exit Reason": t.get("exit_reason", ""),
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
