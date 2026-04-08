"""NIFTY 50 Options Dashboard — real-time metrics via Fyers API."""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import config
from auth import get_valid_token, run_auth_flow, load_token, validate_token
from backtest_ui import render_backtest_tab
from data_fetcher import FyersDataFetcher
from metrics import enrich_with_greeks, calculate_pcr, calculate_max_pain

# ======================================================================
# Page config
# ======================================================================
st.set_page_config(
    page_title="NIFTY Options Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 3rem; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stDataFrame { font-size: 13px; }
    div[data-testid="stMetric"] {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 10px 14px;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Authentication check
# ======================================================================

def check_auth() -> str | None:
    """Return valid token or show auth UI."""
    token = load_token()
    if token and validate_token(token):
        return token

    st.warning("Fyers API token not found or expired. Please authenticate.")

    with st.expander("Authentication", expanded=True):
        st.markdown("""
        **Steps to authenticate:**
        1. Create an app at [myapi.fyers.in](https://myapi.fyers.in/dashboard)
        2. Set your credentials in `.env` file (copy from `.env.example`)
        3. Run `python auth.py` in terminal to complete the OAuth flow
        4. Refresh this page after authentication
        """)

        if st.button("I've completed authentication — Refresh"):
            st.rerun()

    return None


# ======================================================================
# Sidebar
# ======================================================================

def render_sidebar(fetcher: FyersDataFetcher) -> dict:
    """Render sidebar controls and return settings dict."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Refresh interval
        refresh_sec = st.slider(
            "Auto-refresh (seconds)",
            min_value=3,
            max_value=60,
            value=config.REFRESH_INTERVAL_SEC,
            step=1,
        )

        # Strike count
        strike_count = st.slider(
            "Strikes (above & below ATM)",
            min_value=5,
            max_value=30,
            value=config.DEFAULT_STRIKE_COUNT,
            step=1,
        )

        # Expiry selector
        expiry_data = []  # list of dicts: {"date": "13-04-2026", "expiry": "1776074400", ...}
        selected_expiry_ts = ""
        selected_expiry_date = ""
        try:
            expiry_data = fetcher.get_expiry_dates()
        except Exception as e:
            st.error(f"Failed to fetch expiries: {e}")

        if expiry_data:
            date_labels = [e["date"] for e in expiry_data]
            selected_date = st.selectbox(
                "Expiry",
                options=date_labels,
                index=0,
            )
            # Find the matching epoch timestamp
            for e in expiry_data:
                if e["date"] == selected_date:
                    selected_expiry_ts = e["expiry"]
                    selected_expiry_date = e["date"]
                    break

        st.divider()
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

        if st.button("🔄 Manual Refresh"):
            st.rerun()

    return {
        "refresh_sec": refresh_sec,
        "strike_count": strike_count,
        "selected_expiry_ts": selected_expiry_ts,
        "selected_expiry_date": selected_expiry_date,
        "expiry_data": expiry_data,
    }


# ======================================================================
# Compute time to expiry
# ======================================================================

def time_to_expiry(expiry_date_str: str) -> float:
    """Return time to expiry in years from a date string like '13-04-2026' (dd-mm-yyyy)."""
    try:
        expiry = datetime.strptime(expiry_date_str, "%d-%m-%Y")
        # Options expire at 15:30 IST
        expiry = expiry.replace(hour=15, minute=30)
        now = datetime.now()
        delta = (expiry - now).total_seconds()
        return max(delta / (365.25 * 24 * 3600), 1 / (365.25 * 24 * 3600))
    except Exception:
        return 1 / 365.25  # fallback: 1 day


# ======================================================================
# Spot header
# ======================================================================

def render_spot_header(spot: dict):
    """Show NIFTY 50 spot price bar at top."""
    change_color = "green" if spot["change"] >= 0 else "red"
    arrow = "▲" if spot["change"] >= 0 else "▼"

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    with col1:
        st.markdown(
            f"### NIFTY 50 &nbsp; **₹{spot['ltp']:,.2f}** "
            f"<span style='color:{change_color}'>{arrow} {spot['change']:+.2f} "
            f"({spot['change_pct']:+.2f}%)</span>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Open", f"₹{spot['open']:,.2f}")
    with col3:
        st.metric("High", f"₹{spot['high']:,.2f}")
    with col4:
        st.metric("Low", f"₹{spot['low']:,.2f}")
    with col5:
        st.metric("Prev Close", f"₹{spot['prev_close']:,.2f}")


# ======================================================================
# Option chain table
# ======================================================================

def render_option_chain_table(df: pd.DataFrame, spot: float):
    """Render the option chain as a merged CE | Strike | PE table."""
    st.subheader("Option Chain")

    # Pick only the columns we need to avoid duplicates after enrich_with_greeks
    keep_cols = [
        "strike", "option_type", "ltp", "bid", "ask", "change",
        "volume", "oi", "oi_change", "iv", "delta",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    calls = df.loc[df["option_type"] == "CE", keep_cols].copy()
    puts = df.loc[df["option_type"] == "PE", keep_cols].copy()

    # Drop option_type before pivot — no longer needed
    calls = calls.drop(columns=["option_type"]).set_index("strike")
    puts = puts.drop(columns=["option_type"]).set_index("strike")

    # Deduplicate any remaining duplicate columns
    calls = calls.loc[:, ~calls.columns.duplicated()]
    puts = puts.loc[:, ~puts.columns.duplicated()]

    calls.columns = [f"CE_{c}" for c in calls.columns]
    puts.columns = [f"PE_{c}" for c in puts.columns]

    merged = calls.join(puts, how="outer").reset_index()
    merged = merged.sort_values("strike").reset_index(drop=True)

    # Select columns for display
    display_cols = [
        "CE_oi", "CE_oi_change", "CE_volume", "CE_iv", "CE_delta",
        "CE_ltp", "CE_change", "CE_bid", "CE_ask",
        "strike",
        "PE_bid", "PE_ask", "PE_change", "PE_ltp",
        "PE_delta", "PE_iv", "PE_volume", "PE_oi_change", "PE_oi",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    display_df = merged[display_cols].copy()

    # Mark ATM strike row
    atm_idx = (display_df["strike"] - spot).abs().idxmin()
    atm_strike = display_df.loc[atm_idx, "strike"]
    st.caption(f"ATM Strike: **{atm_strike:,.0f}**")

    st.dataframe(
        display_df.round(2),
        use_container_width=True,
        height=500,
    )


# ======================================================================
# OI Analysis
# ======================================================================

def render_oi_analysis(df: pd.DataFrame, spot: float):
    """PCR, max pain, and OI distribution charts."""
    st.subheader("Open Interest Analysis")

    pcr = calculate_pcr(df)
    max_pain = calculate_max_pain(df)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pcr_color = "normal" if 0.8 <= pcr["pcr_oi"] <= 1.2 else "inverse"
        st.metric("PCR (OI)", f"{pcr['pcr_oi']:.3f}", delta_color=pcr_color)
    with col2:
        st.metric("PCR (Volume)", f"{pcr['pcr_volume']:.3f}")
    with col3:
        st.metric("Max Pain", f"₹{max_pain:,.0f}")
    with col4:
        diff = spot - max_pain
        st.metric("Spot vs Max Pain", f"{diff:+,.0f}")

    # OI distribution bar chart
    col_left, col_right = st.columns(2)

    with col_left:
        calls = df[df["option_type"] == "CE"].sort_values("strike")
        puts = df[df["option_type"] == "PE"].sort_values("strike")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=calls["strike"], y=calls["oi"],
            name="Call OI", marker_color="#ef4444", opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            x=puts["strike"], y=puts["oi"],
            name="Put OI", marker_color="#22c55e", opacity=0.8,
        ))
        fig.add_vline(x=spot, line_dash="dash", line_color="yellow",
                      annotation_text=f"Spot {spot:,.0f}")
        fig.add_vline(x=max_pain, line_dash="dot", line_color="cyan",
                      annotation_text=f"MaxPain {max_pain:,.0f}")
        fig.update_layout(
            title="OI Distribution by Strike",
            xaxis_title="Strike", yaxis_title="Open Interest",
            barmode="group", template="plotly_dark", height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=calls["strike"], y=calls["oi_change"],
            name="Call OI Change", marker_color="#ef4444", opacity=0.8,
        ))
        fig2.add_trace(go.Bar(
            x=puts["strike"], y=puts["oi_change"],
            name="Put OI Change", marker_color="#22c55e", opacity=0.8,
        ))
        fig2.add_vline(x=spot, line_dash="dash", line_color="yellow",
                       annotation_text=f"Spot")
        fig2.update_layout(
            title="OI Change by Strike",
            xaxis_title="Strike", yaxis_title="OI Change",
            barmode="group", template="plotly_dark", height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ======================================================================
# Greeks display
# ======================================================================

def render_greeks(df: pd.DataFrame, spot: float):
    """Greeks heatmap and IV smile."""
    st.subheader("Option Greeks & IV")

    col_left, col_right = st.columns(2)

    with col_left:
        # IV Smile
        calls = df[df["option_type"] == "CE"].sort_values("strike")
        puts = df[df["option_type"] == "PE"].sort_values("strike")

        fig = go.Figure()
        if "iv" in calls.columns:
            fig.add_trace(go.Scatter(
                x=calls["strike"], y=calls["iv"],
                mode="lines+markers", name="Call IV",
                line=dict(color="#ef4444"),
            ))
            fig.add_trace(go.Scatter(
                x=puts["strike"], y=puts["iv"],
                mode="lines+markers", name="Put IV",
                line=dict(color="#22c55e"),
            ))
        fig.add_vline(x=spot, line_dash="dash", line_color="yellow",
                      annotation_text="ATM")
        fig.update_layout(
            title="IV Smile / Skew",
            xaxis_title="Strike", yaxis_title="IV (%)",
            template="plotly_dark", height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Delta curve
        fig2 = go.Figure()
        if "delta" in calls.columns:
            fig2.add_trace(go.Scatter(
                x=calls["strike"], y=calls["delta"],
                mode="lines+markers", name="Call Delta",
                line=dict(color="#ef4444"),
            ))
            fig2.add_trace(go.Scatter(
                x=puts["strike"], y=puts["delta"],
                mode="lines+markers", name="Put Delta",
                line=dict(color="#22c55e"),
            ))
        fig2.add_vline(x=spot, line_dash="dash", line_color="yellow",
                       annotation_text="ATM")
        fig2.update_layout(
            title="Delta by Strike",
            xaxis_title="Strike", yaxis_title="Delta",
            template="plotly_dark", height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Greeks table
    with st.expander("Full Greeks Table", expanded=False):
        greek_cols = ["strike", "option_type", "ltp", "iv", "delta",
                      "gamma", "theta", "vega"]
        greek_cols = [c for c in greek_cols if c in df.columns]
        st.dataframe(
            df[greek_cols].sort_values(["strike", "option_type"]),
            use_container_width=True,
            height=400,
        )


# ======================================================================
# Volume & Price section
# ======================================================================

def render_volume_price(df: pd.DataFrame, spot: float):
    """Volume and bid-ask spread analysis."""
    st.subheader("Volume & Spread Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        calls = df[df["option_type"] == "CE"].sort_values("strike")
        puts = df[df["option_type"] == "PE"].sort_values("strike")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=calls["strike"], y=calls["volume"],
            name="Call Volume", marker_color="#ef4444", opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            x=puts["strike"], y=puts["volume"],
            name="Put Volume", marker_color="#22c55e", opacity=0.7,
        ))
        fig.update_layout(
            title="Volume by Strike",
            xaxis_title="Strike", yaxis_title="Volume",
            barmode="group", template="plotly_dark", height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = go.Figure()
        if "spread" in calls.columns:
            fig2.add_trace(go.Scatter(
                x=calls["strike"], y=calls["spread"],
                mode="lines+markers", name="Call Spread",
                line=dict(color="#ef4444"),
            ))
            fig2.add_trace(go.Scatter(
                x=puts["strike"], y=puts["spread"],
                mode="lines+markers", name="Put Spread",
                line=dict(color="#22c55e"),
            ))
        fig2.add_vline(x=spot, line_dash="dash", line_color="yellow",
                       annotation_text="ATM")
        fig2.update_layout(
            title="Bid-Ask Spread by Strike",
            xaxis_title="Strike", yaxis_title="Spread (₹)",
            template="plotly_dark", height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ======================================================================
# Market Intelligence — 4 Sections + Trap Detector
# ======================================================================

def _classify_oi_action(oi_change: float, price_change: float) -> str:
    """Classify OI activity into buildup/unwinding type."""
    if oi_change > 0 and price_change > 0:
        return "Long Buildup"
    elif oi_change > 0 and price_change <= 0:
        return "Short Buildup"
    elif oi_change < 0 and price_change < 0:
        return "Long Unwinding"
    elif oi_change < 0 and price_change >= 0:
        return "Short Covering"
    return "No Change"


def render_market_intelligence(df: pd.DataFrame, spot: float, pcr_data: dict):
    """Render the full market intelligence panel."""

    calls = df[df["option_type"] == "CE"].copy()
    puts = df[df["option_type"] == "PE"].copy()

    # Classify each row
    calls["oi_action"] = calls.apply(
        lambda r: _classify_oi_action(r.get("oi_change", 0), r.get("change", 0)), axis=1
    )
    puts["oi_action"] = puts.apply(
        lambda r: _classify_oi_action(r.get("oi_change", 0), r.get("change", 0)), axis=1
    )

    # Pre-compute shared values
    pcr_oi = pcr_data["pcr_oi"]
    max_pain = calculate_max_pain(df)
    total_ce_oi_change = calls["oi_change"].sum()
    total_pe_oi_change = puts["oi_change"].sum()

    atm_strike = df.loc[(df["strike"] - spot).abs().idxmin(), "strike"]
    atm_calls = calls[calls["strike"] == atm_strike]
    atm_puts = puts[puts["strike"] == atm_strike]
    atm_ce_iv = atm_calls["iv"].iloc[0] if not atm_calls.empty and "iv" in atm_calls.columns else 0
    atm_pe_iv = atm_puts["iv"].iloc[0] if not atm_puts.empty and "iv" in atm_puts.columns else 0
    atm_iv = (atm_ce_iv + atm_pe_iv) / 2 if (atm_ce_iv + atm_pe_iv) > 0 else 0

    otm_calls = calls[calls["strike"] > spot]
    otm_puts = puts[puts["strike"] < spot]
    avg_otm_call_iv = otm_calls["iv"].mean() if not otm_calls.empty else 0
    avg_otm_put_iv = otm_puts["iv"].mean() if not otm_puts.empty else 0
    iv_skew = avg_otm_put_iv - avg_otm_call_iv

    immediate_support = puts.nlargest(1, "oi")["strike"].iloc[0] if not puts.empty else 0
    immediate_resistance = calls.nlargest(1, "oi")["strike"].iloc[0] if not calls.empty else 0
    range_width = immediate_resistance - immediate_support

    # Volatility regime
    if atm_iv > 25:
        vol_status = "HIGH"
        vol_icon = "🔴"
    elif atm_iv > 15:
        vol_status = "MODERATE"
        vol_icon = "🟡"
    else:
        vol_status = "LOW"
        vol_icon = "🟢"

    # Market signal: RANGE vs TREND
    pcr_balanced = 0.7 <= pcr_oi <= 1.3
    oi_balanced = abs(total_ce_oi_change - total_pe_oi_change) < (total_ce_oi_change + total_pe_oi_change) * 0.3 if (total_ce_oi_change + total_pe_oi_change) > 0 else True
    low_vol = atm_iv < 20
    if pcr_balanced and low_vol:
        market_signal = "RANGE-BOUND"
        signal_color = "orange"
    elif not pcr_balanced or atm_iv > 22:
        market_signal = "TRENDING"
        signal_color = "#00bfff"
    else:
        market_signal = "TRANSITIONING"
        signal_color = "yellow"

    # ==================================================================
    # 🟢 SECTION 1: MARKET STATE
    # ==================================================================
    st.markdown("### 🟢 Market State")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("PCR (OI)", f"{pcr_oi:.3f}")
        if pcr_oi > 1.3:
            st.caption("Bullish — heavy put writing")
        elif pcr_oi > 1.0:
            st.caption("Mildly Bullish")
        elif pcr_oi > 0.7:
            st.caption("Neutral zone")
        elif pcr_oi > 0.5:
            st.caption("Mildly Bearish")
        else:
            st.caption("Bearish — heavy call writing")

    with col2:
        st.metric("ATM IV", f"{atm_iv:.1f}%")
        st.caption(f"{vol_icon} {vol_status}")

    with col3:
        st.metric("IV Skew", f"{iv_skew:+.1f}%")
        skew_note = "Put fear premium" if iv_skew > 2 else "Balanced" if abs(iv_skew) <= 2 else "Call demand high"
        st.caption(skew_note)

    with col4:
        st.metric("Max Pain", f"₹{max_pain:,.0f}")
        st.caption(f"Spot gap: {spot - max_pain:+,.0f}")

    with col5:
        st.markdown(
            f"<div style='text-align:center; padding:8px; border:2px solid {signal_color}; "
            f"border-radius:8px; margin-top:4px;'>"
            f"<span style='font-size:0.8em; color:gray;'>Signal</span><br>"
            f"<span style='color:{signal_color}; font-size:1.4em; font-weight:bold;'>"
            f"{market_signal}</span></div>",
            unsafe_allow_html=True,
        )

    # OI sentiment bar
    net_oi = total_pe_oi_change - total_ce_oi_change
    if total_pe_oi_change > total_ce_oi_change and total_pe_oi_change > 0:
        oi_flow = "Bullish — more put writing (support building)"
        oi_flow_color = "green"
    elif total_ce_oi_change > total_pe_oi_change and total_ce_oi_change > 0:
        oi_flow = "Bearish — more call writing (resistance building)"
        oi_flow_color = "red"
    else:
        oi_flow = "Neutral — balanced OI flow"
        oi_flow_color = "gray"

    st.markdown(
        f"**OI Flow:** <span style='color:{oi_flow_color}'>{oi_flow}</span> "
        f"&nbsp;|&nbsp; CE OI chg: {total_ce_oi_change:+,.0f} &nbsp;|&nbsp; PE OI chg: {total_pe_oi_change:+,.0f}",
        unsafe_allow_html=True,
    )

    st.divider()

    # ==================================================================
    # 🟡 SECTION 2: LEVELS
    # ==================================================================
    st.markdown("### 🟡 Key Levels")

    top_put_oi = puts.nlargest(3, "oi")[["strike", "oi"]].reset_index(drop=True)
    top_call_oi = calls.nlargest(3, "oi")[["strike", "oi"]].reset_index(drop=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Support (Put OI Max)", f"₹{immediate_support:,.0f}",
                  delta=f"{spot - immediate_support:+,.0f} from spot")
    with col2:
        st.metric("Resistance (Call OI Max)", f"₹{immediate_resistance:,.0f}",
                  delta=f"{immediate_resistance - spot:,.0f} away")
    with col3:
        near_support = abs(spot - immediate_support) / spot < 0.005 if spot > 0 else False
        near_resistance = abs(spot - immediate_resistance) / spot < 0.005 if spot > 0 else False
        if near_support:
            pos = "AT SUPPORT"
            pos_color = "green"
        elif near_resistance:
            pos = "AT RESISTANCE"
            pos_color = "red"
        elif spot > immediate_resistance:
            pos = "ABOVE RESISTANCE"
            pos_color = "#ff6600"
        elif spot < immediate_support:
            pos = "BELOW SUPPORT"
            pos_color = "#ff6600"
        else:
            pos = "BETWEEN LEVELS"
            pos_color = "gray"
        st.markdown(
            f"<div style='text-align:center; padding:8px; margin-top:4px;'>"
            f"<span style='font-size:0.8em; color:gray;'>Price Position</span><br>"
            f"<span style='color:{pos_color}; font-weight:bold;'>{pos}</span></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.metric("Range Width", f"{range_width:,.0f} pts")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Support Levels (Put OI)**")
        support_df = top_put_oi.copy()
        support_df.columns = ["Strike", "Put OI"]
        support_df["Distance"] = spot - support_df["Strike"]
        st.dataframe(support_df.round(0), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Resistance Levels (Call OI)**")
        resist_df = top_call_oi.copy()
        resist_df.columns = ["Strike", "Call OI"]
        resist_df["Distance"] = resist_df["Strike"] - spot
        st.dataframe(resist_df.round(0), use_container_width=True, hide_index=True)

    st.divider()

    # ==================================================================
    # 🔵 SECTION 3: ACTIVITY
    # ==================================================================
    st.markdown("### 🔵 OI Activity")

    tab_buildup, tab_unwinding = st.tabs(["Fresh OI Buildup", "Unwinding Alerts"])

    with tab_buildup:
        ce_buildup = calls[calls["oi_change"] > 0].nlargest(5, "oi_change")
        pe_buildup = puts[puts["oi_change"] > 0].nlargest(5, "oi_change")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Call OI Buildup** — Resistance forming")
            if not ce_buildup.empty:
                d = ce_buildup[["strike", "oi_change", "oi", "ltp", "change", "oi_action"]].copy()
                d.columns = ["Strike", "OI Chg", "Total OI", "LTP", "Price Chg", "Action"]
                st.dataframe(d.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("No fresh call OI buildup")
        with col2:
            st.markdown("**Put OI Buildup** — Support forming")
            if not pe_buildup.empty:
                d = pe_buildup[["strike", "oi_change", "oi", "ltp", "change", "oi_action"]].copy()
                d.columns = ["Strike", "OI Chg", "Total OI", "LTP", "Price Chg", "Action"]
                st.dataframe(d.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("No fresh put OI buildup")

    with tab_unwinding:
        ce_unwinding = calls[calls["oi_change"] < 0].nsmallest(5, "oi_change")
        pe_unwinding = puts[puts["oi_change"] < 0].nsmallest(5, "oi_change")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Call OI Unwinding** — Resistance weakening")
            if not ce_unwinding.empty:
                d = ce_unwinding[["strike", "oi_change", "oi", "ltp", "change", "oi_action"]].copy()
                d.columns = ["Strike", "OI Chg", "Total OI", "LTP", "Price Chg", "Action"]
                st.dataframe(d.round(2), use_container_width=True, hide_index=True)
            else:
                st.success("No call unwinding — resistance holding firm")
        with col2:
            st.markdown("**Put OI Unwinding** — Support weakening")
            if not pe_unwinding.empty:
                d = pe_unwinding[["strike", "oi_change", "oi", "ltp", "change", "oi_action"]].copy()
                d.columns = ["Strike", "OI Chg", "Total OI", "LTP", "Price Chg", "Action"]
                st.dataframe(d.round(2), use_container_width=True, hide_index=True)
            else:
                st.success("No put unwinding — support holding firm")

    st.divider()

    # ==================================================================
    # 🔴 SECTION 4: TRADE SUGGESTION (with best strikes)
    # ==================================================================
    st.markdown("### 🔴 Trade Suggestion")

    # ----- Helper: pick best strike on a side using OI + volume + spread -----
    def _score_strike(row):
        """Higher score = better strike to trade at. Combines OI wall strength,
        liquidity (volume), and tightness (inverse spread)."""
        oi_score = row.get("oi", 0)
        vol_score = row.get("volume", 0)
        spread = row.get("spread", 0)
        spread_score = 1 / (spread + 0.05)  # tighter spread = better
        # Normalize roughly by weighting
        return oi_score * 0.5 + vol_score * 0.3 + spread_score * 0.2

    calls_scored = calls.copy()
    puts_scored = puts.copy()
    calls_scored["_score"] = calls_scored.apply(_score_strike, axis=1)
    puts_scored["_score"] = puts_scored.apply(_score_strike, axis=1)

    # OTM options only (for selling legs)
    otm_ce = calls_scored[calls_scored["strike"] > spot].sort_values("strike")
    otm_pe = puts_scored[puts_scored["strike"] < spot].sort_values("strike", ascending=False)

    # Best OTM strikes by score
    best_ce_sell = otm_ce.nlargest(1, "_score").iloc[0] if not otm_ce.empty else None
    best_pe_sell = otm_pe.nlargest(1, "_score").iloc[0] if not otm_pe.empty else None

    # ATM strikes (nearest to spot)
    atm_ce = calls_scored.iloc[(calls_scored["strike"] - spot).abs().argsort().iloc[0]] if not calls_scored.empty else None
    atm_pe = puts_scored.iloc[(puts_scored["strike"] - spot).abs().argsort().iloc[0]] if not puts_scored.empty else None

    # Wing strikes (1 strike beyond sell strikes for defined risk)
    all_strikes = sorted(df["strike"].unique())
    def _next_strike(strike, direction=1):
        """Get the next strike above (+1) or below (-1)."""
        idx = all_strikes.index(strike) if strike in all_strikes else -1
        new_idx = idx + direction
        if 0 <= new_idx < len(all_strikes):
            return all_strikes[new_idx]
        return strike + (50 * direction)  # fallback NIFTY step

    # ----- Determine strategy + pick strikes -----
    ce_heavy_unwinding = (calls["oi_change"] < 0).sum() > len(calls) * 0.5
    pe_heavy_unwinding = (puts["oi_change"] < 0).sum() > len(puts) * 0.5
    both_sides_unwinding = ce_heavy_unwinding and pe_heavy_unwinding

    strategies = []  # list of dicts with name, icon, color, reason, legs[]

    # --- Avoid trade (highest priority) ---
    if vol_status == "HIGH" and not pcr_balanced and both_sides_unwinding:
        strategies.append({
            "name": "Avoid Naked Trades",
            "icon": "🚫", "color": "#ff4444",
            "reason": "High IV + skewed PCR + mass unwinding. Whipsaw risk very high.",
            "legs": [],
        })

    # --- Breakout incoming ---
    elif both_sides_unwinding:
        straddle_strike = atm_strike
        ce_leg = calls_scored[calls_scored["strike"] == straddle_strike]
        pe_leg = puts_scored[puts_scored["strike"] == straddle_strike]
        ce_premium = ce_leg["ltp"].iloc[0] if not ce_leg.empty else 0
        pe_premium = pe_leg["ltp"].iloc[0] if not pe_leg.empty else 0
        ce_iv_val = ce_leg["iv"].iloc[0] if not ce_leg.empty and "iv" in ce_leg.columns else 0
        pe_iv_val = pe_leg["iv"].iloc[0] if not pe_leg.empty and "iv" in pe_leg.columns else 0
        total_premium = ce_premium + pe_premium
        strategies.append({
            "name": "Breakout Incoming — Long Straddle",
            "icon": "💥", "color": "#ff8c00",
            "reason": "OI unwinding both sides. Big move expected.",
            "legs": [
                {"action": "BUY", "type": "CE", "strike": straddle_strike,
                 "premium": ce_premium, "iv": ce_iv_val, "why": "ATM Call"},
                {"action": "BUY", "type": "PE", "strike": straddle_strike,
                 "premium": pe_premium, "iv": pe_iv_val, "why": "ATM Put"},
            ],
            "cost": total_premium,
            "breakeven_up": straddle_strike + total_premium,
            "breakeven_dn": straddle_strike - total_premium,
        })

    # --- Range-bound strategies ---
    elif market_signal == "RANGE-BOUND":
        if best_ce_sell is not None and best_pe_sell is not None:
            sell_ce_strike = best_ce_sell["strike"]
            sell_pe_strike = best_pe_sell["strike"]
            buy_ce_strike = _next_strike(sell_ce_strike, +1)
            buy_pe_strike = _next_strike(sell_pe_strike, -1)

            sell_ce_prem = best_ce_sell["ltp"]
            sell_pe_prem = best_pe_sell["ltp"]
            sell_ce_iv_val = best_ce_sell.get("iv", 0)
            sell_pe_iv_val = best_pe_sell.get("iv", 0)
            sell_ce_oi_val = best_ce_sell.get("oi", 0)
            sell_pe_oi_val = best_pe_sell.get("oi", 0)
            sell_ce_vol = best_ce_sell.get("volume", 0)
            sell_pe_vol = best_pe_sell.get("volume", 0)

            # Buy wing premiums
            buy_ce_row = calls_scored[calls_scored["strike"] == buy_ce_strike]
            buy_pe_row = puts_scored[puts_scored["strike"] == buy_pe_strike]
            buy_ce_prem = buy_ce_row["ltp"].iloc[0] if not buy_ce_row.empty else 0
            buy_pe_prem = buy_pe_row["ltp"].iloc[0] if not buy_pe_row.empty else 0

            net_credit = (sell_ce_prem + sell_pe_prem) - (buy_ce_prem + buy_pe_prem)

            strat_name = "Iron Condor" if vol_status == "LOW" else "Iron Butterfly / Credit Spreads"
            strat_reason = (
                "Low IV + range-bound. Premium selling favorable."
                if vol_status == "LOW"
                else "Range-bound + elevated IV. Sell premium with defined risk."
            )
            strategies.append({
                "name": strat_name,
                "icon": "🛡️", "color": "#22c55e",
                "reason": strat_reason,
                "legs": [
                    {"action": "SELL", "type": "CE", "strike": sell_ce_strike,
                     "premium": sell_ce_prem, "iv": sell_ce_iv_val,
                     "oi": sell_ce_oi_val, "volume": sell_ce_vol,
                     "why": f"Highest OI+liquidity resistance ({sell_ce_oi_val:,.0f} OI)"},
                    {"action": "BUY", "type": "CE", "strike": buy_ce_strike,
                     "premium": buy_ce_prem, "iv": 0, "why": "Wing protection"},
                    {"action": "SELL", "type": "PE", "strike": sell_pe_strike,
                     "premium": sell_pe_prem, "iv": sell_pe_iv_val,
                     "oi": sell_pe_oi_val, "volume": sell_pe_vol,
                     "why": f"Highest OI+liquidity support ({sell_pe_oi_val:,.0f} OI)"},
                    {"action": "BUY", "type": "PE", "strike": buy_pe_strike,
                     "premium": buy_pe_prem, "iv": 0, "why": "Wing protection"},
                ],
                "net_credit": net_credit,
                "max_profit": net_credit,
            })

    # --- Directional: Bullish ---
    elif market_signal == "TRENDING" and pcr_oi > 1.2 and total_pe_oi_change > total_ce_oi_change:
        if atm_ce is not None and best_ce_sell is not None:
            buy_strike = atm_ce["strike"]
            sell_strike = best_ce_sell["strike"]
            buy_prem = atm_ce["ltp"]
            sell_prem = best_ce_sell["ltp"]
            buy_iv_val = atm_ce.get("iv", 0)
            buy_delta = atm_ce.get("delta", 0)
            net_debit = buy_prem - sell_prem
            max_profit = (sell_strike - buy_strike) - net_debit
            strategies.append({
                "name": "Bull Call Spread",
                "icon": "📈", "color": "#22c55e",
                "reason": "Trending bullish + high PCR + put writing dominance.",
                "legs": [
                    {"action": "BUY", "type": "CE", "strike": buy_strike,
                     "premium": buy_prem, "iv": buy_iv_val, "delta": buy_delta,
                     "why": f"ATM — high delta ({buy_delta:.2f})"},
                    {"action": "SELL", "type": "CE", "strike": sell_strike,
                     "premium": sell_prem, "iv": best_ce_sell.get("iv", 0),
                     "oi": best_ce_sell.get("oi", 0),
                     "why": f"Best OI wall ({best_ce_sell.get('oi', 0):,.0f} OI) — likely cap"},
                ],
                "net_debit": net_debit,
                "max_profit": max_profit,
            })

    # --- Directional: Bearish ---
    elif market_signal == "TRENDING" and pcr_oi < 0.8 and total_ce_oi_change > total_pe_oi_change:
        if atm_pe is not None and best_pe_sell is not None:
            buy_strike = atm_pe["strike"]
            sell_strike = best_pe_sell["strike"]
            buy_prem = atm_pe["ltp"]
            sell_prem = best_pe_sell["ltp"]
            buy_iv_val = atm_pe.get("iv", 0)
            buy_delta = atm_pe.get("delta", 0)
            net_debit = buy_prem - sell_prem
            max_profit = (buy_strike - sell_strike) - net_debit
            strategies.append({
                "name": "Bear Put Spread",
                "icon": "📉", "color": "#ef4444",
                "reason": "Trending bearish + low PCR + call writing dominance.",
                "legs": [
                    {"action": "BUY", "type": "PE", "strike": buy_strike,
                     "premium": buy_prem, "iv": buy_iv_val, "delta": buy_delta,
                     "why": f"ATM — high delta ({buy_delta:.2f})"},
                    {"action": "SELL", "type": "PE", "strike": sell_strike,
                     "premium": sell_prem, "iv": best_pe_sell.get("iv", 0),
                     "oi": best_pe_sell.get("oi", 0),
                     "why": f"Best OI wall ({best_pe_sell.get('oi', 0):,.0f} OI) — likely floor"},
                ],
                "net_debit": net_debit,
                "max_profit": max_profit,
            })

    # --- Fallback ---
    if not strategies:
        if vol_status == "HIGH":
            strategies.append({
                "name": "Hedged Strategies Only",
                "icon": "🛡️", "color": "orange",
                "reason": "High IV. Avoid naked positions. Use spreads if taking a view.",
                "legs": [],
            })
        else:
            strategies.append({
                "name": "Monitor — No Strong Edge",
                "icon": "⏳", "color": "gray",
                "reason": "No clear setup. Wait for OI buildup or IV expansion.",
                "legs": [],
            })

    # ----- Render each strategy -----
    for strat in strategies:
        # Strategy header
        st.markdown(
            f"<div style='padding:14px; border-left:5px solid {strat['color']}; "
            f"background-color:rgba(255,255,255,0.03); border-radius:4px; margin-bottom:4px;'>"
            f"<span style='font-size:1.3em;'>{strat['icon']}</span> "
            f"<span style='color:{strat['color']}; font-weight:bold; font-size:1.2em;'>"
            f"{strat['name']}</span><br>"
            f"<span style='color:gray; font-size:0.9em;'>{strat['reason']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Legs table
        if strat["legs"]:
            legs_rows = []
            for leg in strat["legs"]:
                row = {
                    "Action": leg["action"],
                    "Type": leg["type"],
                    "Strike": leg["strike"],
                    "Premium": leg.get("premium", 0),
                    "IV %": leg.get("iv", ""),
                    "Why This Strike": leg.get("why", ""),
                }
                if "delta" in leg:
                    row["Delta"] = leg["delta"]
                if "oi" in leg:
                    row["OI"] = leg["oi"]
                if "volume" in leg:
                    row["Volume"] = leg["volume"]
                legs_rows.append(row)

            legs_df = pd.DataFrame(legs_rows)
            st.dataframe(legs_df.round(2), use_container_width=True, hide_index=True)

            # P&L summary
            pnl_parts = []
            if "net_credit" in strat:
                pnl_parts.append(f"Net Credit: **₹{strat['net_credit']:,.2f}**")
            if "net_debit" in strat:
                pnl_parts.append(f"Net Debit: **₹{strat['net_debit']:,.2f}**")
            if "max_profit" in strat:
                pnl_parts.append(f"Max Profit: **₹{strat['max_profit']:,.2f}**")
            if "cost" in strat:
                pnl_parts.append(f"Total Cost: **₹{strat['cost']:,.2f}**")
            if "breakeven_up" in strat:
                pnl_parts.append(f"Breakeven ↑: **₹{strat['breakeven_up']:,.0f}**")
            if "breakeven_dn" in strat:
                pnl_parts.append(f"Breakeven ↓: **₹{strat['breakeven_dn']:,.0f}**")

            if pnl_parts:
                st.markdown(" &nbsp;|&nbsp; ".join(pnl_parts))

    # Signal breakdown
    with st.expander("Signal breakdown"):
        st.markdown(f"""
| Parameter | Value | Reading |
|-----------|-------|---------|
| Market Signal | **{market_signal}** | PCR balanced: {'Yes' if pcr_balanced else 'No'}, Low vol: {'Yes' if low_vol else 'No'} |
| ATM IV | **{atm_iv:.1f}%** | {vol_status} |
| PCR (OI) | **{pcr_oi:.3f}** | {'Bullish' if pcr_oi > 1 else 'Bearish' if pcr_oi < 0.7 else 'Neutral'} |
| Net CE OI Change | **{total_ce_oi_change:+,.0f}** | {'Writing' if total_ce_oi_change > 0 else 'Unwinding'} |
| Net PE OI Change | **{total_pe_oi_change:+,.0f}** | {'Writing' if total_pe_oi_change > 0 else 'Unwinding'} |
| Support | **₹{immediate_support:,.0f}** | Distance: {spot - immediate_support:+,.0f} |
| Resistance | **₹{immediate_resistance:,.0f}** | Distance: {immediate_resistance - spot:+,.0f} |
        """)

    st.divider()

    # ==================================================================
    # 🧠 BONUS: TRAP DETECTOR
    # ==================================================================
    st.markdown("### 🧠 Trap Detector")

    traps = []

    # BULL TRAP: Price breaks above resistance BUT call OI is INCREASING
    # (If breakout were real, call writers would cover -> OI drops.
    #  If OI rises, writers are doubling down = they expect price to come back.)
    resistance_strikes = calls.nlargest(3, "oi")["strike"].tolist()
    for res_strike in resistance_strikes:
        if spot > res_strike:
            ce_at_strike = calls[calls["strike"] == res_strike]
            if not ce_at_strike.empty:
                oi_chg = ce_at_strike["oi_change"].iloc[0]
                if oi_chg > 0:
                    traps.append({
                        "type": "BULL TRAP",
                        "icon": "🐂🪤",
                        "color": "#ef4444",
                        "strike": res_strike,
                        "detail": (
                            f"Spot ₹{spot:,.0f} broke above resistance ₹{res_strike:,.0f} "
                            f"but Call OI INCREASED by {oi_chg:+,.0f}. "
                            f"Writers are adding positions — they expect price to fall back. "
                            f"Likely a fake breakout."
                        ),
                    })

    # BEAR TRAP: Price breaks below support BUT put OI is INCREASING
    # (If breakdown were real, put writers would cover -> OI drops.
    #  If OI rises, writers are doubling down = they expect price to bounce.)
    support_strikes = puts.nlargest(3, "oi")["strike"].tolist()
    for sup_strike in support_strikes:
        if spot < sup_strike:
            pe_at_strike = puts[puts["strike"] == sup_strike]
            if not pe_at_strike.empty:
                oi_chg = pe_at_strike["oi_change"].iloc[0]
                if oi_chg > 0:
                    traps.append({
                        "type": "BEAR TRAP",
                        "icon": "🐻🪤",
                        "color": "#22c55e",
                        "strike": sup_strike,
                        "detail": (
                            f"Spot ₹{spot:,.0f} broke below support ₹{sup_strike:,.0f} "
                            f"but Put OI INCREASED by {oi_chg:+,.0f}. "
                            f"Writers are adding positions — they expect price to bounce back. "
                            f"Likely a fake breakdown."
                        ),
                    })

    if traps:
        for trap in traps:
            st.markdown(
                f"<div style='padding:14px; border:2px solid {trap['color']}; "
                f"border-radius:8px; background-color:rgba(255,255,255,0.03); margin-bottom:10px;'>"
                f"<span style='font-size:1.5em;'>{trap['icon']}</span> "
                f"<span style='color:{trap['color']}; font-size:1.3em; font-weight:bold;'>"
                f"{trap['type']} DETECTED @ ₹{trap['strike']:,.0f}</span><br><br>"
                f"<span style='font-size:0.95em;'>{trap['detail']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='padding:14px; border:1px solid #262730; border-radius:8px; "
            "text-align:center;'>"
            "<span style='font-size:1.2em;'>✅</span> "
            "<span style='color:#22c55e; font-weight:bold;'>No Traps Detected</span><br>"
            "<span style='color:gray; font-size:0.9em;'>"
            "Price is respecting OI-based support/resistance levels. No fake breakout signals.</span>"
            "</div>",
            unsafe_allow_html=True,
        )


# ======================================================================
# Main
# ======================================================================

def render_dashboard(fetcher: FyersDataFetcher):
    """Render the live dashboard tab content."""
    # Auto-refresh (only active on this tab)
    refresh_ms = config.REFRESH_INTERVAL_SEC * 1000
    st_autorefresh(interval=refresh_ms, key="data_refresh")

    settings = render_sidebar(fetcher)

    if settings["refresh_sec"] != config.REFRESH_INTERVAL_SEC:
        st_autorefresh(
            interval=settings["refresh_sec"] * 1000,
            key="data_refresh_custom",
        )

    # Fetch data
    try:
        spot = fetcher.get_spot_quote()
        chain_df = fetcher.get_option_chain(
            strike_count=settings["strike_count"],
            expiry_ts=settings["selected_expiry_ts"],
        )
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.info("Check your API credentials and network connection.")
        return

    if chain_df.empty:
        st.warning("No option chain data returned. Market may be closed.")
        return

    # Compute time to expiry
    T = time_to_expiry(settings["selected_expiry_date"]) if settings["selected_expiry_date"] else 1 / 365.25

    # Enrich with Greeks
    chain_df = enrich_with_greeks(chain_df, spot["ltp"], config.RISK_FREE_RATE, T)

    # Render sections
    render_spot_header(spot)
    st.divider()

    render_option_chain_table(chain_df, spot["ltp"])
    st.divider()

    render_oi_analysis(chain_df, spot["ltp"])
    st.divider()

    render_greeks(chain_df, spot["ltp"])
    st.divider()

    render_volume_price(chain_df, spot["ltp"])
    st.divider()

    pcr_data = calculate_pcr(chain_df)
    render_market_intelligence(chain_df, spot["ltp"], pcr_data)


def main():
    # Create tabs FIRST so they're always visible
    tab_dashboard, tab_backtest = st.tabs(["📊 Live Dashboard", "📈 Backtest"])

    # Auth (shared across both tabs)
    token = check_auth()
    if not token:
        return

    fetcher = FyersDataFetcher(token)

    with tab_dashboard:
        render_dashboard(fetcher)

    with tab_backtest:
        render_backtest_tab(fetcher)


if __name__ == "__main__":
    main()
