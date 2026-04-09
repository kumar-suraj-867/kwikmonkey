"""Options Trading Dashboard — real-time metrics via Fyers API."""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import time as dt_time

import config
from auth import get_valid_token, run_auth_flow, load_token, validate_token, save_token, generate_auth_url, generate_token
from backtest_ui import render_backtest_tab
from paper_trading_ui import render_paper_trading_tab
from data_fetcher import FyersDataFetcher
from metrics import enrich_with_greeks, calculate_pcr, calculate_max_pain
from history_fetcher import HistoryFetcher
import data_store
from price_action import (
    detect_trend, detect_structure, compute_iv_context,
    generate_entry_signals,
)

# ======================================================================
# Page config
# ======================================================================
st.set_page_config(
    page_title="Options Trading Dashboard",
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


def _active_profile() -> dict:
    """Get active index profile from session state, default to NIFTY 50."""
    return st.session_state.get(
        "_index_profile", config.INDEX_PROFILES["NIFTY 50"]
    )


# ======================================================================
# Centralized data fetching with TTL cache
# ======================================================================

def _fetch_shared_data(fetcher: FyersDataFetcher, strike_count: int,
                       expiry_ts: str) -> dict | None:
    """Fetch spot + option chain once per refresh cycle and cache in session state.

    Returns dict with keys: spot, chain_df, expiry_list, fetch_time.
    Returns None on failure.
    """
    cache = st.session_state.get("_shared_data")
    now = datetime.now()

    # Reuse if fetched within this refresh cycle (< 2s old, same params + index)
    index_name = _active_profile().get("name", "")
    if cache and (now - cache["fetch_time"]).total_seconds() < 2:
        if (cache.get("expiry_ts") == expiry_ts
                and cache.get("strike_count") == strike_count
                and cache.get("index_name") == index_name):
            return cache

    try:
        spot = fetcher.get_spot_quote()
        chain_df, expiry_list = fetcher.get_option_chain_with_expiries(
            strike_count=strike_count, expiry_ts=expiry_ts,
        )
    except Exception:
        return None

    data = {
        "spot": spot,
        "chain_df": chain_df,
        "expiry_list": expiry_list,
        "fetch_time": now,
        "expiry_ts": expiry_ts,
        "strike_count": strike_count,
        "index_name": index_name,
    }
    st.session_state["_shared_data"] = data
    return data


# ======================================================================
# Authentication check
# ======================================================================

def check_auth() -> str | None:
    """Return valid token or show auth UI with in-app OAuth."""
    token = load_token()
    if token:
        valid, err = validate_token(token)
        if valid:
            return token
        st.warning(f"Token expired or invalid: {err}")

    # ---- In-app OAuth flow ----
    if not config.FYERS_APP_ID or not config.FYERS_SECRET_KEY:
        st.error("FYERS_APP_ID and FYERS_SECRET_KEY must be set in Streamlit secrets or .env")
        return None

    st.info("Please log in to Fyers to start the dashboard.")

    # Step 1: generate login link
    try:
        auth_url = generate_auth_url()
    except Exception as e:
        st.error(f"Failed to generate auth URL: {e}")
        return None

    st.markdown(f"### Step 1: [Click here to log in to Fyers]({auth_url})")
    st.markdown(
        "After logging in, you'll be redirected to a URL. "
        "Copy the **entire redirect URL** (or just the `auth_code` parameter) and paste it below."
    )

    # Step 2: user pastes redirect URL or auth_code
    user_input = st.text_input(
        "Step 2: Paste redirect URL or auth_code here",
        key="auth_code_input",
    )

    if st.button("Connect", key="auth_connect_btn") and user_input.strip():
        raw = user_input.strip()
        # Extract auth_code from URL if full URL was pasted
        if "auth_code=" in raw:
            try:
                from urllib.parse import urlparse, parse_qs
                parsed = parse_qs(urlparse(raw).query)
                raw = parsed.get("auth_code", [raw])[0]
            except Exception:
                pass

        with st.spinner("Exchanging auth code for token..."):
            try:
                access_token = generate_token(raw)
                st.session_state["_fyers_token"] = access_token
                st.success("Authenticated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")

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

def render_spot_header(spot: dict, index_name: str = "NIFTY 50"):
    """Show spot price bar at top."""
    change_color = "green" if spot["change"] >= 0 else "red"
    arrow = "▲" if spot["change"] >= 0 else "▼"

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    with col1:
        st.markdown(
            f"### {index_name} &nbsp; **₹{spot['ltp']:,.2f}** "
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
        width="stretch",
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
            xaxis_range=[spot - 500, spot + 500],
            barmode="group", template="plotly_dark", height=400,
            margin=dict(t=40, b=40), uirevision="persistent",
        )
        st.plotly_chart(fig, key="oi_dist", width="stretch")

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
            xaxis_range=[spot - 500, spot + 500],
            barmode="group", template="plotly_dark", height=400,
            margin=dict(t=40, b=40), uirevision="persistent",
        )
        st.plotly_chart(fig2, key="oi_chg", width="stretch")


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
            xaxis_range=[spot - 500, spot + 500],
            template="plotly_dark", height=400,
            margin=dict(t=40, b=40), uirevision="persistent",
        )
        st.plotly_chart(fig, key="iv_smile", width="stretch")

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
            xaxis_range=[spot - 500, spot + 500],
            template="plotly_dark", height=400,
            margin=dict(t=40, b=40), uirevision="persistent",
        )
        st.plotly_chart(fig2, key="delta_curve", width="stretch")

    # Greeks table
    with st.expander("Full Greeks Table", expanded=False):
        greek_cols = ["strike", "option_type", "ltp", "iv", "delta",
                      "gamma", "theta", "vega"]
        greek_cols = [c for c in greek_cols if c in df.columns]
        st.dataframe(
            df[greek_cols].sort_values(["strike", "option_type"]),
            width="stretch",
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
            xaxis_range=[spot - 500, spot + 500],
            barmode="group", template="plotly_dark", height=400,
            margin=dict(t=40, b=40), uirevision="persistent",
        )
        st.plotly_chart(fig, key="vol_dist", width="stretch")

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
            xaxis_range=[spot - 500, spot + 500],
            template="plotly_dark", height=400,
            margin=dict(t=40, b=40), uirevision="persistent",
        )
        st.plotly_chart(fig2, key="spread_curve", width="stretch")


# ======================================================================
# Market Intelligence — 4 Sections + Trap Detector
# ======================================================================

# ======================================================================
# Price Action & Entry Trigger helpers
# ======================================================================

def _get_cached_spot_candles(fetcher: FyersDataFetcher) -> pd.DataFrame:
    """Fetch 15-min spot candles (past 5 days), cached 5 minutes in session state."""
    cache_key = "_pa_spot_candles"
    cache_ts_key = "_pa_spot_candles_ts"
    cache_idx_key = "_pa_spot_candles_idx"

    profile = _active_profile()
    now = datetime.now()
    cached_ts = st.session_state.get(cache_ts_key)
    cached_idx = st.session_state.get(cache_idx_key)
    if (cached_ts and (now - cached_ts).total_seconds() < 300
            and cached_idx == profile["name"]):
        cached = st.session_state.get(cache_key)
        if cached is not None and not cached.empty:
            return cached

    try:
        history = HistoryFetcher(fetcher.fyers, underlying=fetcher.underlying)
        from_date = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")
        candles = history.get_spot_candles(from_date, to_date, resolution="15")
        if not candles.empty:
            st.session_state[cache_key] = candles
            st.session_state[cache_ts_key] = now
            st.session_state[cache_idx_key] = profile["name"]
        return candles
    except Exception:
        return pd.DataFrame()


def _get_historical_atm_iv(spot: float, days: int = 30) -> list[float]:
    """Get historical ATM IV readings from local DB snapshots."""
    try:
        data_store.init_db()
        from_dt = datetime.now() - timedelta(days=days)
        to_dt = datetime.now()
        strike_step = _active_profile()["strike_step"]
        atm_strike = round(spot / strike_step) * strike_step

        ce_hist = data_store.get_option_history(atm_strike, "CE", from_dt, to_dt)
        pe_hist = data_store.get_option_history(atm_strike, "PE", from_dt, to_dt)

        if not ce_hist.empty and not pe_hist.empty:
            merged = pd.merge(
                ce_hist[["ts", "iv"]], pe_hist[["ts", "iv"]],
                on="ts", suffixes=("_ce", "_pe"),
            )
            if not merged.empty:
                readings = ((merged["iv_ce"] + merged["iv_pe"]) / 2).tolist()
                return [v for v in readings if v > 0]

        for hist in (ce_hist, pe_hist):
            if not hist.empty:
                vals = hist["iv"].dropna().tolist()
                return [v for v in vals if v > 0]

        return []
    except Exception:
        return []


def _render_price_action_section(trend_data: dict, structure_data: dict, spot: float):
    """Render the price action analysis section."""
    st.markdown("### 📊 Price Action")

    if trend_data.get("trend") == "UNKNOWN":
        st.info("Price action unavailable — no recent spot candles. Data loads on first refresh.")
        return

    trend = trend_data["trend"]
    strength = trend_data.get("strength", 0)

    if "UPTREND" in trend:
        t_color, t_icon = "#22c55e", "▲"
    elif "DOWNTREND" in trend:
        t_color, t_icon = "#ef4444", "▼"
    elif trend == "SIDEWAYS":
        t_color, t_icon = "#f59e0b", "◆"
    else:
        t_color, t_icon = "#6b7280", "◈"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div style='text-align:center; padding:10px; border:2px solid {t_color}; "
            f"border-radius:8px;'>"
            f"<span style='font-size:0.8em; color:gray;'>Trend</span><br>"
            f"<span style='color:{t_color}; font-size:1.3em; font-weight:bold;'>"
            f"{t_icon} {trend}</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Trend Strength", f"{strength}/100")
    with col3:
        recent = trend_data.get("recent_change_pct", 0)
        st.metric("Recent Move", f"{recent:+.2f}%")
    with col4:
        ema_f = trend_data.get("ema_fast", 0)
        ema_m = trend_data.get("ema_medium", 0)
        ema_s = trend_data.get("ema_slow", 0)
        st.markdown(
            f"<div style='text-align:center; padding:6px;'>"
            f"<span style='font-size:0.8em; color:gray;'>EMA Stack</span><br>"
            f"<span style='font-size:0.9em;'>"
            f"<b>{ema_f:,.0f}</b> / {ema_m:,.0f} / {ema_s:,.0f}</span><br>"
            f"<span style='font-size:0.75em; color:gray;'>Fast / Med / Slow</span></div>",
            unsafe_allow_html=True,
        )

    # Structure
    struct = structure_data.get("structure", "UNCLEAR")
    struct_detail = structure_data.get("structure_detail", "")

    if "BULLISH" in struct:
        s_color = "#22c55e"
    elif "BEARISH" in struct:
        s_color = "#ef4444"
    elif struct in ("CONTRACTING", "EXPANDING"):
        s_color = "#f59e0b"
    else:
        s_color = "#6b7280"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<div style='padding:8px; border-left:4px solid {s_color}; "
            f"background:rgba(255,255,255,0.03); border-radius:4px;'>"
            f"<span style='font-size:0.8em; color:gray;'>Structure</span><br>"
            f"<span style='color:{s_color}; font-weight:bold;'>{struct}</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        if struct_detail:
            st.caption(struct_detail)
        p_supports = structure_data.get("price_supports", [])
        p_resists = structure_data.get("price_resistances", [])
        if p_supports or p_resists:
            sr_parts = []
            if p_supports:
                sr_parts.append("Support: " + ", ".join(f"₹{s:,.0f}" for s in p_supports[:3]))
            if p_resists:
                sr_parts.append("Resistance: " + ", ".join(f"₹{r:,.0f}" for r in p_resists[-3:]))
            st.caption(" | ".join(sr_parts))


def _render_iv_context_section(iv_ctx: dict, atm_iv: float):
    """Render the volatility context section."""
    st.markdown("### 🌡️ Volatility Context")

    iv_rank = iv_ctx.get("iv_rank")
    iv_pctile = iv_ctx.get("iv_percentile")
    regime = iv_ctx.get("regime", "UNKNOWN")
    action = iv_ctx.get("action", "")

    if iv_rank is None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current ATM IV", f"{atm_iv:.1f}%")
        with col2:
            if atm_iv > 25:
                st.caption("🔴 HIGH — options expensive, favor selling")
            elif atm_iv > 15:
                st.caption("🟡 MODERATE — options fairly priced")
            else:
                st.caption("🟢 LOW — options cheap, favor buying")
        st.caption("IV Rank/Percentile requires collected data — start collecting in Backtest tab.")
        return

    regime_colors = {
        "LOW": ("#22c55e", "🟢"), "BELOW AVG": ("#86efac", "🟢"),
        "ABOVE AVG": ("#fbbf24", "🟡"), "HIGH": ("#ef4444", "🔴"),
    }
    r_color, r_icon = regime_colors.get(regime, ("#6b7280", "⚪"))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("IV Rank", f"{iv_rank:.1f}")
    with col2:
        st.metric("IV Percentile", f"{iv_pctile:.1f}")
    with col3:
        st.markdown(
            f"<div style='text-align:center; padding:8px; border:2px solid {r_color}; "
            f"border-radius:8px;'>"
            f"<span style='font-size:0.8em; color:gray;'>IV Regime</span><br>"
            f"<span style='color:{r_color}; font-size:1.2em; font-weight:bold;'>"
            f"{r_icon} {regime}</span></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.metric("Current IV", f"{atm_iv:.1f}%")
    with col5:
        iv_min = iv_ctx.get("iv_min", 0)
        iv_max = iv_ctx.get("iv_max", 0)
        iv_mean = iv_ctx.get("iv_mean", 0)
        st.markdown(
            f"<div style='text-align:center; padding:6px;'>"
            f"<span style='font-size:0.8em; color:gray;'>30D Range</span><br>"
            f"<span style='font-size:0.9em;'>"
            f"{iv_min:.1f}% — <b>{iv_mean:.1f}%</b> — {iv_max:.1f}%</span></div>",
            unsafe_allow_html=True,
        )

    if action:
        st.markdown(
            f"<div style='padding:8px; background:rgba(255,255,255,0.03); "
            f"border-radius:4px; text-align:center;'>"
            f"💡 {action}</div>",
            unsafe_allow_html=True,
        )


def _signal_to_legs(strategy: str, spot: float, offset: int = 200) -> list[tuple]:
    """Map a signal strategy name to concrete option legs.

    Returns list of (action, option_type, strike_offset_from_ATM) tuples.
    Offsets are in points relative to ATM (positive = above, negative = below).
    """
    step = _active_profile()["strike_step"]
    atm = round(spot / step) * step

    mapping = {
        "Long Call (ATM)":               [("BUY", "CE", 0)],
        "Bull Call Spread":              [("BUY", "CE", 0), ("SELL", "CE", offset)],
        "Bull Put Spread (sell put)":    [("SELL", "PE", 0), ("BUY", "PE", -offset)],
        "Long Put (ATM)":               [("BUY", "PE", 0)],
        "Bear Put Spread":              [("BUY", "PE", 0), ("SELL", "PE", -offset)],
        "Bear Call Spread (sell call)":  [("SELL", "CE", 0), ("BUY", "CE", offset)],
        "Iron Condor (sell both sides)": [
            ("SELL", "CE", offset), ("BUY", "CE", offset + step),
            ("SELL", "PE", -offset), ("BUY", "PE", -(offset + step)),
        ],
        "Long Straddle":                [("BUY", "CE", 0), ("BUY", "PE", 0)],
    }

    raw = mapping.get(strategy)
    if not raw:
        return []

    return [(action, opt, atm + off) for action, opt, off in raw]


def _render_entry_triggers_section(signals: list[dict],
                                   chain_df: pd.DataFrame = None,
                                   spot: float = 0,
                                   expiry_date: str = ""):
    """Render entry trigger signal cards with optional paper-trade execution."""
    st.markdown("### 🎯 Entry Triggers")

    if not signals:
        st.info("No signals generated.")
        return

    from paper_trading import place_order as pt_place_order

    for i, sig in enumerate(signals):
        status = sig.get("status", "WAIT")
        direction = sig.get("direction", "")
        strategy = sig.get("strategy", "")
        confidence = sig.get("confidence", 0)
        reasoning = sig.get("reasoning", "")
        entry_zone = sig.get("entry_zone", "")
        sl = sig.get("sl", "")
        target = sig.get("target", "")

        status_styles = {
            "ENTER": ("#22c55e", "rgba(34,197,94,0.15)", "🟢"),
            "PREPARE": ("#3b82f6", "rgba(59,130,246,0.15)", "🔵"),
            "ALERT": ("#f59e0b", "rgba(245,158,11,0.15)", "🟡"),
            "WAIT": ("#6b7280", "rgba(107,114,128,0.15)", "⚪"),
        }
        badge_color, badge_bg, badge_icon = status_styles.get(
            status, ("#6b7280", "rgba(107,114,128,0.15)", "⚪")
        )

        if confidence >= 70:
            conf_color = "#22c55e"
        elif confidence >= 50:
            conf_color = "#3b82f6"
        elif confidence >= 30:
            conf_color = "#f59e0b"
        else:
            conf_color = "#6b7280"

        # Check if this signal is executable as a paper trade
        legs = _signal_to_legs(strategy, spot) if spot > 0 else []
        can_execute = bool(legs) and chain_df is not None and not chain_df.empty and status in ("ENTER", "PREPARE")

        # Build leg preview text
        leg_preview = ""
        if can_execute:
            parts = []
            for action, opt_type, strike in legs:
                m = chain_df[(chain_df["strike"] == strike) & (chain_df["option_type"] == opt_type)]
                ltp = m.iloc[0]["ltp"] if not m.empty else 0
                parts.append(f"{action} {opt_type} {strike:.0f} @{ltp:.1f}")
            leg_preview = " | ".join(parts)

        st.markdown(
            f"<div style='padding:14px; border:2px solid {badge_color}; "
            f"border-radius:8px; background:{badge_bg}; margin-bottom:4px;'>"
            f"<span style='font-size:1.5em;'>{badge_icon}</span> "
            f"<span style='color:{badge_color}; font-size:1.3em; font-weight:bold;'>"
            f"{status}</span> "
            f"<span style='color:gray;'> — {direction}</span> "
            f"<span style='float:right; color:{conf_color}; font-weight:bold;'>"
            f"{confidence}%</span><br>"
            f"<b>Strategy:</b> {strategy}<br>"
            f"<span style='font-size:0.9em;'>"
            f"Entry: <b>{entry_zone}</b> &nbsp;|&nbsp; "
            f"SL: <b>{sl}</b> &nbsp;|&nbsp; "
            f"Target: <b>{target}</b></span><br>"
            f"<span style='color:gray; font-size:0.85em;'>{reasoning}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Paper trade execution button
        if can_execute:
            st.caption(f"Legs: {leg_preview}")
            if st.button(f"📝 Paper Trade: {strategy}", key=f"pt_signal_{i}",
                         type="primary"):
                import uuid
                group_id = uuid.uuid4().hex[:12]
                placed = 0
                for action, opt_type, strike in legs:
                    m = chain_df[
                        (chain_df["strike"] == strike) &
                        (chain_df["option_type"] == opt_type)
                    ]
                    ltp = m.iloc[0]["ltp"] if not m.empty else 0
                    if ltp > 0:
                        pt_place_order(
                            expiry_date=expiry_date,
                            strike=strike,
                            option_type=opt_type,
                            action=action,
                            lots=1,
                            entry_price=ltp,
                            strategy=strategy,
                            group_id=group_id,
                            notes=f"From signal: {direction} {confidence}%",
                            lot_size=_active_profile()["lot_size"],
                        )
                        placed += 1
                if placed:
                    st.success(f"Paper trade placed: {strategy} ({placed} legs)")
                else:
                    st.error("No live prices available for the required strikes")


def _render_smart_money_section(df: pd.DataFrame, spot: float):
    """Render smart money tracking: premium flow, unusual OI, rolling, verdict."""
    st.markdown("### 🏦 Smart Money Tracking")

    calls = df[df["option_type"] == "CE"].copy()
    puts = df[df["option_type"] == "PE"].copy()
    profile = _active_profile()
    lot_size = profile["lot_size"]

    # ---- Classify activity per strike (label + proportional split) ----
    # Activity label: standard OI+Price quadrant using change (ltp - prev_close)
    # Proportional split: uses NIFTY SPOT direction instead of per-option
    # price change, because the optionchain API doesn't provide reliable
    # intraday open prices (often 0), and theta decay biases per-option
    # change negative for both CE and PE.
    #
    # Logic: when spot goes UP →
    #   CE fresh OI = more buyer-driven (calls gain value)
    #   PE fresh OI = more writer-driven (put sellers collect theta)
    # And vice versa for spot DOWN.

    spot_data = st.session_state.get("_spot", {})
    if isinstance(spot_data, dict) and spot_data.get("open", 0) > 0:
        spot_intraday_pct = (spot_data["ltp"] - spot_data["open"]) / spot_data["open"] * 100
    elif isinstance(spot_data, dict):
        spot_intraday_pct = spot_data.get("change_pct", 0)
    else:
        spot_intraday_pct = 0

    ce_buyer_frac = _buyer_fraction(spot_intraday_pct)      # spot up → high
    pe_buyer_frac = _buyer_fraction(-spot_intraday_pct)     # spot up → low (writers)

    for subset, buyer_frac in [(calls, ce_buyer_frac), (puts, pe_buyer_frac)]:
        subset["activity"] = subset.apply(
            lambda r: _classify_oi_action(r.get("oi_change", 0), r.get("change", 0)),
            axis=1,
        )
        subset["prem_deployed"] = subset["oi_change"].abs() * subset["ltp"] * lot_size

        fresh_mask = subset["oi_change"] > 0
        subset["buyer_prem"] = 0.0
        subset["writer_prem"] = 0.0
        subset.loc[fresh_mask, "buyer_prem"] = (
            subset.loc[fresh_mask, "prem_deployed"] * buyer_frac
        )
        subset.loc[fresh_mask, "writer_prem"] = (
            subset.loc[fresh_mask, "prem_deployed"] * (1 - buyer_frac)
        )

    # ---- 1. Premium Flow (proportional) ----
    # Bullish: CE buyers + PE writers (put sellers support market)
    ce_buyer_prem = calls["buyer_prem"].sum()
    pe_writer_prem = puts["writer_prem"].sum()
    bullish_prem = ce_buyer_prem + pe_writer_prem

    # Bearish: CE writers + PE buyers (put buyers expect decline)
    ce_writer_prem = calls["writer_prem"].sum()
    pe_buyer_prem = puts["buyer_prem"].sum()
    bearish_prem = ce_writer_prem + pe_buyer_prem

    writer_prem = ce_writer_prem + pe_writer_prem
    buyer_prem = ce_buyer_prem + pe_buyer_prem
    total_fresh = writer_prem + buyer_prem

    # ---- 2. Unusual OI (z-score > 2 per type) ----
    unusual_list = []
    for subset in [calls, puts]:
        oi_abs = subset["oi_change"].abs()
        mean_oi = oi_abs.mean()
        std_oi = oi_abs.std()
        if std_oi > 0:
            subset["oi_zscore"] = (oi_abs - mean_oi) / std_oi
            unusual_list.append(subset[subset["oi_zscore"] > 2.0])
        else:
            subset["oi_zscore"] = 0.0

    unusual = pd.concat(unusual_list).copy() if unusual_list else pd.DataFrame()

    # ---- 3. Rolling detection (up to 2 strikes away) ----
    rolls = []
    avg_oi_chg = df["oi_change"].abs().mean()
    for opt_type, subset in [("CE", calls), ("PE", puts)]:
        ss = subset.sort_values("strike").reset_index(drop=True)
        for i in range(len(ss)):
            for j in range(i + 1, min(i + 3, len(ss))):
                curr, nxt = ss.iloc[i], ss.iloc[j]
                # Rolling UP: lower losing OI, higher gaining
                if curr["oi_change"] < -avg_oi_chg and nxt["oi_change"] > avg_oi_chg:
                    rolls.append({"type": opt_type, "from": curr["strike"],
                                  "to": nxt["strike"], "from_chg": curr["oi_change"],
                                  "to_chg": nxt["oi_change"], "dir": "UP"})
                # Rolling DOWN: higher losing OI, lower gaining
                elif nxt["oi_change"] < -avg_oi_chg and curr["oi_change"] > avg_oi_chg:
                    rolls.append({"type": opt_type, "from": nxt["strike"],
                                  "to": curr["strike"], "from_chg": nxt["oi_change"],
                                  "to_chg": curr["oi_change"], "dir": "DOWN"})

    # ---- 4. OI Concentration & Conviction ----
    pos_oi = df[df["oi_change"] > 0]["oi_change"]
    top3 = pos_oi.nlargest(3).sum() if len(pos_oi) >= 3 else pos_oi.sum()
    total_pos = pos_oi.sum()
    concentration = (top3 / total_pos * 100) if total_pos > 0 else 0

    # Spread tightness at OI walls (institutional proxy)
    top_oi = df.nlargest(6, "oi")
    avg_spread_walls = top_oi["spread"].mean() if not top_oi.empty else 0
    avg_spread_all = df["spread"].mean() if not df.empty else 1
    tight_walls = avg_spread_walls < avg_spread_all * 0.8

    # ---- 5. Smart Money Verdict ----
    bull_pts, bear_pts = 0, 0
    reasons = []

    # a) Premium flow direction (30 pts)
    if bullish_prem > bearish_prem * 1.3 and bullish_prem > 0:
        bull_pts += 30
        reasons.append(f"Bullish flow ₹{bullish_prem / 100000:,.1f}L > bearish ₹{bearish_prem / 100000:,.1f}L")
    elif bearish_prem > bullish_prem * 1.3 and bearish_prem > 0:
        bear_pts += 30
        reasons.append(f"Bearish flow ₹{bearish_prem / 100000:,.1f}L > bullish ₹{bullish_prem / 100000:,.1f}L")

    # b) Unusual OI skew near ATM (20 pts)
    unusual_ce_near = calls[(calls["oi_zscore"] > 2) & (calls["strike"] <= spot + 200)] if "oi_zscore" in calls.columns else pd.DataFrame()
    unusual_pe_near = puts[(puts["oi_zscore"] > 2) & (puts["strike"] >= spot - 200)] if "oi_zscore" in puts.columns else pd.DataFrame()
    if len(unusual_ce_near) > len(unusual_pe_near) + 1:
        bear_pts += 20
        reasons.append(f"Unusual CE buildup near ATM ({len(unusual_ce_near)} strikes)")
    elif len(unusual_pe_near) > len(unusual_ce_near) + 1:
        bull_pts += 20
        reasons.append(f"Unusual PE buildup near ATM ({len(unusual_pe_near)} strikes)")

    # c) Rolling signals (15 pts each, max 2)
    roll_counted = 0
    for roll in rolls:
        if roll_counted >= 2:
            break
        if roll["type"] == "CE" and roll["dir"] == "UP":
            bull_pts += 15; roll_counted += 1
            reasons.append(f"CE rolling UP ₹{roll['from']:,.0f}→₹{roll['to']:,.0f} (resistance higher)")
        elif roll["type"] == "CE" and roll["dir"] == "DOWN":
            bear_pts += 15; roll_counted += 1
            reasons.append(f"CE rolling DOWN ₹{roll['from']:,.0f}→₹{roll['to']:,.0f} (resistance lower)")
        elif roll["type"] == "PE" and roll["dir"] == "UP":
            bull_pts += 15; roll_counted += 1
            reasons.append(f"PE rolling UP ₹{roll['from']:,.0f}→₹{roll['to']:,.0f} (support higher)")
        elif roll["type"] == "PE" and roll["dir"] == "DOWN":
            bear_pts += 15; roll_counted += 1
            reasons.append(f"PE rolling DOWN ₹{roll['from']:,.0f}→₹{roll['to']:,.0f} (support lower)")

    # d) Writer dominance with directional bias (20 pts)
    if writer_prem > buyer_prem * 1.5 and writer_prem > 0:
        if pe_writer_prem > ce_writer_prem * 1.2:
            bull_pts += 20
            reasons.append("Put writers dominant — institutional support")
        elif ce_writer_prem > pe_writer_prem * 1.2:
            bear_pts += 20
            reasons.append("Call writers dominant — institutional capping")

    # e) Informational
    if concentration > 60:
        reasons.append(f"OI concentrated ({concentration:.0f}%) — high conviction")
    if tight_walls:
        reasons.append("Tight spreads at OI walls — active market-making")

    if bull_pts > bear_pts + 20:
        verdict, v_color, v_icon = "BULLISH", "#22c55e", "🟢"
    elif bear_pts > bull_pts + 20:
        verdict, v_color, v_icon = "BEARISH", "#ef4444", "🔴"
    else:
        verdict, v_color, v_icon = "NEUTRAL", "#f59e0b", "🟡"

    # === RENDER ===

    # Verdict badge
    st.markdown(
        f"<div style='text-align:center; padding:12px; border:2px solid {v_color}; "
        f"border-radius:8px; margin-bottom:12px;'>"
        f"<span style='font-size:0.8em; color:gray;'>Smart Money Bias</span><br>"
        f"<span style='color:{v_color}; font-size:1.5em; font-weight:bold;'>"
        f"{v_icon} {verdict}</span></div>",
        unsafe_allow_html=True,
    )

    # Premium flow metrics
    st.markdown("**Premium Deployed** (estimated)")
    total_flow = bullish_prem + bearish_prem
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bullish Flow", f"₹{bullish_prem / 100000:,.1f}L",
                   help="CE buyers + PE writers (supporting market)")
    with col2:
        st.metric("Bearish Flow", f"₹{bearish_prem / 100000:,.1f}L",
                   help="CE writers + PE buyers (capping market)")
    with col3:
        st.metric("Total Fresh OI", f"₹{total_fresh / 100000:,.1f}L",
                   help="Total new premium deployed today")
    with col4:
        conc_label = "HIGH" if concentration > 60 else "LOW" if concentration < 30 else "MED"
        st.metric("OI Concentration", f"{concentration:.0f}% ({conc_label})",
                   help="Top 3 strikes as % of total new OI")

    # Breakdown
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption(f"CE Buyers: ₹{ce_buyer_prem / 100000:,.1f}L")
    with col2:
        st.caption(f"CE Writers: ₹{ce_writer_prem / 100000:,.1f}L")
    with col3:
        st.caption(f"PE Buyers: ₹{pe_buyer_prem / 100000:,.1f}L")
    with col4:
        st.caption(f"PE Writers: ₹{pe_writer_prem / 100000:,.1f}L")

    # Directional flow bar
    if total_flow > 0:
        bull_pct = bullish_prem / total_flow * 100
        if bull_pct > 60:
            flow_note = "Bullish flow dominant — money supporting upside"
            flow_color = "#22c55e"
        elif bull_pct < 40:
            flow_note = "Bearish flow dominant — money positioning for downside"
            flow_color = "#ef4444"
        else:
            flow_note = "Balanced flow — no strong directional bias"
            flow_color = "#f59e0b"
        st.markdown(
            f"<div style='padding:6px; text-align:center;'>"
            f"<span style='color:{flow_color};'>Bullish {bull_pct:.0f}% | Bearish {100 - bull_pct:.0f}%</span>"
            f" — {flow_note}"
            f" <span style='color:gray; font-size:0.8em;'>({profile['name']} {spot_intraday_pct:+.2f}% intraday)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Unusual activity
    if not unusual.empty:
        with st.expander(f"Unusual OI Activity ({len(unusual)} strikes)", expanded=len(unusual) <= 5):
            disp = unusual[["strike", "option_type", "oi_change", "oi", "ltp", "volume", "activity"]].copy()
            disp.columns = ["Strike", "Type", "OI Chg", "Total OI", "LTP", "Volume", "Action"]
            disp = disp.sort_values("OI Chg", key=abs, ascending=False).head(8)
            num_cols = disp.select_dtypes(include="number").columns
            disp[num_cols] = disp[num_cols].round(2)
            st.dataframe(disp, width="stretch", hide_index=True)

    # Rolling
    if rolls:
        with st.expander(f"Position Rolling ({len(rolls)} probable)"):
            for roll in rolls[:5]:
                d_icon = "⬆️" if roll["dir"] == "UP" else "⬇️"
                if roll["type"] == "CE":
                    impl = "resistance shifted higher" if roll["dir"] == "UP" else "resistance shifted lower"
                else:
                    impl = "support shifted higher" if roll["dir"] == "UP" else "support shifted lower"
                st.markdown(
                    f"{d_icon} **{roll['type']}** ₹{roll['from']:,.0f} → ₹{roll['to']:,.0f} "
                    f"(OI: {roll['from_chg']:+,.0f} / {roll['to_chg']:+,.0f}) "
                    f"<span style='color:gray;'>— {impl}</span>",
                    unsafe_allow_html=True,
                )

    # Reasoning
    if reasons:
        with st.expander("Smart Money Reasoning"):
            for r in reasons:
                st.caption(f"• {r}")


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


def _buyer_fraction(change_pct: float) -> float:
    """Sigmoid mapping from directional signal to buyer fraction.

    With k=1.0 (tuned for spot % changes, typically ±0.1 to ±2):
      +1.0% → 0.73 buyer (73% buyer / 27% writer)
      +0.5% → 0.62
       0.0% → 0.50
      -0.5% → 0.38
      -1.0% → 0.27
    """
    import math
    k = 1.0
    # Clamp to avoid overflow in exp
    x = max(min(change_pct * k, 10), -10)
    return 1.0 / (1.0 + math.exp(-x))


def render_market_intelligence(df: pd.DataFrame, spot: float, pcr_data: dict,
                               spot_candles: pd.DataFrame = None,
                               expiry_date: str = ""):
    """Render the full market intelligence panel."""

    calls = df[df["option_type"] == "CE"].copy()
    puts = df[df["option_type"] == "PE"].copy()

    # Classify each row using standard OI+Price quadrant (change from prev close)
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

    # --- Price action analysis (from spot candles) ---
    if spot_candles is not None and not spot_candles.empty:
        trend_data = detect_trend(spot_candles)
        structure_data = detect_structure(spot_candles)
    else:
        trend_data = {"trend": "UNKNOWN", "strength": 0, "details": "no candle data"}
        structure_data = {
            "structure": "UNCLEAR", "structure_detail": "",
            "swing_highs": [], "swing_lows": [],
            "price_supports": [], "price_resistances": [],
        }

    # --- IV context (from local DB history) ---
    iv_history = _get_historical_atm_iv(spot)
    iv_ctx = compute_iv_context(atm_iv, iv_history)

    # --- Entry trigger signals ---
    entry_signals = generate_entry_signals(
        trend=trend_data,
        structure=structure_data,
        iv_ctx=iv_ctx,
        pcr_oi=pcr_oi,
        spot=spot,
        oi_support=immediate_support,
        oi_resistance=immediate_resistance,
        atm_iv=atm_iv,
        max_pain=max_pain,
    )

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
    # 📊 PRICE ACTION
    # ==================================================================
    _render_price_action_section(trend_data, structure_data, spot)

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
        st.dataframe(support_df.round(0), width="stretch", hide_index=True)
    with col2:
        st.markdown("**Resistance Levels (Call OI)**")
        resist_df = top_call_oi.copy()
        resist_df.columns = ["Strike", "Call OI"]
        resist_df["Distance"] = resist_df["Strike"] - spot
        st.dataframe(resist_df.round(0), width="stretch", hide_index=True)

    st.divider()

    # ==================================================================
    # 🌡️ VOLATILITY CONTEXT
    # ==================================================================
    _render_iv_context_section(iv_ctx, atm_iv)

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
                num_cols = d.select_dtypes(include="number").columns
                d[num_cols] = d[num_cols].round(2)
                st.dataframe(d, width="stretch", hide_index=True)
            else:
                st.info("No fresh call OI buildup")
        with col2:
            st.markdown("**Put OI Buildup** — Support forming")
            if not pe_buildup.empty:
                d = pe_buildup[["strike", "oi_change", "oi", "ltp", "change", "oi_action"]].copy()
                d.columns = ["Strike", "OI Chg", "Total OI", "LTP", "Price Chg", "Action"]
                num_cols = d.select_dtypes(include="number").columns
                d[num_cols] = d[num_cols].round(2)
                st.dataframe(d, width="stretch", hide_index=True)
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
                num_cols = d.select_dtypes(include="number").columns
                d[num_cols] = d[num_cols].round(2)
                st.dataframe(d, width="stretch", hide_index=True)
            else:
                st.success("No call unwinding — resistance holding firm")
        with col2:
            st.markdown("**Put OI Unwinding** — Support weakening")
            if not pe_unwinding.empty:
                d = pe_unwinding[["strike", "oi_change", "oi", "ltp", "change", "oi_action"]].copy()
                d.columns = ["Strike", "OI Chg", "Total OI", "LTP", "Price Chg", "Action"]
                num_cols = d.select_dtypes(include="number").columns
                d[num_cols] = d[num_cols].round(2)
                st.dataframe(d, width="stretch", hide_index=True)
            else:
                st.success("No put unwinding — support holding firm")

    st.divider()

    # ==================================================================
    # 🏦 SMART MONEY TRACKING
    # ==================================================================
    _render_smart_money_section(df, spot)

    st.divider()

    # ==================================================================
    # 🎯 ENTRY TRIGGERS
    # ==================================================================
    _render_entry_triggers_section(entry_signals, chain_df=df, spot=spot,
                                    expiry_date=expiry_date)

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
        return strike + (_active_profile()["strike_step"] * direction)

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
    from paper_trading import place_order as _pt_place

    for strat_idx, strat in enumerate(strategies):
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
            # Ensure numeric columns are numeric (avoid Arrow mixed-type error)
            for col in legs_df.columns:
                legs_df[col] = legs_df[col].astype(str)
            st.dataframe(legs_df, width="stretch", hide_index=True)

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

            # --- One-click paper trade execution ---
            all_priced = all(leg.get("premium", 0) > 0 for leg in strat["legs"])
            if all_priced:
                if st.button(
                    f"📝 Paper Trade: {strat['name']}",
                    key=f"pt_suggestion_{strat_idx}",
                    type="primary",
                ):
                    import uuid
                    group_id = uuid.uuid4().hex[:12]
                    placed = 0
                    for leg in strat["legs"]:
                        _pt_place(
                            expiry_date=expiry_date,
                            strike=leg["strike"],
                            option_type=leg["type"],
                            action=leg["action"],
                            lots=1,
                            entry_price=leg["premium"],
                            strategy=strat["name"],
                            group_id=group_id,
                            notes=strat["reason"],
                            lot_size=_active_profile()["lot_size"],
                        )
                        placed += 1
                    st.success(
                        f"Paper trade placed: {strat['name']} ({placed} legs) — "
                        f"check Paper Trading tab"
                    )

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
    settings = render_sidebar(fetcher)

    # Store refresh interval for main() to pick up on next cycle
    st.session_state["_refresh_sec"] = settings["refresh_sec"]

    # Fetch data via shared cache (avoids duplicate calls across tabs)
    data = _fetch_shared_data(fetcher, settings["strike_count"],
                              settings["selected_expiry_ts"])
    if data is None:
        st.error("Failed to fetch data. Check API credentials and network connection.")
        return

    spot = data["spot"]
    chain_df = data["chain_df"]

    if chain_df.empty:
        st.warning("No option chain data returned. Market may be closed.")
        return

    # Debug: warn if strikes are missing
    if (chain_df["strike"] == 0).all():
        st.error("All strikes are 0 — Fyers API response format may have changed. Check data_fetcher.py.")
        st.json({"sample_columns": list(chain_df.columns), "rows": len(chain_df)})
        return

    # Compute time to expiry
    T = time_to_expiry(settings["selected_expiry_date"]) if settings["selected_expiry_date"] else 1 / 365.25

    # Enrich with Greeks (now vectorized — fast)
    chain_df = enrich_with_greeks(chain_df, spot["ltp"], config.RISK_FREE_RATE, T)

    # Store enriched chain for paper trading tab to reuse
    st.session_state["_enriched_chain"] = chain_df
    st.session_state["_spot"] = spot
    st.session_state["_expiry_data"] = data["expiry_list"]
    st.session_state["_expiry_date"] = settings["selected_expiry_date"]

    # Render sections
    render_spot_header(spot, index_name=_active_profile()["name"])
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
    spot_candles = _get_cached_spot_candles(fetcher)
    render_market_intelligence(chain_df, spot["ltp"], pcr_data, spot_candles,
                               expiry_date=settings["selected_expiry_date"])


def main():
    # Index selector (top of sidebar, before everything else)
    with st.sidebar:
        index_options = list(config.INDEX_PROFILES.keys())
        index_name = st.selectbox("Index", index_options, key="index_selector")
        profile = config.INDEX_PROFILES[index_name]
        st.session_state["_index_profile"] = profile
        st.divider()

    # Create tabs FIRST so they're always visible
    tab_dashboard, tab_backtest, tab_paper = st.tabs(
        ["Live Dashboard", "Backtest", "Paper Trading"]
    )

    # Auth (shared across both tabs)
    token = check_auth()
    if not token:
        return

    fetcher = FyersDataFetcher(
        token,
        underlying=profile["underlying"],
        options_symbol=profile["options_symbol"],
    )

    # Dashboard uses st.fragment for auto-refresh without freezing the whole page
    with tab_dashboard:
        refresh_sec = st.session_state.get("_refresh_sec", config.REFRESH_INTERVAL_SEC)

        @st.fragment(run_every=timedelta(seconds=refresh_sec))
        def _live_dashboard():
            render_dashboard(fetcher)

        _live_dashboard()

    with tab_backtest:
        render_backtest_tab(fetcher)

    with tab_paper:
        render_paper_trading_tab(fetcher)


if __name__ == "__main__":
    main()
