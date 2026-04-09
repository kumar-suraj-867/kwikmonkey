"""Price action analysis: trend, structure, support/resistance, entry triggers.

Operates on OHLCV DataFrames (spot candles). No Streamlit dependency.
"""

import numpy as np
import pandas as pd


# ======================================================================
# Trend detection (EMA-based)
# ======================================================================

def compute_emas(df: pd.DataFrame, fast: int = 9, medium: int = 21,
                 slow: int = 50) -> pd.DataFrame:
    """Add EMA columns to a candle DataFrame. Expects 'close' column."""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_medium"] = df["close"].ewm(span=medium, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def detect_trend(df: pd.DataFrame) -> dict:
    """Detect current trend from EMA alignment and price position.

    Returns dict with: trend, strength, ema_fast, ema_medium, ema_slow, details
    """
    if df.empty or len(df) < 5:
        return {"trend": "UNKNOWN", "strength": 0, "details": "insufficient data"}

    df = compute_emas(df)
    latest = df.iloc[-1]
    price = latest["close"]
    ema_f = latest["ema_fast"]
    ema_m = latest["ema_medium"]
    ema_s = latest["ema_slow"]

    # EMA alignment
    bullish_stack = ema_f > ema_m > ema_s  # fast > medium > slow
    bearish_stack = ema_f < ema_m < ema_s

    # Price relative to EMAs
    above_all = price > ema_f > ema_m
    below_all = price < ema_f < ema_m

    # Trend strength: how spread apart are the EMAs
    spread = abs(ema_f - ema_s)
    spread_pct = (spread / price * 100) if price > 0 else 0

    if bullish_stack and above_all:
        trend = "STRONG UPTREND"
        strength = min(spread_pct * 20, 100)  # normalize to 0-100
    elif bullish_stack:
        trend = "UPTREND"
        strength = min(spread_pct * 15, 80)
    elif bearish_stack and below_all:
        trend = "STRONG DOWNTREND"
        strength = min(spread_pct * 20, 100)
    elif bearish_stack:
        trend = "DOWNTREND"
        strength = min(spread_pct * 15, 80)
    elif abs(ema_f - ema_m) / price * 100 < 0.05:
        trend = "SIDEWAYS"
        strength = 0
    else:
        trend = "TRANSITIONING"
        strength = 20

    # Recent momentum: last 5 candles
    recent = df.tail(5)
    recent_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0] * 100

    details = (
        f"EMA {int(ema_f):,} / {int(ema_m):,} / {int(ema_s):,} | "
        f"Recent move: {recent_change:+.2f}%"
    )

    return {
        "trend": trend,
        "strength": round(strength),
        "ema_fast": round(ema_f, 2),
        "ema_medium": round(ema_m, 2),
        "ema_slow": round(ema_s, 2),
        "recent_change_pct": round(recent_change, 2),
        "details": details,
    }


# ======================================================================
# Market structure (HH/HL or LH/LL)
# ======================================================================

def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> dict:
    """Find swing highs and swing lows.

    A swing high: high[i] > high[i-lookback:i] and high[i] > high[i+1:i+lookback]
    Returns dict with swing_highs, swing_lows as lists of (index, price).
    """
    if len(df) < lookback * 2 + 1:
        return {"swing_highs": [], "swing_lows": []}

    highs = df["high"].values
    lows = df["low"].values
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        # Swing high
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            swing_highs.append((i, highs[i]))

        # Swing low
        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            swing_lows.append((i, lows[i]))

    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def detect_structure(df: pd.DataFrame) -> dict:
    """Detect market structure: HH/HL (bullish) or LH/LL (bearish).

    Returns dict with: structure, swing_highs, swing_lows, price_supports, price_resistances
    """
    swings = find_swing_points(df)
    sh = swings["swing_highs"]
    sl = swings["swing_lows"]

    structure = "UNCLEAR"
    structure_detail = ""

    # Need at least 2 of each to determine structure
    if len(sh) >= 2 and len(sl) >= 2:
        last_2_highs = [p for _, p in sh[-2:]]
        last_2_lows = [p for _, p in sl[-2:]]

        hh = last_2_highs[1] > last_2_highs[0]  # Higher High
        hl = last_2_lows[1] > last_2_lows[0]     # Higher Low
        lh = last_2_highs[1] < last_2_highs[0]   # Lower High
        ll = last_2_lows[1] < last_2_lows[0]     # Lower Low

        if hh and hl:
            structure = "BULLISH (HH/HL)"
            structure_detail = (f"HH: {last_2_highs[0]:,.0f} → {last_2_highs[1]:,.0f} | "
                                f"HL: {last_2_lows[0]:,.0f} → {last_2_lows[1]:,.0f}")
        elif lh and ll:
            structure = "BEARISH (LH/LL)"
            structure_detail = (f"LH: {last_2_highs[0]:,.0f} → {last_2_highs[1]:,.0f} | "
                                f"LL: {last_2_lows[0]:,.0f} → {last_2_lows[1]:,.0f}")
        elif hh and ll:
            structure = "EXPANDING"
            structure_detail = "Range widening — breakout imminent"
        elif lh and hl:
            structure = "CONTRACTING"
            structure_detail = "Range narrowing — squeeze forming"
        else:
            structure = "MIXED"
            structure_detail = (f"Highs: {last_2_highs[0]:,.0f} → {last_2_highs[1]:,.0f} | "
                                f"Lows: {last_2_lows[0]:,.0f} → {last_2_lows[1]:,.0f}")

    # Price-based support/resistance from swing points
    price_supports = sorted(set(p for _, p in sl[-5:]), reverse=True) if sl else []
    price_resistances = sorted(set(p for _, p in sh[-5:])) if sh else []

    return {
        "structure": structure,
        "structure_detail": structure_detail,
        "swing_highs": [p for _, p in sh[-5:]],
        "swing_lows": [p for _, p in sl[-5:]],
        "price_supports": price_supports,
        "price_resistances": price_resistances,
    }


# ======================================================================
# IV Context (rank & percentile)
# ======================================================================

def compute_iv_context(current_iv: float, iv_history: list[float]) -> dict:
    """Compute IV Rank and IV Percentile.

    iv_history: list of recent IV readings (e.g., daily ATM IV for past 30 days)

    IV Rank   = (current - min) / (max - min) * 100
    IV %ile   = % of readings below current
    """
    if not iv_history or len(iv_history) < 2:
        return {
            "iv_rank": None, "iv_percentile": None,
            "iv_min": 0, "iv_max": 0, "iv_mean": 0,
            "regime": "UNKNOWN", "action": "",
        }

    iv_arr = np.array(iv_history)
    iv_min = iv_arr.min()
    iv_max = iv_arr.max()
    iv_mean = iv_arr.mean()

    # IV Rank
    if iv_max - iv_min > 0:
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
    else:
        iv_rank = 50

    # IV Percentile
    iv_percentile = (iv_arr < current_iv).sum() / len(iv_arr) * 100

    # Classify
    if iv_rank < 25:
        regime = "LOW"
        action = "Options CHEAP — favor buying (debit strategies)"
    elif iv_rank < 50:
        regime = "BELOW AVG"
        action = "Options fairly priced — slight buy bias"
    elif iv_rank < 75:
        regime = "ABOVE AVG"
        action = "Options getting expensive — slight sell bias"
    else:
        regime = "HIGH"
        action = "Options EXPENSIVE — favor selling (credit strategies)"

    return {
        "iv_rank": round(iv_rank, 1),
        "iv_percentile": round(iv_percentile, 1),
        "iv_min": round(iv_min, 1),
        "iv_max": round(iv_max, 1),
        "iv_mean": round(iv_mean, 1),
        "regime": regime,
        "action": action,
    }


# ======================================================================
# Entry trigger system
# ======================================================================

def generate_entry_signals(trend: dict, structure: dict, iv_ctx: dict,
                           pcr_oi: float, spot: float,
                           oi_support: float, oi_resistance: float,
                           atm_iv: float, max_pain: float) -> list[dict]:
    """Generate actionable entry signals combining all inputs.

    Returns list of signal dicts, each with:
      action, strategy, direction, confidence, reasoning, entry_zone, sl, target
    """
    signals = []
    trend_dir = trend.get("trend", "UNKNOWN")
    struct = structure.get("structure", "UNCLEAR")
    iv_regime = iv_ctx.get("regime", "UNKNOWN")
    price_supports = structure.get("price_supports", [])
    price_resistances = structure.get("price_resistances", [])

    # Scoring: each condition adds confidence
    bull_score = 0
    bear_score = 0
    range_score = 0

    # --- Trend signals ---
    if "UPTREND" in trend_dir:
        bull_score += 30
    elif "DOWNTREND" in trend_dir:
        bear_score += 30
    elif trend_dir == "SIDEWAYS":
        range_score += 30

    # --- Structure signals ---
    if "BULLISH" in struct:
        bull_score += 25
    elif "BEARISH" in struct:
        bear_score += 25
    elif struct in ("CONTRACTING", "EXPANDING"):
        range_score += 15

    # --- OI signals ---
    if pcr_oi > 1.2:
        bull_score += 20  # heavy put writing = support
    elif pcr_oi < 0.8:
        bear_score += 20  # heavy call writing = resistance
    else:
        range_score += 15

    # --- Spot vs levels ---
    if oi_support > 0 and spot > 0:
        dist_to_support_pct = (spot - oi_support) / spot * 100
        dist_to_resist_pct = (oi_resistance - spot) / spot * 100 if oi_resistance > spot else 0

        if dist_to_support_pct < 0.3:  # near support
            bull_score += 15
        if dist_to_resist_pct < 0.3:  # near resistance
            bear_score += 15

    # --- IV context ---
    iv_rank = iv_ctx.get("iv_rank") or 50
    cheap_options = iv_rank < 30
    expensive_options = iv_rank > 70

    # === Generate signals based on scores ===

    # Directional: Bullish
    if bull_score >= 50:
        confidence = min(bull_score, 100)
        status = "ENTER" if confidence >= 65 else "PREPARE"

        if cheap_options:
            signals.append({
                "status": status,
                "direction": "BULLISH",
                "strategy": "Long Call (ATM)" if confidence >= 70 else "Bull Call Spread",
                "confidence": confidence,
                "entry_zone": f"₹{spot:,.0f} (near support ₹{oi_support:,.0f})",
                "sl": f"Below ₹{oi_support:,.0f}" + (f" / price support ₹{price_supports[0]:,.0f}" if price_supports else ""),
                "target": f"₹{oi_resistance:,.0f} (OI resistance)",
                "reasoning": _build_reasoning(trend_dir, struct, pcr_oi, iv_regime, "bullish"),
            })
        else:
            signals.append({
                "status": status,
                "direction": "BULLISH",
                "strategy": "Bull Put Spread (sell put)" if expensive_options else "Bull Call Spread",
                "confidence": confidence,
                "entry_zone": f"₹{spot:,.0f}",
                "sl": f"Below ₹{oi_support:,.0f}",
                "target": f"₹{oi_resistance:,.0f}",
                "reasoning": _build_reasoning(trend_dir, struct, pcr_oi, iv_regime, "bullish"),
            })

    # Directional: Bearish
    if bear_score >= 50:
        confidence = min(bear_score, 100)
        status = "ENTER" if confidence >= 65 else "PREPARE"

        if cheap_options:
            signals.append({
                "status": status,
                "direction": "BEARISH",
                "strategy": "Long Put (ATM)" if confidence >= 70 else "Bear Put Spread",
                "confidence": confidence,
                "entry_zone": f"₹{spot:,.0f} (near resistance ₹{oi_resistance:,.0f})",
                "sl": f"Above ₹{oi_resistance:,.0f}" + (f" / price resistance ₹{price_resistances[-1]:,.0f}" if price_resistances else ""),
                "target": f"₹{oi_support:,.0f} (OI support)",
                "reasoning": _build_reasoning(trend_dir, struct, pcr_oi, iv_regime, "bearish"),
            })
        else:
            signals.append({
                "status": status,
                "direction": "BEARISH",
                "strategy": "Bear Call Spread (sell call)" if expensive_options else "Bear Put Spread",
                "confidence": confidence,
                "entry_zone": f"₹{spot:,.0f}",
                "sl": f"Above ₹{oi_resistance:,.0f}",
                "target": f"₹{oi_support:,.0f}",
                "reasoning": _build_reasoning(trend_dir, struct, pcr_oi, iv_regime, "bearish"),
            })

    # Range-bound
    if range_score >= 40 and bull_score < 50 and bear_score < 50:
        confidence = min(range_score + 20, 90)
        status = "ENTER" if confidence >= 60 else "PREPARE"

        if expensive_options:
            signals.append({
                "status": status,
                "direction": "NEUTRAL",
                "strategy": "Iron Condor (sell both sides)",
                "confidence": confidence,
                "entry_zone": f"Range: ₹{oi_support:,.0f} — ₹{oi_resistance:,.0f}",
                "sl": "Break of range ±50 pts",
                "target": "Theta decay / premium collection",
                "reasoning": _build_reasoning(trend_dir, struct, pcr_oi, iv_regime, "range"),
            })
        else:
            signals.append({
                "status": "WAIT",
                "direction": "NEUTRAL",
                "strategy": "Wait for IV rise or directional breakout",
                "confidence": confidence,
                "entry_zone": f"Range: ₹{oi_support:,.0f} — ₹{oi_resistance:,.0f}",
                "sl": "—",
                "target": "—",
                "reasoning": "Range-bound but IV too low to sell premium. Wait for directional trigger or IV spike.",
            })

    # Breakout alert
    if struct in ("CONTRACTING", "EXPANDING"):
        signals.append({
            "status": "ALERT",
            "direction": "BREAKOUT",
            "strategy": "Long Straddle" if cheap_options else "Wait for direction, then trade breakout",
            "confidence": 40,
            "entry_zone": f"₹{spot:,.0f}",
            "sl": "Time-based (exit if no move in 2 hrs)",
            "target": f"Break above ₹{oi_resistance:,.0f} or below ₹{oi_support:,.0f}",
            "reasoning": f"Structure {struct.lower()} — big move likely. "
                         f"{'IV cheap, straddle favorable' if cheap_options else 'IV high, wait for direction first'}.",
        })

    # If no signals generated
    if not signals:
        signals.append({
            "status": "WAIT",
            "direction": "UNCLEAR",
            "strategy": "No clear setup — stay flat",
            "confidence": 0,
            "entry_zone": "—",
            "sl": "—",
            "target": "—",
            "reasoning": "Conflicting signals across trend, structure, OI, and IV. Wait for alignment.",
        })

    # Sort by confidence
    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals


def _build_reasoning(trend: str, structure: str, pcr: float,
                     iv_regime: str, bias: str) -> str:
    """Build human-readable reasoning string."""
    parts = []

    if bias == "bullish":
        if "UPTREND" in trend:
            parts.append(f"Trend: {trend}")
        if "BULLISH" in structure:
            parts.append(f"Structure: {structure}")
        if pcr > 1.0:
            parts.append(f"PCR {pcr:.2f} — put writers supporting")
    elif bias == "bearish":
        if "DOWNTREND" in trend:
            parts.append(f"Trend: {trend}")
        if "BEARISH" in structure:
            parts.append(f"Structure: {structure}")
        if pcr < 1.0:
            parts.append(f"PCR {pcr:.2f} — call writers capping")
    elif bias == "range":
        parts.append(f"Trend: {trend}")
        if "CONTRACT" in structure or "EXPAND" in structure:
            parts.append(f"Structure: {structure}")
        parts.append(f"PCR {pcr:.2f} — balanced")

    parts.append(f"IV: {iv_regime}")
    return " | ".join(parts)


# ======================================================================
# Composite trade signal
# ======================================================================

def generate_composite_signal(
    trend: dict, structure: dict, iv_ctx: dict,
    pcr_oi: float, spot: float,
    max_pain: float,
    vix_ltp: float = None,
    futures_basis: dict = None,
) -> dict:
    """Generate a weighted composite signal from -100 (bearish) to +100 (bullish).

    Components and weights:
    - Trend (25%): EMA alignment and price position
    - OI/PCR (25%): Put-call ratio and max pain position
    - Structure (15%): Swing highs/lows pattern
    - IV/VIX (15%): Volatility regime
    - Futures basis (10%): Premium = bullish, discount = bearish
    - Mean reversion (10%): Spot vs max pain distance
    """
    components = []
    score = 0.0

    # --- 1. Trend (25%) ---
    trend_score = 0
    t = trend.get("trend", "UNKNOWN")
    strength = trend.get("strength", 0)
    if "UPTREND" in t:
        trend_score = min(strength * 25, 100)
    elif "DOWNTREND" in t:
        trend_score = -min(strength * 25, 100)
    components.append({"name": "Trend", "score": trend_score, "weight": 25,
                       "detail": f"{t} (strength {strength})"})
    score += trend_score * 0.25

    # --- 2. OI / PCR (25%) ---
    pcr_score = 0
    if pcr_oi > 1.3:
        pcr_score = min((pcr_oi - 1.0) * 100, 100)
    elif pcr_oi < 0.7:
        pcr_score = max((pcr_oi - 1.0) * 100, -100)
    else:
        pcr_score = (pcr_oi - 1.0) * 100
    components.append({"name": "PCR/OI", "score": round(pcr_score), "weight": 25,
                       "detail": f"PCR {pcr_oi:.2f}"})
    score += pcr_score * 0.25

    # --- 3. Structure (15%) ---
    struct_score = 0
    s = structure.get("pattern", "UNKNOWN")
    if "BULLISH" in s:
        struct_score = 60
    elif "BEARISH" in s:
        struct_score = -60
    elif "EXPAND" in s:
        struct_score = 20 if "UPTREND" in t else -20
    components.append({"name": "Structure", "score": struct_score, "weight": 15,
                       "detail": s})
    score += struct_score * 0.15

    # --- 4. IV / VIX (15%) ---
    iv_score = 0
    regime = iv_ctx.get("regime", "UNKNOWN") if iv_ctx else "UNKNOWN"
    if regime == "LOW":
        iv_score = 30  # cheap options, favor buying
    elif regime == "HIGH":
        iv_score = -30  # expensive options, favor selling
    elif regime == "ELEVATED":
        iv_score = -15
    if vix_ltp is not None:
        if vix_ltp > 20:
            iv_score -= 20  # high fear
        elif vix_ltp < 13:
            iv_score += 15  # low fear / complacency
    components.append({"name": "IV/VIX", "score": round(iv_score), "weight": 15,
                       "detail": f"Regime: {regime}" + (f", VIX: {vix_ltp:.1f}" if vix_ltp else "")})
    score += iv_score * 0.15

    # --- 5. Futures basis (10%) ---
    basis_score = 0
    if futures_basis:
        bp = futures_basis.get("basis_pct", 0)
        if bp > 0.1:
            basis_score = min(bp * 200, 80)
        elif bp < -0.1:
            basis_score = max(bp * 200, -80)
        components.append({"name": "Futures", "score": round(basis_score), "weight": 10,
                           "detail": f"{futures_basis['status']} {bp:+.3f}%"})
    else:
        components.append({"name": "Futures", "score": 0, "weight": 10, "detail": "N/A"})
    score += basis_score * 0.10

    # --- 6. Spot vs Max Pain (10%) ---
    mp_score = 0
    if max_pain > 0 and spot > 0:
        dist_pct = (spot - max_pain) / max_pain * 100
        # Spot above max pain = bearish pull-down expected, below = bullish pull-up
        mp_score = max(min(-dist_pct * 20, 80), -80)
        components.append({"name": "Max Pain", "score": round(mp_score), "weight": 10,
                           "detail": f"Spot {dist_pct:+.1f}% from MP {max_pain:.0f}"})
    else:
        components.append({"name": "Max Pain", "score": 0, "weight": 10, "detail": "N/A"})
    score += mp_score * 0.10

    # --- Composite ---
    total = round(max(min(score, 100), -100))

    if total >= 50:
        bias = "STRONG_BUY"
    elif total >= 20:
        bias = "BUY"
    elif total <= -50:
        bias = "STRONG_SELL"
    elif total <= -20:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    return {
        "score": total,
        "bias": bias,
        "components": components,
    }
