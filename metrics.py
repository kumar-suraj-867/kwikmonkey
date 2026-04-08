"""Option metrics: Black-Scholes Greeks, IV, PCR, Max Pain."""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes core
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str) -> float:
    """Black-Scholes option price. option_type: 'CE' or 'PE'."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "CE" else max(K - S, 0)
        return max(intrinsic, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(market_price: float, S: float, K: float, T: float,
                       r: float, option_type: str,
                       tol: float = 1e-6, max_iter: int = 100) -> float | None:
    """Compute IV using Newton-Raphson. Returns None if it doesn't converge."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    sigma = 0.3  # initial guess

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = _vega(S, K, T, r, sigma)

        if vega < 1e-12:
            break

        sigma -= (price - market_price) / vega

        if sigma <= 0.001:
            sigma = 0.001
        if sigma > 5.0:
            return None
        if abs(price - market_price) < tol:
            return sigma

    return sigma if abs(bs_price(S, K, T, r, sigma, option_type) - market_price) < 1.0 else None


# ---------------------------------------------------------------------------
# Individual Greeks
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def _vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str) -> dict:
    """Return dict with delta, gamma, theta, vega, iv for one option."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)

    # Delta
    delta = norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)

    # Theta (per day)
    common = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == "CE":
        theta = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta /= 365  # per calendar day

    # Vega (per 1% move in vol)
    vega = S * norm.pdf(d1) * sqrt_T / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
    }


# ---------------------------------------------------------------------------
# Enrich option chain DataFrame with Greeks
# ---------------------------------------------------------------------------

def enrich_with_greeks(df: pd.DataFrame, spot: float, r: float,
                       T: float) -> pd.DataFrame:
    """Add Greeks columns to the option chain DataFrame.

    Expects columns: strike, option_type, ltp (and optionally iv).
    """
    greeks_data = []

    for _, row in df.iterrows():
        strike = row["strike"]
        opt_type = row["option_type"]
        ltp = row.get("ltp", 0)

        # Compute IV if not provided
        iv = row.get("iv")
        if iv is None or iv <= 0 or pd.isna(iv):
            iv = implied_volatility(ltp, spot, strike, T, r, opt_type)
        else:
            iv = iv / 100  # convert from percentage

        if iv and iv > 0:
            g = calculate_greeks(spot, strike, T, r, iv, opt_type)
            g["iv"] = round(iv * 100, 2)
        else:
            g = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": 0}

        greeks_data.append(g)

    greeks_df = pd.DataFrame(greeks_data)
    # Drop columns from df that greeks_df will replace to avoid duplicates
    overlap = [c for c in greeks_df.columns if c in df.columns]
    df_clean = df.drop(columns=overlap).reset_index(drop=True)
    return pd.concat([df_clean, greeks_df], axis=1)


# ---------------------------------------------------------------------------
# OI Metrics
# ---------------------------------------------------------------------------

def calculate_pcr(df: pd.DataFrame) -> dict:
    """Compute Put-Call Ratio from option chain DataFrame."""
    calls = df[df["option_type"] == "CE"]
    puts = df[df["option_type"] == "PE"]

    total_call_oi = calls["oi"].sum()
    total_put_oi = puts["oi"].sum()
    total_call_vol = calls["volume"].sum()
    total_put_vol = puts["volume"].sum()

    pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0
    pcr_vol = round(total_put_vol / total_call_vol, 3) if total_call_vol > 0 else 0

    return {
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_vol,
        "total_call_oi": int(total_call_oi),
        "total_put_oi": int(total_put_oi),
        "total_call_volume": int(total_call_vol),
        "total_put_volume": int(total_put_vol),
    }


def calculate_max_pain(df: pd.DataFrame) -> float:
    """Find the strike where total option buyers' loss is maximized (writers' gain).

    For each candidate strike K:
      pain = sum over all strikes of:
        call_oi[strike] * max(strike - K, 0)   (calls expire worthless below K)
      + put_oi[strike]  * max(K - strike, 0)    (puts expire worthless above K)
    Max pain = K with minimum total pain for buyers.
    """
    calls = df[df["option_type"] == "CE"].set_index("strike")["oi"]
    puts = df[df["option_type"] == "PE"].set_index("strike")["oi"]

    strikes = sorted(df["strike"].unique())
    if not strikes:
        return 0

    min_pain = float("inf")
    max_pain_strike = strikes[0]

    for K in strikes:
        call_pain = sum(
            oi * max(strike - K, 0) for strike, oi in calls.items()
        )
        put_pain = sum(
            oi * max(K - strike, 0) for strike, oi in puts.items()
        )
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = K

    return max_pain_strike
