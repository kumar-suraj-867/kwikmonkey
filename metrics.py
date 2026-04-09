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
    """Add Greeks columns to the option chain DataFrame (vectorized).

    Expects columns: strike, option_type, ltp (and optionally iv).
    """
    if df.empty or T <= 0:
        for col in ("delta", "gamma", "theta", "vega"):
            df[col] = 0.0
        if "iv" not in df.columns:
            df["iv"] = 0.0
        return df

    df = df.copy()
    K = df["strike"].values.astype(float)
    is_ce = (df["option_type"] == "CE").values

    # --- IV: use API-provided when available, else Newton-Raphson ---
    iv_raw = df["iv"].values.astype(float) if "iv" in df.columns else np.zeros(len(df))
    sigma = np.where((iv_raw > 0) & ~np.isnan(iv_raw), iv_raw / 100, 0.0)

    # Compute IV only for rows missing it (much fewer Newton-Raphson calls)
    needs_iv = sigma <= 0
    if needs_iv.any():
        ltp_arr = df["ltp"].values.astype(float)
        for idx in np.where(needs_iv)[0]:
            opt_type = "CE" if is_ce[idx] else "PE"
            computed = implied_volatility(ltp_arr[idx], spot, K[idx], T, r, opt_type)
            if computed and computed > 0:
                sigma[idx] = computed

    # --- Vectorized Greeks ---
    valid = (sigma > 0) & (K > 0)
    S = float(spot)
    sqrt_T = np.sqrt(T)

    d1 = np.zeros(len(df))
    d2 = np.zeros(len(df))
    d1[valid] = (np.log(S / K[valid]) + (r + 0.5 * sigma[valid]**2) * T) / (sigma[valid] * sqrt_T)
    d2[valid] = d1[valid] - sigma[valid] * sqrt_T

    nd1 = norm.cdf(d1)
    npdf_d1 = norm.pdf(d1)
    nd2 = norm.cdf(d2)

    # Delta
    delta = np.where(is_ce, nd1, nd1 - 1)
    delta[~valid] = 0

    # Gamma
    gamma = np.zeros(len(df))
    gamma[valid] = npdf_d1[valid] / (S * sigma[valid] * sqrt_T)

    # Theta (per calendar day)
    theta = np.zeros(len(df))
    common = -(S * npdf_d1 * sigma) / (2 * sqrt_T)
    theta_ce = common - r * K * np.exp(-r * T) * nd2
    theta_pe = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = np.where(is_ce, theta_ce, theta_pe) / 365
    theta[~valid] = 0

    # Vega (per 1% vol move)
    vega = np.zeros(len(df))
    vega[valid] = S * npdf_d1[valid] * sqrt_T / 100

    df["delta"] = np.round(delta, 4)
    df["gamma"] = np.round(gamma, 6)
    df["theta"] = np.round(theta, 2)
    df["vega"] = np.round(vega, 2)
    df["iv"] = np.round(sigma * 100, 2)
    return df


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


def calculate_futures_basis(spot_ltp: float, futures_ltp: float,
                           days_to_expiry: float) -> dict:
    """Compute futures basis (premium/discount) vs spot."""
    basis = futures_ltp - spot_ltp
    basis_pct = (basis / spot_ltp * 100) if spot_ltp > 0 else 0
    annualized = (basis_pct * 365 / days_to_expiry) if days_to_expiry > 0 else 0
    return {
        "basis": round(basis, 2),
        "basis_pct": round(basis_pct, 3),
        "annualized_pct": round(annualized, 2),
        "status": "Premium" if basis >= 0 else "Discount",
    }


def compare_expiry_oi(chain1: pd.DataFrame, chain2: pd.DataFrame,
                      label1: str = "Weekly", label2: str = "Monthly") -> dict:
    """Compare OI across two expiry chains.

    Returns per-strike comparison data and aggregate metrics.
    """
    def _agg(df):
        ce = df[df["option_type"] == "CE"]
        pe = df[df["option_type"] == "PE"]
        return {
            "ce_oi": ce.set_index("strike")["oi"].to_dict(),
            "pe_oi": pe.set_index("strike")["oi"].to_dict(),
            "total_ce_oi": int(ce["oi"].sum()),
            "total_pe_oi": int(pe["oi"].sum()),
            "pcr": round(pe["oi"].sum() / ce["oi"].sum(), 3) if ce["oi"].sum() > 0 else 0,
        }

    agg1 = _agg(chain1) if not chain1.empty else {"ce_oi": {}, "pe_oi": {}, "total_ce_oi": 0, "total_pe_oi": 0, "pcr": 0}
    agg2 = _agg(chain2) if not chain2.empty else {"ce_oi": {}, "pe_oi": {}, "total_ce_oi": 0, "total_pe_oi": 0, "pcr": 0}

    all_strikes = sorted(set(list(agg1["ce_oi"].keys()) + list(agg1["pe_oi"].keys()) +
                              list(agg2["ce_oi"].keys()) + list(agg2["pe_oi"].keys())))
    return {
        "strikes": all_strikes,
        label1: agg1,
        label2: agg2,
        "labels": (label1, label2),
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
