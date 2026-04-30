"""
indicators.py
-------------
All indicator implementations for the ORB trading bot.

Indicators computed:
  EMA Fast     — 9-period EMA
  EMA Slow     — 20-period EMA
  EMA Macro    — 50-period EMA (session macro trend)
  VWAP         — Volume-Weighted Average Price (anchored per day)
  RSI          — 14-period RSI (Wilder's smoothing)
  Volume Avg   — Rolling mean over VOLUME_LOOKBACK candles
  SuperTrend   — ATR-based trailing stop; flips direction on trend reversal.
                 Used in v3 as a profit-protection exit (ST_EXIT).
  ORB 15-min   — Opening range (9:15–9:30 IST) high/low/established
  ORB 30-min   — Opening range (9:15–9:45 IST) high/low/established
  Prev Close   — Previous trading day's last close (for gap calculation)
  Day Open     — First candle Open of the trading day (for gap calculation)
"""

import numpy as np
import pandas as pd
import pytz

from config import (
    EMA_FAST,
    EMA_MACRO,
    EMA_SLOW,
    ORB_MINUTES,
    ORB_MINUTES_SECONDARY,
    ORB_SUPERTREND_MULTIPLIER,
    ORB_SUPERTREND_PERIOD,
    RSI_PERIOD,
    VOLUME_LOOKBACK,
)

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Column name constants (imported by strategy module)
# ---------------------------------------------------------------------------
EMA_FAST_COL        = "ema_fast"
EMA_SLOW_COL        = "ema_slow"
EMA_MACRO_COL       = "ema_macro"
VWAP_COL            = "vwap"
RSI_COL             = "rsi"
VOLAVG_COL          = "vol_avg"

# Primary ORB (15-min)
ORB_HIGH_COL        = "orb_high"
ORB_LOW_COL         = "orb_low"
ORB_ESTABLISHED_COL = "orb_established"

# Secondary ORB (30-min)
ORB_HIGH_30_COL     = "orb_high_30"
ORB_LOW_30_COL      = "orb_low_30"
ORB_EST_30_COL      = "orb_established_30"

PREV_DAY_CLOSE_COL  = "prev_day_close"
DAY_OPEN_COL        = "day_open"

# SuperTrend (v3 — profit-protection exit signal)
ST_BULL_COL         = "st_bull"     # True = bullish (price above ST support)
ST_LINE_COL         = "st_line"     # the trailing line value (support/resistance)


# ---------------------------------------------------------------------------
# EMA — matches TradingView's ta.ema()
# ---------------------------------------------------------------------------
def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# RSI — Wilder's smoothing, matches TradingView's ta.rsi()
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# VWAP — anchored to each calendar day, resets daily
# ---------------------------------------------------------------------------
def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=df.index)
    for _, group_idx in df.groupby(df.index.date).groups.items():
        grp       = df.loc[group_idx]
        tp        = (grp["High"] + grp["Low"] + grp["Close"]) / 3
        cum_vol   = grp["Volume"].cumsum()
        cum_tpvol = (tp * grp["Volume"]).cumsum()
        result.loc[group_idx] = (cum_tpvol / cum_vol.replace(0, np.nan)).values
    return result


# ---------------------------------------------------------------------------
# ORB — Opening Range for a given window length (in minutes)
# ---------------------------------------------------------------------------
def _opening_range(df: pd.DataFrame, window_minutes: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute ORB high, low, and established flag for each trading day.

    window_minutes: length of the opening range window.
      15 → 9:15–9:30 IST (primary ORB)
      30 → 9:15–9:45 IST (secondary ORB)

    Returns (orb_high, orb_low, orb_established), all indexed to df.
    orb_established is True only for candles AFTER the window has closed.
    """
    orb_high        = pd.Series(np.nan,  index=df.index)
    orb_low         = pd.Series(np.nan,  index=df.index)
    orb_established = pd.Series(False,   index=df.index)

    for date, group_idx in df.groupby(df.index.date).groups.items():
        grp = df.loc[group_idx].sort_index()
        if grp.empty:
            continue

        first_ts = grp.index[0]
        orb_end  = first_ts + pd.Timedelta(minutes=window_minutes)

        orb_candles = grp[grp.index < orb_end]
        if orb_candles.empty:
            continue

        high = orb_candles["High"].max()
        low  = orb_candles["Low"].min()

        orb_high.loc[group_idx] = high
        orb_low.loc[group_idx]  = low

        post_orb_idx = grp[grp.index >= orb_end].index
        orb_established.loc[post_orb_idx] = True

    return orb_high, orb_low, orb_established


# ---------------------------------------------------------------------------
# Prev Day Close + Day Open — for gap-direction filter
# ---------------------------------------------------------------------------
def _prev_day_close_and_day_open(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prev_close_series = pd.Series(np.nan, index=df.index)
    day_open_series   = pd.Series(np.nan, index=df.index)

    dates = sorted(set(df.index.date))

    for i, date in enumerate(dates):
        today_mask = df.index.date == date
        today_idx  = df[today_mask].index
        today_df   = df.loc[today_idx].sort_index()

        if today_df.empty:
            continue
        day_open_series.loc[today_idx] = float(today_df["Open"].iloc[0])

        if i == 0:
            continue
        prev_date  = dates[i - 1]
        prev_mask  = df.index.date == prev_date
        prev_df    = df[prev_mask].sort_index()
        if prev_df.empty:
            continue
        prev_close = float(prev_df["Close"].iloc[-1])
        prev_close_series.loc[today_idx] = prev_close

    return prev_close_series, day_open_series


# ---------------------------------------------------------------------------
# SuperTrend — ATR-based trailing stop / trend direction
# ---------------------------------------------------------------------------
def _supertrend(
    df:         pd.DataFrame,
    period:     int   = ORB_SUPERTREND_PERIOD,
    multiplier: float = ORB_SUPERTREND_MULTIPLIER,
) -> tuple[pd.Series, pd.Series]:
    """
    Standard SuperTrend indicator (continuous, not day-anchored).

    Algorithm:
      1. ATR via Wilder's smoothing (EWM, alpha = 1/period)
      2. Basic bands = midpoint ± multiplier × ATR
      3. Final bands track in one direction, reset only when price crosses them
      4. Direction: bullish when close ≥ final lower band; bearish otherwise

    Returns:
      st_bull  : pd.Series[bool]  — True = bullish (price above trailing support)
      st_line  : pd.Series[float] — the SuperTrend line value
                   (final_lower when bullish, final_upper when bearish)

    For v3 exit logic:
      BUY  position: exit when st_bull flips False (trend turns bearish)
      SELL position: exit when st_bull flips True  (trend turns bullish)
      Only fires when position gain ≥ ORB_SUPERTREND_MIN_GAIN_R (profit protection only).
    """
    close = df["Close"]
    hl2   = (df["High"] + df["Low"]) / 2

    # True Range → ATR (Wilder's smoothing = EWM with alpha=1/period)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - close.shift(1)).abs(),
        (df["Low"]  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    basic_upper = (hl2 + multiplier * atr).to_numpy(dtype=float)
    basic_lower = (hl2 - multiplier * atr).to_numpy(dtype=float)
    close_arr   = close.to_numpy(dtype=float)
    n           = len(df)

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    bull        = np.ones(n, dtype=bool)   # True = bullish; first candle assumed bullish

    for i in range(1, n):
        # Final upper: only moves DOWN (tightens), resets if price crossed above it
        if close_arr[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]          # reset after breakout above
        else:
            final_upper[i] = min(basic_upper[i], final_upper[i - 1])

        # Final lower: only moves UP (tightens), resets if price crossed below it
        if close_arr[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]          # reset after breakdown below
        else:
            final_lower[i] = max(basic_lower[i], final_lower[i - 1])

        # Direction: stay bullish unless close falls below lower band;
        #            stay bearish unless close rises above upper band
        if bull[i - 1]:
            bull[i] = close_arr[i] >= final_lower[i]
        else:
            bull[i] = close_arr[i] > final_upper[i]

    # ST line = support when bullish, resistance when bearish
    st_line = np.where(bull, final_lower, final_upper)

    return (
        pd.Series(bull,    index=df.index, dtype=bool),
        pd.Series(st_line, index=df.index, dtype=float),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append all indicators to a copy of df.

    Works on multi-day DataFrames for proper EMA/RSI warmup.
    VWAP, ORB, PREV_DAY_CLOSE, and DAY_OPEN all reset per day.
    """
    df = df.copy()

    df[EMA_FAST_COL]  = _ema(df["Close"], EMA_FAST)
    df[EMA_SLOW_COL]  = _ema(df["Close"], EMA_SLOW)
    df[EMA_MACRO_COL] = _ema(df["Close"], EMA_MACRO)
    df[VWAP_COL]      = _vwap_daily(df)
    df[RSI_COL]       = _rsi(df["Close"])
    df[VOLAVG_COL]    = df["Volume"].rolling(window=VOLUME_LOOKBACK).mean()

    # SuperTrend — profit-protection exit for v3
    st_bull, st_line  = _supertrend(df)
    df[ST_BULL_COL]   = st_bull
    df[ST_LINE_COL]   = st_line

    # Primary ORB (15-min)
    orb_h, orb_l, orb_est = _opening_range(df, ORB_MINUTES)
    df[ORB_HIGH_COL]        = orb_h
    df[ORB_LOW_COL]         = orb_l
    df[ORB_ESTABLISHED_COL] = orb_est

    # Secondary ORB (30-min)
    orb_h30, orb_l30, orb_est30 = _opening_range(df, ORB_MINUTES_SECONDARY)
    df[ORB_HIGH_30_COL] = orb_h30
    df[ORB_LOW_30_COL]  = orb_l30
    df[ORB_EST_30_COL]  = orb_est30

    prev_close, day_open = _prev_day_close_and_day_open(df)
    df[PREV_DAY_CLOSE_COL] = prev_close
    df[DAY_OPEN_COL]       = day_open

    return df
