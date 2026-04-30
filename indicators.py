"""
indicators.py
-------------
All indicator implementations for the ORB trading bot.

New vs v1: computes BOTH 15-min and 30-min ORB windows so the strategy
can check both in one pass without re-fetching data.

Indicators computed:
  EMA Fast   — 9-period EMA
  EMA Slow   — 20-period EMA
  EMA Macro  — 50-period EMA (session macro trend)
  VWAP       — Volume-Weighted Average Price (anchored per day)
  RSI        — 14-period RSI (Wilder's smoothing)
  Volume Avg — Rolling mean over VOLUME_LOOKBACK candles
  ORB 15-min — Opening range (9:15–9:30 IST) high/low/established
  ORB 30-min — Opening range (9:15–9:45 IST) high/low/established
  Prev Close — Previous trading day's last close (for gap calculation)
  Day Open   — First candle Open of the trading day (for gap calculation)
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
