"""
market_regime.py
----------------
Intraday NIFTY50 market regime detection.

Returns a regime dict that main.py uses to:
  - Cap max_positions (fewer trades on bear days)
  - Set direction_filter (LONG_ONLY / SHORT_ONLY / BOTH)

NIFTY scoring:
  1. VWAP position  (weight 0.40) — above/below daily VWAP
  2. EMA trend      (weight 0.40) — 9 EMA vs 50 EMA
  3. Day change     (weight 0.20) — price vs session open

  BULL  (score > +0.20) → LONG_ONLY,  full ORB_MAX_POSITIONS
  NEUTRAL               → BOTH,       REGIME_NEUTRAL_MAX_POSITIONS
  BEAR  (score < -0.20) → SHORT_ONLY, REGIME_BEAR_MAX_POSITIONS
"""

import logging
import time

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from config import (
    CANDLE_INTERVAL,
    ORB_MAX_POSITIONS,
    REGIME_BEAR_MAX_POSITIONS,
    REGIME_BEAR_THRESHOLD,
    REGIME_BULL_MAX_POSITIONS,
    REGIME_BULL_THRESHOLD,
    REGIME_NEUTRAL_MAX_POSITIONS,
)
from indicators import add_indicators

IST          = pytz.timezone("Asia/Kolkata")
logger       = logging.getLogger(__name__)
NIFTY_TICKER = "^NSEI"

_NEUTRAL_REGIME = {
    "regime":           "NEUTRAL",
    "score":            0.0,
    "max_positions":    ORB_MAX_POSITIONS,
    "direction_filter": "BOTH",
}


def _fetch_nifty(period: str = "5d") -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            df = yf.Ticker(NIFTY_TICKER).history(interval=CANDLE_INTERVAL, period=period)
            if df.empty:
                logger.warning(f"NIFTY: empty data on attempt {attempt + 1}")
                time.sleep(2)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(IST)
            else:
                df.index = df.index.tz_convert(IST)
            return df
        except Exception as e:
            logger.warning(f"NIFTY fetch error attempt {attempt + 1}: {e}")
            time.sleep(2)
    logger.error("NIFTY: all fetch attempts failed")
    return None


def get_nifty_regime() -> dict:
    """
    Compute the current intraday NIFTY50 market regime.
    Returns a dict with regime, score, max_positions, direction_filter.
    """
    df = _fetch_nifty()
    if df is None or len(df) < 10:
        logger.warning("NIFTY data unavailable — defaulting to NEUTRAL")
        return dict(_NEUTRAL_REGIME)

    try:
        df_ind   = add_indicators(df)
        today    = df_ind.index[-1].date()
        today_df = df_ind[df_ind.index.date == today]

        if len(today_df) < 3:
            return dict(_NEUTRAL_REGIME)

        row   = today_df.iloc[-2]
        close = float(row["Close"])

        components: list[float] = []
        weights:    list[float] = []

        vwap = row.get("vwap")
        if not pd.isna(vwap) and float(vwap) > 0:
            vwap_dev = (close - float(vwap)) / float(vwap)
            components.append(float(np.tanh(vwap_dev * 100)))
            weights.append(0.40)

        ema9  = row.get("ema_fast")
        ema50 = row.get("ema_macro")
        if not any(pd.isna(x) for x in (ema9, ema50)):
            components.append(1.0 if float(ema9) > float(ema50) else -1.0)
            weights.append(0.40)

        day_open = row.get("day_open")
        if not pd.isna(day_open) and float(day_open) > 0:
            day_chg = (close - float(day_open)) / float(day_open)
            components.append(float(np.tanh(day_chg * 50)))
            weights.append(0.20)

        if not components:
            return dict(_NEUTRAL_REGIME)

        total_w = sum(weights)
        score   = float(np.clip(
            sum(c * w / total_w for c, w in zip(components, weights)),
            -1.0, 1.0
        ))

        if score > REGIME_BULL_THRESHOLD:
            result = {
                "regime":           "BULL",
                "score":            score,
                "max_positions":    REGIME_BULL_MAX_POSITIONS,
                "direction_filter": "LONG_ONLY",
            }
        elif score < REGIME_BEAR_THRESHOLD:
            result = {
                "regime":           "BEAR",
                "score":            score,
                "max_positions":    REGIME_BEAR_MAX_POSITIONS,
                "direction_filter": "SHORT_ONLY",
            }
        else:
            result = {
                "regime":           "NEUTRAL",
                "score":            score,
                "max_positions":    REGIME_NEUTRAL_MAX_POSITIONS,
                "direction_filter": "BOTH",
            }

        logger.info(
            f"NIFTY: {result['regime']} (score={score:+.3f} nifty={close:.0f}) | "
            f"max_pos={result['max_positions']} dir={result['direction_filter']}"
        )
        return result

    except Exception as e:
        logger.warning(f"Regime computation error: {e} — defaulting to NEUTRAL")
        return dict(_NEUTRAL_REGIME)
