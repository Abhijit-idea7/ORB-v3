"""
strategy_orb.py
---------------
ORB v3 — Opening Range Breakout with Time-Decay Stop Management.

SOURCE: Andrew Aziz, "Advanced Techniques in Day Trading"
V3 CONCEPT: "A position entered is assumed wrong until the market proves it
right within a defined time window." — progressive SL tightening.

ENTRY RULES (15-min primary window only — same as v2)
------------------------------------------------------
  1. ORB window has closed (established flag = True)
  2. Before entry cutoff time (11:00 IST)
  3. Gap direction aligned: gap-up → LONG only; gap-down → SHORT only
  4. Close > ORB high (BUY) or Close < ORB low (SELL)
  5. Extension ≤ chase limit (< 1.0% beyond ORB level)
  6. Volume on breakout candle ≥ 1.15 × 10-candle average
  7. Close > VWAP (BUY) or Close < VWAP (SELL) — fair-value confirmation
  8. ORB range: 0.3%–4% of price (meaningful but not extreme)

STOP LOSS   BUY: ORB Low  |  SELL: ORB High
TARGET      BUY/SELL: entry ± (ORB range × 1.5)

EXIT SIGNALS (v3 — evaluated in priority order)
------------------------------------------------
  TARGET      — candle High/Low touches target (intrabar detection)
  STOP_LOSS   — candle Low/High breaches effective SL.
                Effective SL is the TIGHTEST (most protective) of:
                  • Original ORB SL (always the floor)
                  • Profit ratchet 0.5R → SL moved to breakeven
                  • Profit ratchet 1.0R → SL locks 40% of gain
                  • T+20: gain < 0R → tight SL = entry ± 30% of initial risk
                  • T+40: gain < 0.3R → SL forced to breakeven
                Note: effective_sl is stored in position["_v3_effective_sl"]
                so backtest.py can use the correct exit price.
  ST_EXIT     — SuperTrend (7,3.0) on 2-min chart flips against position
                AND gain ≥ 0.3R. Pure profit-protection: catches momentum
                reversals before the ratchet SL or target are reached.
                Below 0.3R gain, the 2-min ST is too noisy — time-decay
                handles flat/losing trades instead.
  TIME_EXIT   — T+60: trade hasn't reached 0.5R gain → exit at market (close)
  ORB_FAILED  — close pulled back > 0.5% inside ORB range
  SQUARE_OFF  — forced close at 15:10 IST (handled by main loop)
"""

import logging
from datetime import datetime

import pandas as pd
import pytz

from config import (
    ORB_CHASE_LIMIT_PCT,
    ORB_CHASE_LIMIT_SECONDARY,
    ORB_ENTRY_CUTOFF_SECONDARY,
    ORB_ENTRY_CUTOFF_TIME,
    ORB_FAILED_BUFFER_PCT,
    ORB_MAX_RANGE_PCT,
    ORB_MIN_GAP_PCT,
    ORB_MIN_RANGE_PCT,
    ORB_POSITION_SCALE,
    ORB_SECONDARY_WINDOW_ENABLED,
    ORB_SUPERTREND_MIN_GAIN_R,
    ORB_TARGET_MULTIPLIER,
    ORB_V3_CHECKPOINT_1_MINS,
    ORB_V3_CHECKPOINT_2_MINS,
    ORB_V3_RATCHET_BE_R,
    ORB_V3_RATCHET_LOCK_R,
    ORB_V3_T20_MIN_GAIN_R,
    ORB_V3_T20_TIGHT_SL_FACTOR,
    ORB_V3_T40_MIN_GAIN_R,
    ORB_V3_TIME_EXIT_MIN_GAIN_R,
    ORB_V3_TIME_EXIT_MINS,
    ORB_V3_TRAIL_LOCK_FRACTION,
    ORB_VOLUME_MULT_SECONDARY,
    ORB_VOLUME_MULTIPLIER,
)
from indicators import (
    DAY_OPEN_COL,
    ORB_EST_30_COL,
    ORB_ESTABLISHED_COL,
    ORB_HIGH_30_COL,
    ORB_HIGH_COL,
    ORB_LOW_30_COL,
    ORB_LOW_COL,
    PREV_DAY_CLOSE_COL,
    ST_BULL_COL,
    ST_LINE_COL,
    VOLAVG_COL,
    VWAP_COL,
)

IST    = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)
_HOLD  = {"action": "HOLD", "sl": 0.0, "target": 0.0}

STRATEGY_NAME = "ORB"


def _is_past_cutoff(now_ist: datetime, cutoff_str: str) -> bool:
    h, m = map(int, cutoff_str.split(":"))
    return now_ist >= now_ist.replace(hour=h, minute=m, second=0, microsecond=0)


def _minutes_since_entry(df: pd.DataFrame, position: dict) -> float:
    """
    Return minutes elapsed between trade entry and the current evaluation candle.

    entry_time is stored as "HH:MM" string in position dict (by both trade_tracker.py
    and backtest.py). We reconstruct the full timestamp using the candle's calendar date
    to handle overnight or next-day edge cases correctly.

    Returns 0.0 if entry_time is missing or can't be parsed.
    """
    entry_time_str = position.get("entry_time")
    if not entry_time_str:
        return 0.0
    try:
        candle_ts = df.index[-2]          # same row used for close/high/low evaluation
        parts     = str(entry_time_str).split(":")
        h, m      = int(parts[0]), int(parts[1])
        # Build entry datetime on the candle's date (preserves timezone)
        entry_dt  = candle_ts.normalize() + pd.Timedelta(hours=h, minutes=m)
        elapsed   = (candle_ts - entry_dt).total_seconds() / 60.0
        return max(0.0, elapsed)
    except Exception:
        return 0.0


def _check_orb_window(
    row,
    symbol:      str,
    now_ist:     datetime,
    cutoff_time: str,
    orb_high_key: str,
    orb_low_key:  str,
    orb_est_key:  str,
    vol_multiplier: float,
    chase_limit:    float,
    window_label:   str,
    gap_pct:        float,
    gap_up:         bool,
    gap_down:       bool,
    close:          float,
    vol_ratio:      float,
    vwap:           float,
) -> dict:
    """
    Evaluate one ORB window for entry signal.
    Returns a signal dict or _HOLD.
    """
    if _is_past_cutoff(now_ist, cutoff_time):
        return _HOLD

    if not bool(row.get(orb_est_key, False)):
        return _HOLD

    orb_high_val = row.get(orb_high_key)
    orb_low_val  = row.get(orb_low_key)
    if pd.isna(orb_high_val) or pd.isna(orb_low_val):
        return _HOLD

    orb_high  = float(orb_high_val)
    orb_low   = float(orb_low_val)
    orb_range = orb_high - orb_low

    if orb_range <= 0:
        return _HOLD

    range_pct = orb_range / orb_high
    if range_pct < ORB_MIN_RANGE_PCT:
        logger.info(f"{symbol} ORB-{window_label}: range too narrow ({range_pct:.2%})")
        return _HOLD
    if range_pct > ORB_MAX_RANGE_PCT:
        logger.info(f"{symbol} ORB-{window_label}: range too wide ({range_pct:.2%})")
        return _HOLD

    logger.info(
        f"{symbol} ORB-{window_label}: close={close:.2f} "
        f"orb=[{orb_low:.2f}–{orb_high:.2f}] range={range_pct:.2%} "
        f"gap={gap_pct:+.2%} vol={vol_ratio:.2f}x vwap={vwap:.2f}"
    )

    # ---- LONG: breakout above ORB high ----
    if close > orb_high:
        if gap_down:
            logger.info(f"{symbol} ORB-{window_label}: LONG rejected — gap-down ({gap_pct:+.2%})")
            return _HOLD

        extension = (close - orb_high) / orb_high
        if extension > chase_limit:
            logger.info(f"{symbol} ORB-{window_label}: LONG rejected — chasing {extension:.2%}")
            return _HOLD
        if vol_ratio < vol_multiplier:
            logger.info(f"{symbol} ORB-{window_label}: LONG rejected — weak volume {vol_ratio:.2f}x")
            return _HOLD
        if vwap > 0 and close < vwap:
            logger.info(f"{symbol} ORB-{window_label}: LONG rejected — below VWAP {vwap:.2f}")
            return _HOLD

        sl   = orb_low
        risk = close - sl
        if risk <= 0:
            return _HOLD
        target = close + (orb_range * ORB_TARGET_MULTIPLIER)
        rr     = (target - close) / risk
        logger.info(
            f"{symbol} ORB-{window_label}: *** BUY *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"R:R={rr:.1f} vol={vol_ratio:.2f}x gap={gap_pct:+.2%}"
        )
        return {
            "action": "BUY",
            "sl": sl,
            "target": target,
            "strategy": STRATEGY_NAME,
            "quantity_scale": ORB_POSITION_SCALE,
            "orb_breakout_level": orb_high,   # stored for ORB_FAILED exit check
            "window": window_label,
        }

    # ---- SHORT: breakdown below ORB low ----
    if close < orb_low:
        if gap_up:
            logger.info(f"{symbol} ORB-{window_label}: SHORT rejected — gap-up ({gap_pct:+.2%})")
            return _HOLD

        extension = (orb_low - close) / orb_low
        if extension > chase_limit:
            logger.info(f"{symbol} ORB-{window_label}: SHORT rejected — chasing {extension:.2%}")
            return _HOLD
        if vol_ratio < vol_multiplier:
            logger.info(f"{symbol} ORB-{window_label}: SHORT rejected — weak volume {vol_ratio:.2f}x")
            return _HOLD
        if vwap > 0 and close > vwap:
            logger.info(f"{symbol} ORB-{window_label}: SHORT rejected — above VWAP {vwap:.2f}")
            return _HOLD

        sl   = orb_high
        risk = sl - close
        if risk <= 0:
            return _HOLD
        target = close - (orb_range * ORB_TARGET_MULTIPLIER)
        rr     = (close - target) / risk
        logger.info(
            f"{symbol} ORB-{window_label}: *** SELL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"R:R={rr:.1f} vol={vol_ratio:.2f}x gap={gap_pct:+.2%}"
        )
        return {
            "action": "SELL",
            "sl": sl,
            "target": target,
            "strategy": STRATEGY_NAME,
            "quantity_scale": ORB_POSITION_SCALE,
            "orb_breakout_level": orb_low,   # stored for ORB_FAILED exit check
            "window": window_label,
        }

    return _HOLD


def generate_signal(df: pd.DataFrame, symbol: str = "", sim_time=None) -> dict:
    """
    Evaluate the last completed candle (iloc[-2]) for an ORB entry signal.

    Checks primary 15-min window first. If no signal, checks secondary 30-min window
    (only if ORB_SECONDARY_WINDOW_ENABLED — disabled in v3 by default).

    sim_time: candle timestamp for backtesting.
    """
    now_ist = sim_time if sim_time is not None else datetime.now(IST)
    if hasattr(now_ist, "tzinfo") and now_ist.tzinfo is None:
        now_ist = IST.localize(now_ist)

    if len(df) < 3:
        return _HOLD

    row = df.iloc[-2]

    # --- Common values ---
    close   = float(row["Close"])
    volume  = float(row["Volume"])
    vol_avg = float(row[VOLAVG_COL]) if not pd.isna(row.get(VOLAVG_COL)) else 0.0
    vwap    = float(row[VWAP_COL])   if not pd.isna(row.get(VWAP_COL))   else 0.0

    vol_ratio = (volume / vol_avg) if vol_avg > 0 else 0.0

    # Gap calculation
    prev_close = row.get(PREV_DAY_CLOSE_COL)
    day_open   = row.get(DAY_OPEN_COL)
    gap_pct    = 0.0
    if not pd.isna(prev_close) and not pd.isna(day_open) and float(prev_close) > 0:
        gap_pct = (float(day_open) - float(prev_close)) / float(prev_close)

    gap_up   = gap_pct >= ORB_MIN_GAP_PCT
    gap_down = gap_pct <= -ORB_MIN_GAP_PCT

    common = dict(
        symbol=symbol, now_ist=now_ist,
        gap_pct=gap_pct, gap_up=gap_up, gap_down=gap_down,
        close=close, vol_ratio=vol_ratio, vwap=vwap,
    )

    # --- Primary window (15-min ORB) ---
    signal = _check_orb_window(
        row,
        cutoff_time=ORB_ENTRY_CUTOFF_TIME,
        orb_high_key=ORB_HIGH_COL,
        orb_low_key=ORB_LOW_COL,
        orb_est_key=ORB_ESTABLISHED_COL,
        vol_multiplier=ORB_VOLUME_MULTIPLIER,
        chase_limit=ORB_CHASE_LIMIT_PCT,
        window_label="15m",
        **common,
    )
    if signal["action"] in ("BUY", "SELL"):
        return signal

    # --- Secondary window (30-min ORB) — only if enabled in config ---
    if ORB_SECONDARY_WINDOW_ENABLED:
        signal = _check_orb_window(
            row,
            cutoff_time=ORB_ENTRY_CUTOFF_SECONDARY,
            orb_high_key=ORB_HIGH_30_COL,
            orb_low_key=ORB_LOW_30_COL,
            orb_est_key=ORB_EST_30_COL,
            vol_multiplier=ORB_VOLUME_MULT_SECONDARY,
            chase_limit=ORB_CHASE_LIMIT_SECONDARY,
            window_label="30m",
            **common,
        )
        return signal

    return _HOLD


def check_exit_signal(df: pd.DataFrame, position: dict) -> str | None:
    """
    v3 exit logic with time-decay stop management.

    Effective SL is recomputed every call — it is the TIGHTEST (most protective)
    of all applicable rules. The SL only ever moves in the trader's favour:
      • original_sl is the floor (never widened)
      • ratchet and time rules only raise it (BUY) / lower it (SELL)

    For backtest accuracy, the final effective_sl is stored in
    position["_v3_effective_sl"] so backtest.py can use the correct exit price
    for STOP_LOSS trades (not the wider original SL).

    Priority:
      1. TARGET      — intrabar: candle High (BUY) / Low (SELL) touched target
      2. STOP_LOSS   — intrabar: candle Low (BUY) / High (SELL) hit effective SL
      3. ST_EXIT     — SuperTrend flipped against position AND gain ≥ 0.3R
                       Exit at candle close. Profit-protection only.
      4. TIME_EXIT   — T+60 and gain < 0.5R → exit at market (candle close)
      5. ORB_FAILED  — close-based: price closed back inside ORB range by > 0.5%
    """
    if len(df) < 2:
        return None

    row      = df.iloc[-2]
    close    = float(row["Close"])
    candle_h = float(row["High"])
    candle_l = float(row["Low"])

    direction    = position["direction"]
    target       = float(position["target"])
    original_sl  = float(position["sl"])
    entry_price  = float(position.get("entry_price", original_sl))
    initial_risk = abs(entry_price - original_sl)

    # ORB_FAILED breakout level — stored at entry (correct for both windows)
    breakout_level = position.get("orb_breakout_level")
    if breakout_level is None:
        raw = row.get(ORB_HIGH_COL if direction == "BUY" else ORB_LOW_COL)
        breakout_level = float(raw) if raw is not None and not pd.isna(raw) else None

    # --- Minutes elapsed since entry ---
    mins = _minutes_since_entry(df, position)

    # -----------------------------------------------------------------------
    # Compute effective SL — progressive tightening, never widening
    # -----------------------------------------------------------------------
    if direction == "BUY":
        # Current gain in R-units (close-based, conservative)
        gain_R = (close - entry_price) / initial_risk if initial_risk > 0 else 0.0

        # Start from original SL; only ever move UP (more protective)
        eff_sl = original_sl

        # --- Profit ratchet (always active, independent of time) ---
        if initial_risk > 0:
            if gain_R >= ORB_V3_RATCHET_BE_R:
                # 0.5R gain → move SL to breakeven
                eff_sl = max(eff_sl, entry_price)
                logger.debug(f"BUY ratchet BE: gain={gain_R:.2f}R → eff_sl={eff_sl:.2f}")

            if gain_R >= ORB_V3_RATCHET_LOCK_R:
                # 1.0R gain → lock 40% of gain — worst case is now a 0.4R win
                locked = entry_price + ORB_V3_TRAIL_LOCK_FRACTION * initial_risk
                eff_sl = max(eff_sl, locked)
                logger.debug(f"BUY ratchet lock: gain={gain_R:.2f}R → eff_sl={eff_sl:.2f}")

        # --- Time checkpoint 1 (T+20): negative position → tighten SL ---
        if mins >= ORB_V3_CHECKPOINT_1_MINS and gain_R < ORB_V3_T20_MIN_GAIN_R:
            tight = entry_price - ORB_V3_T20_TIGHT_SL_FACTOR * initial_risk
            eff_sl = max(eff_sl, tight)
            logger.info(
                f"BUY T+{mins:.0f}m CP1: gain={gain_R:.2f}R < {ORB_V3_T20_MIN_GAIN_R}R "
                f"→ tight eff_sl={eff_sl:.2f}"
            )

        # --- Time checkpoint 2 (T+40): stalling → force breakeven ---
        if mins >= ORB_V3_CHECKPOINT_2_MINS and gain_R < ORB_V3_T40_MIN_GAIN_R:
            eff_sl = max(eff_sl, entry_price)
            logger.info(
                f"BUY T+{mins:.0f}m CP2: gain={gain_R:.2f}R < {ORB_V3_T40_MIN_GAIN_R}R "
                f"→ breakeven eff_sl={eff_sl:.2f}"
            )

        # Store effective SL so backtest can compute accurate exit price
        position["_v3_effective_sl"] = eff_sl

        # --- Exit checks (priority order) ---
        # 1. TARGET — intrabar high touched target
        if candle_h >= target:
            return "TARGET"

        # 2. STOP_LOSS — intrabar low hit effective SL (time-decay / ratchet)
        if candle_l <= eff_sl:
            return "STOP_LOSS"

        # 3. ST_EXIT — SuperTrend flipped bearish AND trade is in meaningful profit
        #    Only fires above ORB_SUPERTREND_MIN_GAIN_R to avoid 2-min noise
        #    exiting flat/early trades. TIME_EXIT handles those instead.
        if gain_R >= ORB_SUPERTREND_MIN_GAIN_R:
            st_bull = row.get(ST_BULL_COL)
            if st_bull is not None and not pd.isna(st_bull) and not bool(st_bull):
                st_lvl = row.get(ST_LINE_COL, float("nan"))
                logger.info(
                    f"BUY ST_EXIT: gain={gain_R:.2f}R ≥ {ORB_SUPERTREND_MIN_GAIN_R}R, "
                    f"ST turned bearish (line={st_lvl:.2f}) → exit at close {close:.2f}"
                )
                return "ST_EXIT"

        # 4. TIME_EXIT — T+60 and trade hasn't earned 0.5R (cut slow/stalling trades)
        if mins >= ORB_V3_TIME_EXIT_MINS and gain_R < ORB_V3_TIME_EXIT_MIN_GAIN_R:
            logger.info(
                f"BUY T+{mins:.0f}m TIME_EXIT: gain={gain_R:.2f}R < "
                f"{ORB_V3_TIME_EXIT_MIN_GAIN_R}R → exit at close {close:.2f}"
            )
            return "TIME_EXIT"

        # 5. ORB_FAILED — close pulled back inside the ORB range
        if breakout_level and close < breakout_level * (1 - ORB_FAILED_BUFFER_PCT):
            return "ORB_FAILED"

    else:  # SELL
        # Current gain in R-units for SHORT (positive = price moved down)
        gain_R = (entry_price - close) / initial_risk if initial_risk > 0 else 0.0

        # Start from original SL; only ever move DOWN (more protective for shorts)
        eff_sl = original_sl

        # --- Profit ratchet ---
        if initial_risk > 0:
            if gain_R >= ORB_V3_RATCHET_BE_R:
                # 0.5R gain → breakeven
                eff_sl = min(eff_sl, entry_price)
                logger.debug(f"SELL ratchet BE: gain={gain_R:.2f}R → eff_sl={eff_sl:.2f}")

            if gain_R >= ORB_V3_RATCHET_LOCK_R:
                # 1.0R gain → lock 40%
                locked = entry_price - ORB_V3_TRAIL_LOCK_FRACTION * initial_risk
                eff_sl = min(eff_sl, locked)
                logger.debug(f"SELL ratchet lock: gain={gain_R:.2f}R → eff_sl={eff_sl:.2f}")

        # --- Time checkpoint 1 (T+20) ---
        if mins >= ORB_V3_CHECKPOINT_1_MINS and gain_R < ORB_V3_T20_MIN_GAIN_R:
            tight = entry_price + ORB_V3_T20_TIGHT_SL_FACTOR * initial_risk
            eff_sl = min(eff_sl, tight)
            logger.info(
                f"SELL T+{mins:.0f}m CP1: gain={gain_R:.2f}R < {ORB_V3_T20_MIN_GAIN_R}R "
                f"→ tight eff_sl={eff_sl:.2f}"
            )

        # --- Time checkpoint 2 (T+40) ---
        if mins >= ORB_V3_CHECKPOINT_2_MINS and gain_R < ORB_V3_T40_MIN_GAIN_R:
            eff_sl = min(eff_sl, entry_price)
            logger.info(
                f"SELL T+{mins:.0f}m CP2: gain={gain_R:.2f}R < {ORB_V3_T40_MIN_GAIN_R}R "
                f"→ breakeven eff_sl={eff_sl:.2f}"
            )

        # Store effective SL for backtest
        position["_v3_effective_sl"] = eff_sl

        # --- Exit checks (priority order) ---
        # 1. TARGET — intrabar low touched target
        if candle_l <= target:
            return "TARGET"

        # 2. STOP_LOSS — intrabar high hit effective SL (time-decay / ratchet)
        if candle_h >= eff_sl:
            return "STOP_LOSS"

        # 3. ST_EXIT — SuperTrend flipped bullish AND trade is in meaningful profit
        #    Only fires above ORB_SUPERTREND_MIN_GAIN_R to avoid 2-min noise
        #    exiting flat/early trades. TIME_EXIT handles those instead.
        if gain_R >= ORB_SUPERTREND_MIN_GAIN_R:
            st_bull = row.get(ST_BULL_COL)
            if st_bull is not None and not pd.isna(st_bull) and bool(st_bull):
                st_lvl = row.get(ST_LINE_COL, float("nan"))
                logger.info(
                    f"SELL ST_EXIT: gain={gain_R:.2f}R ≥ {ORB_SUPERTREND_MIN_GAIN_R}R, "
                    f"ST turned bullish (line={st_lvl:.2f}) → exit at close {close:.2f}"
                )
                return "ST_EXIT"

        # 4. TIME_EXIT — T+60 and trade hasn't earned 0.5R (cut slow/stalling trades)
        if mins >= ORB_V3_TIME_EXIT_MINS and gain_R < ORB_V3_TIME_EXIT_MIN_GAIN_R:
            logger.info(
                f"SELL T+{mins:.0f}m TIME_EXIT: gain={gain_R:.2f}R < "
                f"{ORB_V3_TIME_EXIT_MIN_GAIN_R}R → exit at close {close:.2f}"
            )
            return "TIME_EXIT"

        # 5. ORB_FAILED — close pulled back inside the ORB range
        if breakout_level and close > breakout_level * (1 + ORB_FAILED_BUFFER_PCT):
            return "ORB_FAILED"

    return None
