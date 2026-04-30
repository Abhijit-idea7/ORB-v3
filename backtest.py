#!/usr/bin/env python3
"""
backtest.py
-----------
Offline backtest of the ORB v3 strategy over the last N trading days.

Usage:
    python backtest.py                  # 30 days, regime filter OFF
    python backtest.py --days 45
    python backtest.py --days 30 --regime   # enable NIFTY regime filter

How it works:
  1. Fetches 59 days of 2-min candles from yfinance for all ORB_STOCK_UNIVERSE stocks
     (maximum available from Yahoo Finance — gives EMA/RSI full warmup history)
  2. Computes all indicators once on the full dataset per symbol
  3. Simulates the live-bot loop day by day, candle by candle:
       - Ranks stocks daily by ATR% (same as live bot)
       - Checks 15-min ORB (primary window)
       - Respects ORB_MAX_POSITIONS simultaneous open positions
       - Hard square-off at 15:15 IST
  4. Prints per-day P&L table with exit reason breakdown
  5. Saves all trades to backtest_results.csv

v3 specifics:
  - TIME_EXIT: position evicted at close if T+60 and gain < 0.5R
  - STOP_LOSS exit price uses effective SL (tightened by ratchet / time checkpoints),
    not the original ORB SL — accurate P&L calculation is critical here.

Note on yfinance data:
  Yahoo Finance provides up to ~59 days of 2-min candles.
  We always fetch the full 59d window regardless of --days so EMA/RSI are warmed up.
"""

import argparse
import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from config import (
    DAILY_LOSS_CIRCUIT_BREAKER,
    ONE_TRADE_PER_STOCK_PER_DAY,
    ORB_MAX_POSITIONS,
    ORB_STOCK_UNIVERSE,
    ORB_TOP_N_STOCKS,
    POSITION_SIZE_INR,
    REGIME_BEAR_THRESHOLD,
    REGIME_BULL_THRESHOLD,
)
from indicators import add_indicators
from strategy_orb import check_exit_signal, generate_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

IST          = pytz.timezone("Asia/Kolkata")
TRADE_START  = time(9, 20)
SQUARE_OFF   = time(15, 15)
BROKERAGE    = 40         # Rs20 × 2 legs per trade (Zerodha intraday)
OUTPUT_CSV   = Path("backtest_results.csv")
NIFTY_TICKER = "^NSEI"

CSV_FIELDS = [
    "date", "symbol", "window",
    "direction", "entry_time", "exit_time",
    "entry_price", "exit_price",
    "quantity", "pnl_inr", "exit_reason",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BtPosition:
    symbol:             str
    direction:          str
    entry_price:        float
    sl:                 float
    target:             float
    quantity:           int
    entry_time:         str
    strategy_name:      str = "ORB"
    orb_breakout_level: float = 0.0
    window:             str = "15m"


@dataclass
class BtTrade:
    date:        str
    symbol:      str
    window:      str
    direction:   str
    entry_time:  str
    exit_time:   str
    entry_price: float
    exit_price:  float
    quantity:    int
    pnl_inr:     float
    exit_reason: str


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def _ns(symbol: str) -> str:
    return f"{symbol}.NS"


def fetch_with_indicators(symbol: str) -> pd.DataFrame | None:
    """
    Fetch ~59 days of 2-min candles, compute all indicators,
    return a timezone-aware (IST) DataFrame or None on failure.
    """
    try:
        df = yf.Ticker(_ns(symbol)).history(interval="2m", period="59d")
        if df is None or df.empty:
            logger.warning(f"{symbol}: no data from yfinance")
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        if len(df) < 30:
            logger.warning(f"{symbol}: only {len(df)} candles — too few for warmup")
            return None

        df = add_indicators(df)
        logger.info(f"{symbol}: {len(df)} candles with indicators")
        return df

    except Exception as e:
        logger.error(f"{symbol}: fetch error — {e}")
        return None


def rank_by_atr(symbol_dfs: dict, date_: datetime.date, top_n: int = ORB_TOP_N_STOCKS) -> list[str]:
    """Rank symbols by ATR% using daily data prior to the backtest date."""
    scores: dict[str, float] = {}
    for symbol, df in symbol_dfs.items():
        try:
            hist = df[df.index.date < date_]
            if hist.empty:
                continue
            daily      = hist["Close"].resample("1D").last().dropna()
            daily_high = hist["High"].resample("1D").max().dropna()
            daily_low  = hist["Low"].resample("1D").min().dropna()
            daily, daily_high = daily.align(daily_high, join="inner")
            daily, daily_low  = daily.align(daily_low,  join="inner")
            if len(daily) < 3:
                continue
            prev_close = daily.shift(1)
            tr = pd.concat([
                daily_high - daily_low,
                (daily_high - prev_close).abs(),
                (daily_low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr     = tr.iloc[-10:].mean()
            atr_pct = atr / daily.iloc[-1] if daily.iloc[-1] > 0 else 0
            scores[symbol] = atr_pct
        except Exception:
            pass

    if not scores:
        return list(symbol_dfs.keys())[:top_n]
    return sorted(scores, key=lambda s: scores[s], reverse=True)[:top_n]


def calculate_quantity(price: float, scale: float = 1.0) -> int:
    return int(POSITION_SIZE_INR * scale // price) if price > 0 else 0


# ---------------------------------------------------------------------------
# NIFTY Regime helpers
# ---------------------------------------------------------------------------
def fetch_nifty_with_indicators() -> pd.DataFrame | None:
    try:
        df = yf.Ticker(NIFTY_TICKER).history(interval="2m", period="59d")
        if df is None or df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
        if len(df) < 30:
            return None
        df = add_indicators(df)
        logger.info(f"NIFTY: {len(df)} candles fetched with indicators")
        return df
    except Exception as e:
        logger.error(f"NIFTY fetch error: {e}")
        return None


def compute_day_regime(nifty_df: pd.DataFrame, date_) -> dict:
    """
    Classify the NIFTY regime for a given backtest date using data up to 10:00 IST.

    Using 10:00 cutoff simulates what the live bot sees before most ORB entries
    (primary cutoff 11:00, secondary 11:30). Avoids look-ahead bias.

    Scoring mirrors market_regime.py exactly:
      VWAP position (0.40) + EMA 9 vs 50 (0.40) + Day change (0.20)
    """
    _neutral = {"regime": "NEUTRAL", "score": 0.0, "direction_filter": "BOTH"}

    try:
        day_df = nifty_df[nifty_df.index.date == date_]
        if day_df.empty:
            return _neutral

        cutoff_time = time(10, 0)
        day_df = day_df[day_df.index.time <= cutoff_time]
        if len(day_df) < 3:
            return _neutral

        row   = day_df.iloc[-1]
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
            return _neutral

        total_w = sum(weights)
        score   = float(np.clip(
            sum(c * w / total_w for c, w in zip(components, weights)),
            -1.0, 1.0
        ))

        if score > REGIME_BULL_THRESHOLD:
            return {"regime": "BULL",    "score": score, "direction_filter": "LONG_ONLY"}
        elif score < REGIME_BEAR_THRESHOLD:
            return {"regime": "BEAR",    "score": score, "direction_filter": "SHORT_ONLY"}
        else:
            return {"regime": "NEUTRAL", "score": score, "direction_filter": "BOTH"}

    except Exception as e:
        logger.warning(f"compute_day_regime({date_}) error: {e} — using NEUTRAL")
        return _neutral


# ---------------------------------------------------------------------------
# Single-day simulation
# ---------------------------------------------------------------------------
def simulate_day(
    date_:            datetime.date,
    candidates:       list[str],
    symbol_dfs:       dict,
    max_positions:    int = ORB_MAX_POSITIONS,
    direction_filter: str = "BOTH",
) -> list[BtTrade]:
    """
    Simulate a full trading day candle-by-candle.
    Returns a list of BtTrade records for all trades closed that day.
    """
    trades:               list[BtTrade]           = []
    open_positions:       dict[str, BtPosition]   = {}
    traded_today:         set[str]                = set()
    daily_realized_pnl:   float                   = 0.0

    # Build per-symbol DataFrames for this day
    day_data: dict[str, pd.DataFrame] = {}
    for sym in candidates:
        if sym not in symbol_dfs:
            continue
        df     = symbol_dfs[sym]
        day_df = df[df.index.date == date_].copy()
        if len(day_df) >= 3:
            day_data[sym] = day_df

    if not day_data:
        return trades

    all_times = sorted({ts for df in day_data.values() for ts in df.index})

    def _close_position(symbol: str, exit_px: float, reason: str, ts_str: str):
        nonlocal daily_realized_pnl
        pos = open_positions.pop(symbol)
        traded_today.add(symbol)
        pnl = (
            (exit_px - pos.entry_price) * pos.quantity
            if pos.direction == "BUY"
            else (pos.entry_price - exit_px) * pos.quantity
        )
        daily_realized_pnl += pnl
        trades.append(BtTrade(
            date        = date_.isoformat(),
            symbol      = symbol,
            window      = pos.window,
            direction   = pos.direction,
            entry_time  = pos.entry_time,
            exit_time   = ts_str,
            entry_price = pos.entry_price,
            exit_price  = exit_px,
            quantity    = pos.quantity,
            pnl_inr     = round(pnl, 2),
            exit_reason = reason,
        ))

    for ts in all_times:
        ts_time = ts.time()
        ts_str  = ts.strftime("%H:%M")

        # Hard square-off gate
        if ts_time >= SQUARE_OFF:
            for symbol in list(open_positions.keys()):
                df = day_data.get(symbol)
                px = (
                    float(df.loc[ts, "Close"])
                    if df is not None and ts in df.index
                    else open_positions[symbol].entry_price
                )
                _close_position(symbol, px, "SQUARE_OFF", ts_str)
            break

        if ts_time < TRADE_START:
            continue

        # --- Exit checks (strictly closed candles only: df.index < ts) ---
        for symbol in list(open_positions.keys()):
            df = day_data.get(symbol)
            if df is None:
                continue
            pos      = open_positions[symbol]
            df_slice = df[df.index < ts]
            if len(df_slice) < 2:
                continue

            reason = check_exit_signal(df_slice, pos.__dict__)
            if reason:
                sig_candle = df_slice.iloc[-1]
                if reason == "TARGET":
                    exit_px = float(pos.target)
                elif reason == "STOP_LOSS":
                    # v3: use effective SL (tightened by ratchet/time checkpoints).
                    # check_exit_signal stores it in pos.__dict__["_v3_effective_sl"].
                    # Falls back to original pos.sl if key absent (safety).
                    exit_px = pos.__dict__.get("_v3_effective_sl", float(pos.sl))
                else:
                    # TIME_EXIT, ORB_FAILED, SQUARE_OFF — exit at candle close
                    exit_px = float(sig_candle["Close"])
                _close_position(symbol, exit_px, reason, ts_str)

        # --- Entry checks ---
        circuit_tripped = daily_realized_pnl <= DAILY_LOSS_CIRCUIT_BREAKER
        if len(open_positions) < max_positions and not circuit_tripped:
            for symbol in candidates:
                if len(open_positions) >= max_positions:
                    break
                if symbol in open_positions:
                    continue
                if ONE_TRADE_PER_STOCK_PER_DAY and symbol in traded_today:
                    continue

                df = day_data.get(symbol)
                if df is None:
                    continue
                df_slice = df[df.index < ts]
                if len(df_slice) < 3:
                    continue

                # Pass sim_time=ts so the cutoff gate in generate_signal()
                # uses the backtest candle time, not datetime.now().
                signal = generate_signal(df_slice, symbol=symbol, sim_time=ts)
                if signal["action"] not in ("BUY", "SELL"):
                    continue

                if direction_filter == "LONG_ONLY" and signal["action"] == "SELL":
                    continue
                if direction_filter == "SHORT_ONLY" and signal["action"] == "BUY":
                    continue

                entry_price = float(df_slice.iloc[-1]["Close"])
                quantity    = calculate_quantity(entry_price, scale=signal.get("quantity_scale", 1.0))
                if quantity < 1:
                    continue

                open_positions[symbol] = BtPosition(
                    symbol             = symbol,
                    direction          = signal["action"],
                    entry_price        = entry_price,
                    sl                 = signal["sl"],
                    target             = signal["target"],
                    quantity           = quantity,
                    entry_time         = ts_str,
                    strategy_name      = "ORB",
                    orb_breakout_level = signal.get("orb_breakout_level", 0.0),
                    window             = signal.get("window", "15m"),
                )
                break   # one position per symbol per loop tick

    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_overall_summary(
    all_trades:  list[BtTrade],
    days_tested: int,
    use_regime:  bool,
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BACKTEST SUMMARY — ORB v3 (Time-Decay Stops | 15-min ORB)")
    print(sep)

    if not all_trades:
        print("  No trades generated in the backtest period.")
        print(sep)
        return

    total     = len(all_trades)
    gross     = sum(t.pnl_inr for t in all_trades)
    brokerage = total * BROKERAGE
    net       = gross - brokerage
    wins      = [t for t in all_trades if t.pnl_inr > 0]
    win_rate  = len(wins) / total * 100
    avg_day   = net / days_tested if days_tested else 0

    by_reason: dict[str, int] = {}
    for t in all_trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    by_window: dict[str, list] = {}
    for t in all_trades:
        by_window.setdefault(t.window, []).append(t)

    best  = max(all_trades, key=lambda t: t.pnl_inr)
    worst = min(all_trades, key=lambda t: t.pnl_inr)

    print(f"  Backtest period  : {days_tested} trading days")
    print(f"  Universe         : {len(ORB_STOCK_UNIVERSE)} stocks (top {ORB_TOP_N_STOCKS}/day by ATR%)")
    print(f"  Capital per trade: Rs{POSITION_SIZE_INR:,.0f} × scale (max {ORB_MAX_POSITIONS} simultaneous)")
    print(f"  Regime filter    : {'ON' if use_regime else 'OFF'}")
    print(sep)
    print(f"  Total trades     : {total}  ({total/days_tested:.1f}/day avg)")
    print(f"  Win rate         : {len(wins)}/{total} = {win_rate:.1f}%")
    print(f"  Exit breakdown   : {by_reason}")
    print(sep)

    # Per-window breakdown
    for win_label in sorted(by_window.keys()):
        wt      = by_window[win_label]
        w_wins  = sum(1 for t in wt if t.pnl_inr > 0)
        w_wr    = w_wins / len(wt) * 100
        w_gross = sum(t.pnl_inr for t in wt)
        w_net   = w_gross - BROKERAGE * len(wt)
        print(
            f"  [ORB-{win_label}] trades={len(wt)} wins={w_wins} ({w_wr:.0f}%) "
            f"gross=Rs{w_gross:+,.0f}  net=Rs{w_net:+,.0f}"
        )

    print(sep)
    print(f"  Gross P&L        : Rs{gross:+,.0f}")
    print(f"  Brokerage (est.) : -Rs{brokerage:,.0f}  (Rs{BROKERAGE}/trade × {total})")
    print(f"  Net P&L (est.)   : Rs{net:+,.0f}")
    print(f"  Avg net/day      : Rs{avg_day:+,.0f}")
    print(sep)
    print(f"  Best  : {best.symbol} [{best.window}] {best.direction} {best.date} Rs{best.pnl_inr:+,.0f} [{best.exit_reason}]")
    print(f"  Worst : {worst.symbol} [{worst.window}] {worst.direction} {worst.date} Rs{worst.pnl_inr:+,.0f} [{worst.exit_reason}]")
    print(sep)


def save_to_csv(all_trades: list[BtTrade]) -> None:
    if not all_trades:
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in all_trades:
            writer.writerow({
                "date":        t.date,
                "symbol":      t.symbol,
                "window":      t.window,
                "direction":   t.direction,
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "pnl_inr":     t.pnl_inr,
                "exit_reason": t.exit_reason,
            })
    logger.info(f"Saved {len(all_trades)} trades to {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(days: int, use_regime: bool = False) -> None:
    print(f"\n{'=' * 70}")
    print(f"  ORB v3 BACKTEST — last {days} trading days")
    print(f"  Universe : {len(ORB_STOCK_UNIVERSE)} stocks → top {ORB_TOP_N_STOCKS}/day by ATR%")
    print(f"  Window   : 15-min ORB (primary only)")
    print(f"  Exits    : TARGET | STOP_LOSS (ratchet) | TIME_EXIT (T+60) | ORB_FAILED")
    print(f"  Regime   : {'NIFTY filter ON' if use_regime else 'OFF (use --regime to enable)'}")
    print(f"{'=' * 70}\n")

    # 1. Fetch data + indicators
    logger.info("Fetching 59-day 2-min data (this may take ~2 minutes for 55 stocks)...")
    symbol_dfs: dict[str, pd.DataFrame] = {}
    for symbol in ORB_STOCK_UNIVERSE:
        df = fetch_with_indicators(symbol)
        if df is not None:
            symbol_dfs[symbol] = df

    if not symbol_dfs:
        logger.error("No data fetched. Exiting.")
        return

    # 2. Determine trading days
    all_dates      = sorted({d for df in symbol_dfs.values() for d in df.index.date})
    backtest_dates = all_dates[-days:]
    logger.info(f"Testing {len(backtest_dates)} days: {backtest_dates[0]} — {backtest_dates[-1]}")

    # 2b. NIFTY regime filter (optional, --regime flag)
    day_regimes: dict = {}
    if use_regime:
        logger.info("Fetching NIFTY50 data for daily regime classification...")
        nifty_df = fetch_nifty_with_indicators()
        if nifty_df is not None:
            for d in backtest_dates:
                day_regimes[d] = compute_day_regime(nifty_df, d)
            counts: dict[str, int] = {}
            for r in day_regimes.values():
                lbl = r.get("regime", "NEUTRAL")
                counts[lbl] = counts.get(lbl, 0) + 1
            logger.info(f"Regime classification: {counts}")
            print(f"\n  Regime at 10:00 IST: {counts}")
            print(f"  BULL → LONG_ONLY  |  BEAR → SHORT_ONLY  |  NEUTRAL → BOTH\n")
        else:
            logger.warning("NIFTY fetch failed — proceeding without regime filter")
            use_regime = False

    # 3. Simulate day by day
    all_trades: list[BtTrade] = []

    if use_regime:
        hdr = f"  {'Date':12s}  {'Rgm':>4}  {'15m':>4}  {'30m':>4}  {'Trades':>6}  {'Win%':>5}  {'Gross':>12}  {'Net':>12}"
    else:
        hdr = f"  {'Date':12s}  {'15m':>4}  {'30m':>4}  {'Trades':>6}  {'Win%':>5}  {'Gross':>12}  {'Net':>12}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for date_ in backtest_dates:
        regime_info  = day_regimes.get(date_, {"regime": "BOTH", "direction_filter": "BOTH", "score": 0.0})
        dir_filter   = regime_info.get("direction_filter", "BOTH")
        regime_label = regime_info.get("regime", "BOTH")[:4]

        effective_max = ORB_MAX_POSITIONS
        if use_regime:
            from config import REGIME_BEAR_MAX_POSITIONS, REGIME_BULL_MAX_POSITIONS, REGIME_NEUTRAL_MAX_POSITIONS
            regime_label_full = regime_info.get("regime", "NEUTRAL")
            if regime_label_full == "BULL":
                effective_max = REGIME_BULL_MAX_POSITIONS
            elif regime_label_full == "BEAR":
                effective_max = REGIME_BEAR_MAX_POSITIONS
            elif regime_label_full == "NEUTRAL":
                effective_max = REGIME_NEUTRAL_MAX_POSITIONS

        candidates = rank_by_atr(symbol_dfs, date_, top_n=ORB_TOP_N_STOCKS)
        day_trades  = simulate_day(
            date_, candidates, symbol_dfs,
            max_positions=effective_max,
            direction_filter=dir_filter,
        )
        all_trades.extend(day_trades)

        if day_trades:
            g     = sum(t.pnl_inr for t in day_trades)
            n     = g - BROKERAGE * len(day_trades)
            wins  = sum(1 for t in day_trades if t.pnl_inr > 0)
            wr    = wins / len(day_trades) * 100
            c15   = sum(1 for t in day_trades if t.window == "15m")
            c30   = sum(1 for t in day_trades if t.window == "30m")
            if use_regime:
                print(
                    f"  {str(date_):12s}  {regime_label:>4s}  {c15:>4d}  {c30:>4d}  "
                    f"{len(day_trades):>6d}  {wr:>4.0f}%  Rs{g:>+9,.0f}  Rs{n:>+9,.0f}"
                )
            else:
                print(
                    f"  {str(date_):12s}  {c15:>4d}  {c30:>4d}  "
                    f"{len(day_trades):>6d}  {wr:>4.0f}%  Rs{g:>+9,.0f}  Rs{n:>+9,.0f}"
                )
        else:
            if use_regime:
                print(f"  {str(date_):12s}  {regime_label:>4s}  {'—':>4}  {'—':>4}  {'—':>6}  {'—':>5}  {'Rs0':>12}  {'Rs0':>12}")
            else:
                print(f"  {str(date_):12s}  {'—':>4}  {'—':>4}  {'—':>6}  {'—':>5}  {'Rs0':>12}  {'Rs0':>12}")

    # 4. Summary + CSV
    print_overall_summary(all_trades, len(backtest_dates), use_regime)
    save_to_csv(all_trades)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest ORB v3 time-decay stop strategy")
    parser.add_argument(
        "--days", type=int, default=30, choices=range(1, 60), metavar="N",
        help="Trading days to backtest (1–59, default: 30)",
    )
    parser.add_argument(
        "--regime", action="store_true",
        help="Enable NIFTY50 regime filter (LONG_ONLY on BULL days, SHORT_ONLY on BEAR days)",
    )
    args = parser.parse_args()
    run(args.days, use_regime=args.regime)
