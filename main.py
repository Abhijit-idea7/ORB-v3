"""
main.py
-------
ORB v3 — Opening Range Breakout with Time-Decay Stop Management.

15-min ORB window (primary only). Positions are progressively expelled if
the market doesn't prove the trade right within 20 / 40 / 60 minutes.

Exit rules (in addition to standard TARGET / STOP_LOSS / ORB_FAILED):
  T+20: gain < 0R  → SL tightened to entry − 30% of initial risk
  T+40: gain < 0.3R → SL forced to breakeven
  T+60: gain < 0.5R → TIME_EXIT (exit at market regardless of SL)
  Ratchet: ≥ 0.5R → breakeven | ≥ 1.0R → locks 40% gain

Lifecycle (GitHub Actions):
  1. Runner starts at 09:10 IST (manual workflow_dispatch trigger)
  2. Bot waits until TRADE_START_TIME (09:20 IST)
  3. Selects top 25 candidates by ATR% from 55-stock universe
  4. Loop every 2 minutes:
       a. Fetch NIFTY50 regime (direction filter + position cap)
       b. Check exits for all open positions (v3 time-decay logic)
       c. Scan candidates for new entry signals (15-min ORB)
  5. At 15:10 IST: force-close all open positions (SQUARE_OFF)
  6. Print daily P&L summary and save to performance_log_ORB.csv
  7. GitHub Actions commits the CSV back to the repo
"""

import logging
import time
from datetime import datetime

import pytz

from config import (
    LOOP_SLEEP_SECONDS,
    ONE_TRADE_PER_STOCK_PER_DAY,
    ORB_MAX_POSITIONS,
    ORB_STOCK_UNIVERSE,
    ORB_TOP_N_STOCKS,
    SQUARE_OFF_TIME,
    TRADE_START_TIME,
)
from data_feed import fetch_candles_for_warmup, get_top_candidates
from indicators import add_indicators
from order_manager import calculate_quantity, place_order, square_off
from performance_tracker import PerformanceTracker
from strategy_orb import check_exit_signal, generate_signal
from trade_tracker import TradeTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

IST         = pytz.timezone("Asia/Kolkata")
MIN_CANDLES = 20


def ist_now() -> datetime:
    return datetime.now(IST)


def current_time_str() -> str:
    return ist_now().strftime("%H:%M")


def is_past(hhmm: str) -> bool:
    now  = ist_now()
    h, m = map(int, hhmm.split(":"))
    return now >= now.replace(hour=h, minute=m, second=0, microsecond=0)


def fetch_and_prepare(symbol: str):
    """
    Fetch 5 days of 2-min candles for EMA/RSI warmup, compute all indicators,
    return today's candles only for signal generation.
    """
    df_full = fetch_candles_for_warmup(symbol, period="5d")
    if df_full is None or len(df_full) < MIN_CANDLES:
        logger.info(
            f"{symbol}: only {len(df_full) if df_full is not None else 0} candles "
            f"— need {MIN_CANDLES}, skipping."
        )
        return None
    try:
        df_ind   = add_indicators(df_full)
        today    = df_ind.index[-1].date()
        df_today = df_ind[df_ind.index.date == today]
        if len(df_today) < 3:
            logger.info(f"{symbol}: only {len(df_today)} candles today — waiting.")
            return None
        return df_today
    except Exception as e:
        logger.warning(f"{symbol}: indicator calculation failed — {e}")
        return None


def check_exits(tracker: TradeTracker, perf: PerformanceTracker, closed_today: set) -> None:
    for position in tracker.all_positions():
        symbol = position.symbol
        try:
            df = fetch_and_prepare(symbol)
            if df is None:
                continue

            reason = check_exit_signal(df, position.__dict__)
            if reason:
                exit_price = float(df["Close"].iloc[-2])
                ok = square_off(symbol, position.direction, position.quantity)
                if ok:
                    tracker.record_closed_pnl(
                        position.entry_price, exit_price,
                        position.quantity, position.direction,
                    )
                    tracker.remove_position(symbol)
                    closed_today.add(symbol)
                    perf.record_trade(
                        symbol      = symbol,
                        direction   = position.direction,
                        entry_price = position.entry_price,
                        exit_price  = exit_price,
                        quantity    = position.quantity,
                        entry_time  = position.entry_time,
                        exit_reason = reason,
                        window      = position.window,
                    )
        except Exception as e:
            logger.error(f"Error checking exit for {symbol}: {e}")


def square_off_all(tracker: TradeTracker, perf: PerformanceTracker, closed_today: set) -> None:
    logger.info("=== SQUARE-OFF TIME: closing all open positions ===")
    for position in tracker.all_positions():
        try:
            df         = fetch_and_prepare(position.symbol)
            exit_price = (
                float(df["Close"].iloc[-2]) if df is not None else position.entry_price
            )
            ok = False
            for attempt in range(1, 4):
                ok = square_off(position.symbol, position.direction, position.quantity)
                if ok:
                    break
                logger.warning(
                    f"{position.symbol}: square_off attempt {attempt}/3 failed"
                    + (", retrying..." if attempt < 3 else " — giving up.")
                )
                if attempt < 3:
                    time.sleep(5)

            reason = "SQUARE_OFF" if ok else "SQUARE_OFF_FAILED"
            if not ok:
                logger.error(
                    f"{position.symbol}: all square_off attempts failed. "
                    f"Zerodha MIS auto-close at ~15:20 will handle it. "
                    f"Recording at last known price Rs{exit_price:.2f}."
                )

            tracker.record_closed_pnl(
                position.entry_price, exit_price,
                position.quantity, position.direction,
            )
            tracker.remove_position(position.symbol)
            closed_today.add(position.symbol)
            perf.record_trade(
                symbol      = position.symbol,
                direction   = position.direction,
                entry_price = position.entry_price,
                exit_price  = exit_price,
                quantity    = position.quantity,
                entry_time  = position.entry_time,
                exit_reason = reason,
                window      = position.window,
            )
        except Exception as e:
            logger.error(f"Error squaring off {position.symbol}: {e}")
    logger.info("Square-off complete.")


def scan_for_entries(
    candidates:   list[str],
    tracker:      TradeTracker,
    closed_today: set,
    regime:       dict,
) -> None:
    effective_max    = regime.get("max_positions", ORB_MAX_POSITIONS)
    direction_filter = regime.get("direction_filter", "BOTH")

    for symbol in candidates:
        if tracker.open_count() >= effective_max:
            logger.info(
                f"Position cap reached ({tracker.open_count()}/{effective_max} "
                f"regime={regime.get('regime', 'N/A')}) — pausing entries."
            )
            break

        if tracker.has_position(symbol):
            continue

        if ONE_TRADE_PER_STOCK_PER_DAY and symbol in closed_today:
            continue

        try:
            df = fetch_and_prepare(symbol)
            if df is None:
                continue

            signal = generate_signal(df, symbol=symbol)

            if signal["action"] not in ("BUY", "SELL"):
                continue

            if direction_filter == "LONG_ONLY" and signal["action"] == "SELL":
                logger.info(f"{symbol}: SELL blocked — BULL regime (LONG_ONLY)")
                continue
            if direction_filter == "SHORT_ONLY" and signal["action"] == "BUY":
                logger.info(f"{symbol}: BUY blocked — BEAR regime (SHORT_ONLY)")
                continue

            entry_price    = float(df["Close"].iloc[-2])
            quantity_scale = signal.get("quantity_scale", 1.0)
            quantity       = calculate_quantity(entry_price, scale=quantity_scale)

            if quantity < 1:
                logger.warning(f"{symbol}: qty rounds to 0 at Rs{entry_price:.2f}, skipping.")
                continue

            ok = place_order(symbol, signal["action"], quantity)
            if ok:
                tracker.add_position(
                    symbol             = symbol,
                    direction          = signal["action"],
                    entry_price        = entry_price,
                    sl                 = signal["sl"],
                    target             = signal["target"],
                    quantity           = quantity,
                    orb_breakout_level = signal.get("orb_breakout_level", 0.0),
                    window             = signal.get("window", "15m"),
                )

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")


def run() -> None:
    logger.info("=" * 65)
    logger.info("ORB v3 — Time-Decay Stop Management Bot")
    logger.info(f"Universe    : {len(ORB_STOCK_UNIVERSE)} stocks → top {ORB_TOP_N_STOCKS} daily")
    logger.info(f"Max positions: {ORB_MAX_POSITIONS}")
    logger.info(f"Window      : 15-min ORB (entries →11:00 IST)")
    logger.info(f"Exits       : TARGET | SL (ratchet) | TIME_EXIT (T+60) | ORB_FAILED")
    logger.info(f"Trade window: {TRADE_START_TIME} — {SQUARE_OFF_TIME} IST")
    logger.info("=" * 65)

    while not is_past(TRADE_START_TIME):
        logger.info(f"Waiting for {TRADE_START_TIME} IST... (now {current_time_str()})")
        time.sleep(30)

    logger.info("Selecting today's top candidates by ATR%...")
    candidates = get_top_candidates(universe=ORB_STOCK_UNIVERSE, top_n=ORB_TOP_N_STOCKS)
    logger.info(f"Watchlist ({len(candidates)} stocks): {candidates}")

    tracker      = TradeTracker()
    perf         = PerformanceTracker(log_file="performance_log_ORB.csv")
    closed_today: set = set()

    while True:
        logger.info(f"--- Loop tick at {current_time_str()} IST ---")

        if is_past(SQUARE_OFF_TIME):
            square_off_all(tracker, perf, closed_today)
            break

        # Fetch NIFTY regime for direction filter and position cap
        regime = {"regime": "NEUTRAL", "max_positions": ORB_MAX_POSITIONS, "direction_filter": "BOTH"}
        try:
            from market_regime import get_nifty_regime
            regime = get_nifty_regime()
            logger.info(
                f"NIFTY: {regime['regime']} (score={regime['score']:+.3f}) "
                f"→ dir={regime['direction_filter']} max={regime['max_positions']}"
            )
        except Exception as e:
            logger.warning(f"Regime fetch failed: {e} — using NEUTRAL defaults")

        if tracker.open_count() > 0:
            check_exits(tracker, perf, closed_today)

        if tracker.can_open_new_trade():
            scan_for_entries(candidates, tracker, closed_today, regime)

        logger.info(tracker.summary())
        logger.info(f"Sleeping {LOOP_SLEEP_SECONDS}s until next candle...")
        time.sleep(LOOP_SLEEP_SECONDS)

    perf.daily_summary()
    perf.save_to_csv()
    logger.info("Bot exited cleanly.")


if __name__ == "__main__":
    run()
