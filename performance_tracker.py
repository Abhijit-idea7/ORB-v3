"""
performance_tracker.py
----------------------
Records every closed trade, computes daily P&L, and appends to
performance_log_ORB.csv which GitHub Actions commits back to the repo.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytz

logger   = logging.getLogger(__name__)
IST      = pytz.timezone("Asia/Kolkata")
_DEFAULT_LOG_FILE = Path("performance_log_ORB.csv")

CSV_FIELDS = [
    "date", "symbol", "window",
    "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "quantity",
    "pnl_inr", "exit_reason",
]


@dataclass
class TradeRecord:
    date:        str
    symbol:      str
    window:      str    # "15m" or "30m" — which ORB window fired
    direction:   str
    entry_time:  str
    exit_time:   str
    entry_price: float
    exit_price:  float
    quantity:    int
    pnl_inr:     float
    exit_reason: str


class PerformanceTracker:
    def __init__(self, log_file: str | None = None) -> None:
        self.trades: list[TradeRecord] = []
        self._log_file = Path(log_file) if log_file else _DEFAULT_LOG_FILE

    def record_trade(
        self,
        symbol:      str,
        direction:   str,
        entry_price: float,
        exit_price:  float,
        quantity:    int,
        entry_time:  str,
        exit_reason: str,
        window:      str = "15m",
    ) -> TradeRecord:
        now      = datetime.now(IST)
        exit_time = now.strftime("%H:%M")
        pnl = (
            (exit_price - entry_price) * quantity
            if direction == "BUY"
            else (entry_price - exit_price) * quantity
        )
        pnl = round(pnl, 2)

        record = TradeRecord(
            date        = now.strftime("%Y-%m-%d"),
            symbol      = symbol,
            window      = window,
            direction   = direction,
            entry_time  = entry_time,
            exit_time   = exit_time,
            entry_price = entry_price,
            exit_price  = exit_price,
            quantity    = quantity,
            pnl_inr     = pnl,
            exit_reason = exit_reason,
        )
        self.trades.append(record)

        sign = "+" if pnl >= 0 else "-"
        logger.info(
            f"[PERF] [{sign}] ORB/{window} | {direction} {symbol} | "
            f"entry={entry_price:.2f} exit={exit_price:.2f} qty={quantity} | "
            f"P&L=Rs{pnl:+,.0f} | reason={exit_reason}"
        )
        return record

    def daily_summary(self) -> None:
        sep = "=" * 65
        logger.info(f"[PERF] {sep}")

        if not self.trades:
            logger.info("[PERF] No trades executed today.")
            logger.info(f"[PERF] {sep}")
            return

        total      = len(self.trades)
        profitable = [t for t in self.trades if t.pnl_inr > 0]
        gross_pnl  = sum(t.pnl_inr for t in self.trades)
        brokerage  = total * 40   # Rs20/order × 2 legs
        net_pnl    = gross_pnl - brokerage
        win_rate   = len(profitable) / total * 100
        best       = max(self.trades, key=lambda t: t.pnl_inr)
        worst      = min(self.trades, key=lambda t: t.pnl_inr)

        by_window: dict[str, list] = {}
        for t in self.trades:
            by_window.setdefault(t.window, []).append(t)

        by_reason: dict[str, int] = {}
        for t in self.trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

        lines = [
            f"ORB DAILY PERFORMANCE — {self.trades[0].date}",
            sep,
            f"  Total trades : {total}",
            f"  Winners      : {len(profitable)}  ({win_rate:.1f}% win rate)",
            f"  Losers       : {total - len(profitable)}",
            f"  Exit reasons : {by_reason}",
            sep,
        ]

        for win_label, win_trades in sorted(by_window.items()):
            w_pnl  = sum(t.pnl_inr for t in win_trades)
            w_wins = sum(1 for t in win_trades if t.pnl_inr > 0)
            w_wr   = w_wins / len(win_trades) * 100
            lines.append(
                f"  [ORB-{win_label}] trades={len(win_trades)} "
                f"wins={w_wins} ({w_wr:.0f}%) gross=Rs{w_pnl:+,.0f}"
            )

        lines += [
            sep,
            f"  Gross P&L         : Rs{gross_pnl:+,.0f}",
            f"  Brokerage (est.)  : -Rs{brokerage:,.0f}",
            f"  Net P&L (est.)    : Rs{net_pnl:+,.0f}",
            sep,
            f"  Best  : {best.symbol} ({best.window}) {best.direction} "
            f"Rs{best.pnl_inr:+,.0f} ({best.exit_reason})",
            f"  Worst : {worst.symbol} ({worst.window}) {worst.direction} "
            f"Rs{worst.pnl_inr:+,.0f} ({worst.exit_reason})",
            sep,
            "  TRADE BREAKDOWN:",
        ]

        for t in self.trades:
            sign = "+" if t.pnl_inr >= 0 else "-"
            lines.append(
                f"    [{sign}] {t.symbol:12s} [{t.window}] {t.direction:4s} "
                f"{t.entry_time}->{t.exit_time}  "
                f"Rs{t.entry_price:.2f}->Rs{t.exit_price:.2f}  "
                f"qty={t.quantity}  P&L=Rs{t.pnl_inr:+,.0f}  [{t.exit_reason}]"
            )
        lines.append(sep)

        for line in lines:
            logger.info(f"[PERF] {line}")

    def save_to_csv(self) -> None:
        if not self.trades:
            logger.info("[PERF] No trades to save.")
            return

        file_exists = self._log_file.exists()
        with open(self._log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            for t in self.trades:
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

        logger.info(f"[PERF] {len(self.trades)} trade(s) saved to {self._log_file}")
