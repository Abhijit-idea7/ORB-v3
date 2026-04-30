"""
trade_tracker.py
----------------
In-memory position state for the ORB trading session.

Fresh TradeTracker created each trading day. No cross-day persistence needed.
Key fix vs v1: uses ORB_MAX_POSITIONS directly in can_open_new_trade()
so the check is consistent with the scan_for_entries cap in main.py.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pytz

from config import DAILY_LOSS_CIRCUIT_BREAKER, ORB_MAX_POSITIONS

IST    = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol:              str
    direction:           str    # "BUY" or "SELL"
    entry_price:         float
    sl:                  float
    target:              float
    quantity:            int
    entry_time:          str
    strategy_name:       str = "ORB"
    orb_breakout_level:  float = 0.0   # ORB level that was broken (for ORB_FAILED check)
    window:              str = "15m"   # which ORB window fired ("15m" or "30m")
    signal_scores:       dict = field(default_factory=dict)


class TradeTracker:
    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self.daily_trades: int = 0
        self.daily_realized_pnl: float = 0.0

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def open_count(self) -> int:
        return len(self._positions)

    def can_open_new_trade(self) -> bool:
        if self.open_count() >= ORB_MAX_POSITIONS:
            return False
        if self.daily_realized_pnl <= DAILY_LOSS_CIRCUIT_BREAKER:
            logger.info(
                f"[CIRCUIT BREAKER] Daily P&L Rs{self.daily_realized_pnl:+,.0f} "
                f"≤ limit — no new entries."
            )
            return False
        return True

    def record_closed_pnl(self, entry_price: float, exit_price: float,
                          quantity: int, direction: str) -> None:
        mult = 1 if direction == "BUY" else -1
        pnl  = mult * (exit_price - entry_price) * quantity
        self.daily_realized_pnl += pnl
        logger.info(
            f"[TRACKER] Realized P&L: trade={pnl:+,.0f} "
            f"day_total={self.daily_realized_pnl:+,.0f}"
        )

    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def add_position(
        self,
        symbol:             str,
        direction:          str,
        entry_price:        float,
        sl:                 float,
        target:             float,
        quantity:           int,
        strategy_name:      str = "ORB",
        orb_breakout_level: float = 0.0,
        window:             str = "15m",
        signal_scores:      dict = None,
    ) -> None:
        entry_time = datetime.now(IST).strftime("%H:%M")
        self._positions[symbol] = Position(
            symbol              = symbol,
            direction           = direction,
            entry_price         = entry_price,
            sl                  = sl,
            target              = target,
            quantity            = quantity,
            entry_time          = entry_time,
            strategy_name       = strategy_name,
            orb_breakout_level  = orb_breakout_level,
            window              = window,
            signal_scores       = signal_scores or {},
        )
        self.daily_trades += 1
        logger.info(
            f"[TRACKER] {strategy_name}/{window} | {direction} {symbol} | "
            f"entry={entry_price:.2f} sl={sl:.2f} tgt={target:.2f} qty={quantity} | "
            f"open={self.open_count()}/{ORB_MAX_POSITIONS} | today={self.daily_trades}"
        )

    def remove_position(self, symbol: str) -> None:
        if symbol in self._positions:
            del self._positions[symbol]
            logger.info(f"[TRACKER] Closed {symbol} | open={self.open_count()}")

    def summary(self) -> str:
        if not self._positions:
            return "No open positions."
        lines = [f"Open positions ({self.open_count()}/{ORB_MAX_POSITIONS}):"]
        for p in self._positions.values():
            arrow = "↑" if p.direction == "BUY" else "↓"
            lines.append(
                f"  {arrow} {p.symbol} [{p.window}] {p.direction} qty={p.quantity} "
                f"entry={p.entry_price:.2f} sl={p.sl:.2f} tgt={p.target:.2f} @ {p.entry_time}"
            )
        lines.append(f"Total trades today: {self.daily_trades}")
        return "\n".join(lines)
