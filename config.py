"""
config.py
---------
Configuration for the ORB v3 (Opening Range Breakout) trading bot.

ORB v3 adds time-decay stop management on top of v2's entry framework:
  • Positions are treated as "wrong until proven right"
  • Progressive SL tightening at T+20, T+40, T+60 minutes after entry
  • Profit ratchet: breakeven at 0.5R, locked profit at 1.0R
  • New exit: TIME_EXIT — position evicted if it doesn't reach 0.5R by T+60

Primary ORB window (15-min) only — secondary disabled after v2 backtest showed
36% win rate on 30-min ORB trades (vs 50% for 15-min ORB).

All params can be overridden via environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Stock Universe — 55 NSE F&O stocks for ORB gap-and-breakout plays
# ---------------------------------------------------------------------------
# Selection criteria:
#   1. Active F&O participation → overnight institutional positioning drives gaps
#   2. Sector catalyst exposure → news-driven overnight moves
#   3. Sufficient liquidity → clean fills at market price
# ---------------------------------------------------------------------------
ORB_STOCK_UNIVERSE = [
    # Banking — most liquid, cleanest ORB structure
    "HDFCBANK", "SBIN", "AXISBANK", "ICICIBANK", "KOTAKBANK",
    "BANKBARODA", "PNB", "INDUSINDBK", "FEDERALBNK",
    "CANBK", "UNIONBANK", "IDFCFIRSTB",

    # Financials — rate-sensitive, gap on RBI/macro news
    "BAJFINANCE", "CHOLAFIN", "BAJAJFINSV", "LICHSGFIN", "MANAPPURAM",

    # IT — US tech moves and USD/INR drive overnight gaps
    "INFY", "WIPRO", "HCLTECH", "TCS", "TECHM", "LTIM", "PERSISTENT", "COFORGE",

    # Auto — monthly sales data and commodity input costs create gaps
    "TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "ASHOKLEY",

    # Metals — LME copper/steel overnight moves
    "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "HINDCOPPER", "NMDC", "SAIL",

    # Oil & Gas — crude oil price gaps
    "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL",

    # Power / Infra — policy and sector news sensitive
    "TATAPOWER", "ADANIGREEN", "ADANIPORTS", "ADANIENT", "NTPC", "POWERGRID",

    # High-beta / momentum
    "SUZLON", "ETERNAL", "IRCTC",

    # Pharma — FDA headline and USFDA inspection driven gaps
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
]

# ---------------------------------------------------------------------------
# Candidate Selection
# ---------------------------------------------------------------------------
ORB_TOP_N_STOCKS = int(os.getenv("ORB_TOP_N_STOCKS", "25"))
# ATR%-ranked from universe. Picks 25 most volatile stocks each morning.
# More candidates → more signal opportunities without lowering quality.

# ---------------------------------------------------------------------------
# Primary ORB Parameters (15-min window: 9:15–9:30 IST)
# ---------------------------------------------------------------------------
ORB_MINUTES             = int(os.getenv("ORB_MINUTES",            "15"))
# First 15 minutes of NSE session define the opening range.

ORB_VOLUME_MULTIPLIER   = float(os.getenv("ORB_VOLUME_MULTIPLIER", "2.0"))
# Breakout candle volume >= this × 10-candle avg.
# v3 backtest at 1.15× showed 53% STOP_LOSS rate — most were low-volume fakeouts.
# Raised to 2.0×: only explosive-volume breakouts qualify. Institutional conviction
# signals (2× avg) have dramatically higher follow-through. Cuts bad entries sharply.

ORB_MIN_RANGE_PCT       = float(os.getenv("ORB_MIN_RANGE_PCT",     "0.005"))
# Min ORB size as % of price (0.5%). Raised from 0.3%:
# 0.3% on Rs500 stock = Rs1.50 range → Rs900 target P&L at 2× mult.
# 0.5% on Rs500 stock = Rs2.50 range → Rs1,500 target P&L at 2× mult.
# Minimum range filter directly lifts gross P&L per winning trade.

ORB_MAX_RANGE_PCT       = float(os.getenv("ORB_MAX_RANGE_PCT",     "0.04"))
# Max ORB size as % of price (4%). Avoids extreme gap events.

ORB_CHASE_LIMIT_PCT     = float(os.getenv("ORB_CHASE_LIMIT_PCT",   "0.010"))
# Max extension beyond ORB level before entry blocked.
# Relaxed from 0.8% → 1.0% — captures breakouts that extend slightly further.

ORB_TARGET_MULTIPLIER   = float(os.getenv("ORB_TARGET_MULTIPLIER", "2.0"))
# Target = entry ± (ORB range × 2.0).
# v3 backtest at 1.5× showed only 7% of trades reached target — avg win too small.
# Raised to 2.0× alongside tighter entry filters (volume 2.0×, gap 0.5%):
# high-conviction breakouts have stronger momentum → 2.0× is achievable.
# TIME_EXIT at T+60 still provides partial profit capture on trades that stall.
# Expected: fewer but larger wins; net/trade target Rs400–500.

ORB_ENTRY_CUTOFF_TIME   = os.getenv("ORB_ENTRY_CUTOFF_TIME",       "11:00")
# Primary ORB entries cut off at 11:00 IST (90-min window after ORB establishes).

ORB_MIN_GAP_PCT         = float(os.getenv("ORB_MIN_GAP_PCT",       "0.005"))
# Gap-direction filter threshold (0.5%). Raised from 0.2%:
# 0.2% gaps are daily noise — almost any stock qualifies. 0.5% gaps reflect real
# overnight positioning by institutions (news, earnings, sector moves). ORB
# breakouts in the direction of a 0.5%+ gap have much higher follow-through.
# Core edge of the strategy — gap direction + gap magnitude both matter.

ORB_POSITION_SCALE      = float(os.getenv("ORB_POSITION_SCALE",    "1.0"))
# Capital scale per trade. 1.0 = Rs150,000 per trade.
# With 10 max positions: Rs150k × 10 = Rs1.5M max deployment.

ORB_FAILED_BUFFER_PCT   = float(os.getenv("ORB_FAILED_BUFFER_PCT", "0.005"))
# 0.5% close back inside range triggers ORB_FAILED exit.
# Tightened from 0.8% → 0.5%: cuts failing breakouts faster, reducing loss
# per failed trade. ORB_FAILED is the cleanest exit type — trigger it sooner.

ORB_BREAKEVEN_TRIGGER_R = float(os.getenv("ORB_BREAKEVEN_TRIGGER_R", "0.5"))
# Move SL to breakeven once trade gains 50% of initial risk.
# Tightened from 0.6R → 0.5R for faster capital protection at higher scale.

# ---------------------------------------------------------------------------
# Secondary ORB Parameters (30-min window: 9:15–9:45 IST)
# ---------------------------------------------------------------------------
ORB_SECONDARY_WINDOW_ENABLED = False
# DISABLED: backtest showed 30-min ORB had 36% win rate vs 50% for 15-min,
# producing net -Rs6,887 on 14 trades over 30 days. The wider range creates
# looser setups with lower conviction. Re-enable via env var if desired.
# Override: ORB_SECONDARY_WINDOW_ENABLED=true in .env

ORB_MINUTES_SECONDARY       = int(os.getenv("ORB_MINUTES_SECONDARY",       "30"))
ORB_ENTRY_CUTOFF_SECONDARY  = os.getenv("ORB_ENTRY_CUTOFF_SECONDARY",      "11:30")
ORB_VOLUME_MULT_SECONDARY   = float(os.getenv("ORB_VOLUME_MULT_SECONDARY", "1.20"))
ORB_CHASE_LIMIT_SECONDARY   = float(os.getenv("ORB_CHASE_LIMIT_SECONDARY", "0.010"))

# ---------------------------------------------------------------------------
# Position Management
# ---------------------------------------------------------------------------
ORB_MAX_POSITIONS            = int(os.getenv("ORB_MAX_POSITIONS", "10"))
# Max simultaneous open positions. Up from 5 → 10.

POSITION_SIZE_INR            = int(os.getenv("POSITION_SIZE_INR", "150000"))
# Capital per trade in INR. 10 × Rs150k = Rs1.5M max deployment.

DAILY_LOSS_CIRCUIT_BREAKER   = -999_999
# Effectively disabled. Re-enable with a sensible value if needed.

ONE_TRADE_PER_STOCK_PER_DAY  = True
# Once a stock closes (win or loss), blocked from re-entry same day.

# ---------------------------------------------------------------------------
# Shared Indicator Parameters
# ---------------------------------------------------------------------------
EMA_FAST        = 9
EMA_SLOW        = 20
EMA_MACRO       = 50
RSI_PERIOD      = 14
VOLUME_LOOKBACK = 10

# SuperTrend (used as profit-protection exit in v3)
ORB_SUPERTREND_PERIOD      = int(float(os.getenv("ORB_SUPERTREND_PERIOD",      "7")))
# Standard SuperTrend period. 7 is widely used; lower = more responsive,
# higher = smoother but slower to react to reversals.

ORB_SUPERTREND_MULTIPLIER  = float(os.getenv("ORB_SUPERTREND_MULTIPLIER",      "3.0"))
# ATR multiplier for band width. 3.0 on 2-min charts gives clean signals
# without excessive whipsaw. 2.0 is tighter (more exits, more noise).

ORB_SUPERTREND_MIN_GAIN_R  = float(os.getenv("ORB_SUPERTREND_MIN_GAIN_R",      "0.3"))
# ST_EXIT only fires when the position has at least this much gain (in R-units).
# Below 0.3R the 2-min ST is too noisy for reliable profit protection;
# time-decay stops handle flat/losing trades. At 0.3R+ we have real profit
# to protect and the ST flip signals genuine momentum reversal.

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
TRADE_START_TIME    = "09:20"   # Wait until this time before entering the loop
SQUARE_OFF_TIME     = "15:10"   # Force-close all positions; 10-min buffer before Zerodha MIS
CANDLE_INTERVAL     = "2m"      # yfinance interval string
LOOP_SLEEP_SECONDS  = 120       # Sleep between strategy iterations (2-min candle)

# ---------------------------------------------------------------------------
# NIFTY50 Market Regime Filter
# ---------------------------------------------------------------------------
# Detects intraday market direction and adjusts position limits / allowed sides.
# BULL  → LONG_ONLY direction filter, full position count
# BEAR  → SHORT_ONLY direction filter, reduced position count
# NEUTRAL → both directions, moderate position count
# ---------------------------------------------------------------------------
REGIME_BULL_THRESHOLD        = 0.20
REGIME_BEAR_THRESHOLD        = -0.20
REGIME_BULL_MAX_POSITIONS    = 3    # v3 quality-over-quantity: only top 3 ATR%-ranked
                                    # setups. Trades 4-5 are lower conviction and dilute
                                    # expectancy. Fewer positions, better average quality.
REGIME_BEAR_MAX_POSITIONS    = 3    # Same — top 3 setups only per regime.
REGIME_NEUTRAL_MAX_POSITIONS = 3    # Consistent cap across all regimes.

# ---------------------------------------------------------------------------
# Stocksdeveloper / Zerodha Webhook
# ---------------------------------------------------------------------------
STOCKSDEVELOPER_URL     = "https://tv.stocksdeveloper.in/"
STOCKSDEVELOPER_API_KEY = os.getenv("STOCKSDEVELOPER_API_KEY")
STOCKSDEVELOPER_ACCOUNT = os.getenv("STOCKSDEVELOPER_ACCOUNT", "AbhiZerodha")

if not STOCKSDEVELOPER_API_KEY:
    raise EnvironmentError(
        "STOCKSDEVELOPER_API_KEY is not set. "
        "Add it to your .env file or GitHub Actions secrets."
    )

EXCHANGE     = "NSE"
PRODUCT_TYPE = "INTRADAY"
ORDER_TYPE   = "MARKET"
VARIETY      = "REGULAR"

# ---------------------------------------------------------------------------
# ORB v3 — Time-Decay Stop Parameters
# ---------------------------------------------------------------------------
# Philosophy: "A position entered is assumed wrong until the market proves it
# right within a defined time window." — progressive SL tightening forces
# each trade to earn its right to stay open.
#
# At each checkpoint, if the trade hasn't met the minimum gain threshold,
# the stop loss is tightened or the trade is exited entirely (TIME_EXIT).
#
# Profit ratchet runs in parallel — locks gains independently of time.
# ---------------------------------------------------------------------------

# Time checkpoints (minutes after entry)
ORB_V3_CHECKPOINT_1_MINS    = int(float(os.getenv("ORB_V3_CHECKPOINT_1_MINS",    "20")))
# T+20: trade must have moved positive (gain > 0R). If not → tighten SL.

ORB_V3_CHECKPOINT_2_MINS    = int(float(os.getenv("ORB_V3_CHECKPOINT_2_MINS",    "40")))
# T+40: trade must have gained at least 0.3R. If not → SL forced to breakeven.

ORB_V3_TIME_EXIT_MINS       = int(float(os.getenv("ORB_V3_TIME_EXIT_MINS",       "60")))
# T+60: trade must have gained at least 0.5R. If not → TIME_EXIT (flat close).

# Minimum gain thresholds at each checkpoint (in R-multiples)
ORB_V3_T20_MIN_GAIN_R       = float(os.getenv("ORB_V3_T20_MIN_GAIN_R",           "0.0"))
# At T+20: gain < 0R → apply tight SL (30% of original risk from entry).

ORB_V3_T20_TIGHT_SL_FACTOR  = float(os.getenv("ORB_V3_T20_TIGHT_SL_FACTOR",     "0.30"))
# Tight SL = entry ± (initial_risk × 0.30). Cuts remaining loss to 30% of original.

ORB_V3_T40_MIN_GAIN_R       = float(os.getenv("ORB_V3_T40_MIN_GAIN_R",           "0.30"))
# At T+40: gain < 0.3R → move SL to breakeven (entry price).

ORB_V3_TIME_EXIT_MIN_GAIN_R = float(os.getenv("ORB_V3_TIME_EXIT_MIN_GAIN_R",     "0.50"))
# At T+60: gain < 0.5R → TIME_EXIT regardless of SL. Exit at market (close price).

# Profit ratchet — runs independently of time checkpoints
ORB_V3_RATCHET_BE_R         = float(os.getenv("ORB_V3_RATCHET_BE_R",             "0.50"))
# At gain >= 0.5R: move SL to breakeven (entry price). Replaces v2 BREAKEVEN_TRIGGER_R.

ORB_V3_RATCHET_LOCK_R       = float(os.getenv("ORB_V3_RATCHET_LOCK_R",           "1.00"))
# At gain >= 1.0R: lock a fraction of profit by raising SL above entry.

ORB_V3_TRAIL_LOCK_FRACTION  = float(os.getenv("ORB_V3_TRAIL_LOCK_FRACTION",      "0.40"))
# Locked SL = entry + (initial_risk × 0.40).
# After 1.0R gain, the worst outcome is a 0.4R win (never a loss).
