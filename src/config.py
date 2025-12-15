import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Map full symbol to short coin name for directory structure
SYMBOL_MAP: Dict[str, str] = {"BTCUSDT": "BTC", "XRPUSDT": "XRP", "SOLUSDT": "SOL"}


@dataclass
class PairConfig:
    """
    Configuration specific to a trading pair.
    """

    symbol: str
    # Data paths are now dynamic via get_data_path()

    # Strategy Flags
    enable_mean_reversion: bool = False
    enable_trend_following: bool = False
    enable_dca_mode: bool = False

    # Risk / Volatility
    atr_multiplier: float = 2.5
    max_volatility_threshold: float = 0.003

    # Indicators
    ema_period_slow: int = 200
    ema_period_fast: int = 50
    adx_threshold: int = 25
    adx_threshold_strategy: Optional[int] = None  # If None, uses adx_threshold

    # ML Thresholds
    ml_threshold_range: float = 0.5
    ml_threshold_trend: float = 0.5

    # Trend Strategy Params
    trend_trailing_stop_type: str = "ATR"  # "ATR" or "PERCENT"
    trend_trailing_stop_multiplier: float = 2.0
    trend_max_extension: float = 0.015  # Max distance from EMA200 (1.5%)
    trend_rsi_max: float = 60.0  # Max RSI for Longs (Strict)
    trend_adx_max: float = 40.0  # Max ADX (Avoid Exhaustion)
    trend_vol_max: float = 1.2  # Max Volume Ratio (Strict)

    # Execution
    cooldown_minutes: int = 30

    # Models
    # Model paths are now dynamic via get_model_path()

    # Data Availability
    has_5m: bool = False

    # DCA Specifics (BTC)
    dca_interval_minutes: Optional[int] = None
    dca_notional_per_trade: Optional[float] = None
    dca_target_allocation: float = 0.20  # Target 20% allocation
    dca_dip_threshold_rsi: int = 50  # Buy only if RSI < 50


class Config:
    """
    Centralized configuration for XRP-Sentinel V3.
    """

    # --- Environment Variables ---
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Risk Management
    # Default to 0.02 (2%) if not set
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))

    # Database
    DB_PATH = os.getenv("DB_PATH", "data/sentinel.db")

    # Proxy
    HTTP_PROXY = os.getenv("HTTP_PROXY")
    HTTPS_PROXY = os.getenv("HTTPS_PROXY")

    # --- System Flags ---
    # If True, requires ML score to confirm signals. If False, uses Strategy logic only.
    ML_ENABLED = True  # os.getenv("ML_ENABLED", "True").lower() == "true"
    # If True, simulates trades without placing orders on Kraken.
    DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

    # If True, uses L2 features (OBI, Spread) for filtering.
    # Set to False for backtesting if L2 data is mocked/unreliable.
    USE_L2_FILTERS = True

    # Close-Only Stops (High Beta Optimization)
    CLOSE_ONLY_STOPS = True

    # --- Strategy Constants (Hardcoded) ---
    SYMBOL = "XRP/USDT"
    TIMEFRAME_PRIMARY = "5m"
    TIMEFRAME_CONFIRM = "1m"

    # Donchian Channel
    DONCHIAN_PERIOD = 48

    # EMA
    EMA_PERIOD: int = 200
    EMA_PERIOD_SLOW: int = 200
    EMA_PERIOD_FAST: int = 50

    # RSI
    RSI_PERIOD: int = 14
    # Standard thresholds (ML will filter)
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30

    # ATR
    ATR_PERIOD: int = 14
    # Medium stop
    ATR_MULTIPLIER: float = 2.5

    # ADX
    ADX_PERIOD: int = 14
    ADX_THRESHOLD: int = 25

    # --- Strategy Parameters (Mean Reversion) ---
    # Market Regime
    # Relaxed ranging definition (ML will filter)
    ADX_RANGING_THRESHOLD = 50  # Max ADX to consider "Ranging"

    # Volatility Filter (Avoid catching knives in high vol crashes)
    MAX_VOLATILITY_THRESHOLD = 0.003  # Max ATR/Close ratio

    # ML Thresholds (Aggressive Calibration for Frequency)
    # Tier 1 (High Confidence) - Full Size
    ML_LONG_THRESHOLD = 0.50  # Accept neutral-bullish
    ML_SHORT_THRESHOLD = 0.45  # Accept neutral-bearish

    # Tier 2 (Medium Confidence) - Half Size
    ML_LONG_THRESHOLD_LOW = 0.45
    ML_SHORT_THRESHOLD_LOW = 0.50

    # Trend Following Thresholds
    ML_TREND_LONG_THRESHOLD = 0.65

    # Trend Exit Defaults
    TREND_TRAILING_STOP_TYPE = "ATR"
    TREND_TRAILING_STOP_MULTIPLIER = 2.0
    TREND_MAX_EXTENSION = 0.01
    TREND_RSI_MAX = 60.0

    # Band touch buffer in ATR units (for much stronger extremes)
    BAND_ATR_BUFFER = 0.0

    # Minimum time between entries (per side) in minutes
    MIN_TRADE_COOLDOWN_MINUTES = 30

    # Exit Logic
    MEAN_REV_MIN_ROI = 0.005  # 0.5% Minimum ROI Target (Relaxed from 1.5%)
    MEAN_REV_MIN_PROFIT = 0.0  # 0.0% Minimum Potential Profit

    # OBI Thresholds
    OBI_LONG_THRESHOLD = -0.4  # Filter extreme sell pressure (Crash protection)
    OBI_SHORT_THRESHOLD = 0.4  # Filter extreme buy pressure (Pump protection)

    # Ratcheting Stop Loss
    RATCHET_BREAKEVEN_ROI = 0.01  # 1.0% - Move SL to breakeven
    RATCHET_LOCK_PROFIT_ROI = 0.025  # 2.5% - Lock in 1.0% profit
    RATCHET_TRAIL_ROI = 0.020  # 2.0% - Start dynamic trailing
    RATCHET_TRAIL_ATR_MULTIPLIER = 2.0  # Trail by 2.0x ATR

    # Volume MA
    VOL_MA_PERIOD = 20

    # Execution
    ORDER_TIMEOUT_SECONDS = 3
    LIMIT_ORDER_BUFFER = 0.005  # 0.5% aggressive limit
    ORDER_BOOK_DEPTH = 10  # Depth for L2 Data

    # Split Position Logic
    TP1_RATIO = 0.5  # Close 50% at TP1
    BREAKEVEN_BUFFER = 0.002  # 0.2% buffer for fees when moving to BE

    @classmethod
    def validate(cls) -> None:
        """Ensures critical configuration is present."""
        missing = []
        if not cls.KRAKEN_API_KEY:
            missing.append("KRAKEN_API_KEY")
        if not cls.KRAKEN_API_SECRET:
            missing.append("KRAKEN_API_SECRET")
        if not cls.TELEGRAM_TOKEN:
            missing.append("TELEGRAM_TOKEN")
        if not cls.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")

        if missing:
            raise ValueError(f"Missing critical environment variables: {', '.join(missing)}")


# --- Per-Pair Configurations ---

PAIR_CONFIGS: Dict[str, PairConfig] = {
    "XRPUSDT": PairConfig(
        symbol="XRPUSDT",
        has_5m=True,
        enable_mean_reversion=True,
        enable_trend_following=True,
        enable_dca_mode=False,
        # Optimized Params (WFO Dec 2025)
        atr_multiplier=2.8,
        max_volatility_threshold=0.003,
        ema_period_slow=180,
        ema_period_fast=15,
        adx_threshold=30,
        ml_threshold_range=0.68,
        ml_threshold_trend=0.50,
        cooldown_minutes=30,
    ),
    "BTCUSDT": PairConfig(
        symbol="BTCUSDT",
        has_5m=False,
        enable_mean_reversion=False,
        enable_trend_following=False,
        enable_dca_mode=True,
        dca_interval_minutes=60,
        dca_notional_per_trade=10.0,
        dca_dip_threshold_rsi=50,
    ),
    "SOLUSDT": PairConfig(
        symbol="SOLUSDT",
        has_5m=True,
        enable_mean_reversion=False,
        enable_trend_following=True,
        enable_dca_mode=False,
        # Optimized Params (WFO Feb 2025)
        atr_multiplier=3.0,
        max_volatility_threshold=0.005,
        ema_period_slow=180,
        ema_period_fast=15,
        adx_threshold=28,
        adx_threshold_strategy=28,
        ml_threshold_range=0.68,
        ml_threshold_trend=0.5,
        cooldown_minutes=30,
        trend_trailing_stop_type="ATR",
        trend_trailing_stop_multiplier=2.0,
        trend_max_extension=0.03,
        trend_rsi_max=60.0,
        trend_adx_max=40.0,
        trend_vol_max=1.2,
    ),
}


def get_pair_config(symbol: str) -> PairConfig:
    """
    Retrieves the configuration for a specific pair.
    Defaults to XRPUSDT if not found.
    """
    return PAIR_CONFIGS.get(symbol, PAIR_CONFIGS["XRPUSDT"])


def get_data_path(symbol: str, timeframe: str = "1m") -> Path:
    """
    Resolves the file path for a pair's data.
    Format: data/{ShortName}/{Symbol}_{Timeframe}.csv
    Example: data/XRP/XRPUSDT_1m.csv
    """
    short_name = SYMBOL_MAP.get(symbol, symbol)
    return Path(f"data/{short_name}/{symbol}_{timeframe}.csv")


def get_model_path(symbol: str, model_type: str, version: str = "v1", ext: str = ".joblib") -> str:
    """
    Resolves the file path for a model.
    Format: models/{ModelType}_{Symbol}_{Version}{ext}
    Example: models/rf_tpsl_XRPUSDT_v1.joblib
    """
    return f"models/{model_type}_{symbol}_{version}{ext}"
