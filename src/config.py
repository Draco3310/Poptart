import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Map full symbol to short coin name for directory structure
SYMBOL_MAP: Dict[str, str] = {"BTCUSDT": "BTC", "BTC/USDT": "BTC", "XRPUSDT": "XRP", "SOLUSDT": "SOL"}


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

    # --- Indicators ---
    ema_period_slow: int = 200
    ema_period_fast: int = 50
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    adx_period: int = 14
    adx_threshold: int = 25
    adx_threshold_strategy: Optional[int] = None  # If None, uses adx_threshold
    atr_period: int = 14
    donchian_period: int = 48
    vol_ma_period: int = 20

    # --- Risk / Volatility ---
    atr_multiplier: float = 2.5
    max_volatility_threshold: float = 0.003
    
    # --- ML Thresholds ---
    ml_threshold_range: float = 0.5
    ml_threshold_trend: float = 0.5
    ml_long_threshold: float = 0.50
    ml_short_threshold: float = 0.45
    ml_long_threshold_low: float = 0.45
    ml_short_threshold_low: float = 0.50

    # --- Mean Reversion Strategy Params ---
    mean_rev_min_roi: float = 0.005
    mean_rev_min_profit: float = 0.0
    mean_rev_trend_ml_threshold: float = 0.6
    mean_rev_onchain_multiplier: float = 1.2
    mean_rev_lock_profit_multiplier_long: float = 1.005
    mean_rev_lock_profit_multiplier_short: float = 0.995
    adx_ranging_threshold: int = 50
    band_atr_buffer: float = 0.0
    
    # OBI Thresholds
    obi_long_threshold: float = -0.4
    obi_short_threshold: float = 0.4

    # --- Trend Strategy Params ---
    trend_trailing_stop_type: str = "ATR"  # "ATR" or "PERCENT"
    trend_trailing_stop_multiplier: float = 2.0
    trend_max_extension: float = 0.015  # Max distance from EMA200 (1.5%)
    trend_rsi_max: float = 60.0  # Max RSI for Longs (Strict)
    trend_adx_max: float = 40.0  # Max ADX (Avoid Exhaustion)
    trend_vol_max: float = 1.2  # Max Volume Ratio (Strict)
    trend_dist_to_poc_max: float = 0.02
    trend_vol_ratio_mania: float = 2.0
    trend_vol_ratio_min: float = 1.0
    trend_vol_ratio_climax: float = 1.5
    trend_size_multiplier: float = 0.6
    trend_ml_exit_threshold: float = 0.45
    trend_trail_atr_mania: float = 3.0
    trend_trail_atr_standard: float = 2.0
    ml_trend_long_threshold: float = 0.65

    # --- Ratcheting Stop Loss ---
    ratchet_breakeven_roi: float = 0.01
    ratchet_lock_profit_roi: float = 0.025
    ratchet_trail_roi: float = 0.020
    ratchet_trail_atr_multiplier: float = 2.0

    # --- Execution ---
    cooldown_minutes: int = 30
    order_timeout_seconds: int = 3
    limit_order_buffer: float = 0.005
    order_book_depth: int = 10
    tp1_ratio: float = 0.5
    breakeven_buffer: float = 0.002

    # Models
    # Model paths are now dynamic via get_model_path()

    # Data Availability
    has_5m: bool = False

    # DCA Specifics (BTC)
    dca_interval_minutes: Optional[int] = None
    dca_notional_per_trade: Optional[float] = None
    dca_target_allocation: float = 0.20  # Target 20% allocation
    dca_dip_threshold_rsi: int = 50  # Buy only if RSI < 50
    dca_rebalance_buffer_lower: float = 0.95
    dca_rebalance_buffer_upper: float = 1.05
    dca_min_trade_amount: float = 10.0


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
    MAX_DRAWDOWN_LIMIT = 0.05  # 5% Max Drawdown
    VOL_TARGET_ANNUAL = 0.40  # 40% Annualized Volatility Target
    
    # Regime Scaling
    REGIME_SCALE_TREND = 1.2
    REGIME_SCALE_CHOP = 0.8
    REGIME_SCALE_VOLATILITY = 0.0
    REGIME_HYSTERESIS = 5.0
    
    # Microstructure Filters
    MAX_SPREAD_PERCENT = 0.005  # 0.5% Max Spread

    # Database
    DB_PATH = os.getenv("DB_PATH", "data/sentinel.db")
    
    # Backtest Results
    BACKTEST_RESULTS_DIR = os.getenv("BACKTEST_RESULTS_DIR", "backtesting_results")

    # Proxy
    HTTP_PROXY = os.getenv("HTTP_PROXY")
    HTTPS_PROXY = os.getenv("HTTPS_PROXY")

    # --- System Flags ---
    # If True, requires ML score to confirm signals. If False, uses Strategy logic only.
    ML_ENABLED = os.getenv("ML_ENABLED", "True").lower() == "true"
    # If True, simulates trades without placing orders on Kraken.
    DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

    # If True, uses L2 features (OBI, Spread) for filtering.
    # Set to False for backtesting if L2 data is mocked/unreliable.
    USE_L2_FILTERS = os.getenv("USE_L2_FILTERS", "True").lower() == "true"

    # Close-Only Stops (High Beta Optimization)
    CLOSE_ONLY_STOPS = os.getenv("CLOSE_ONLY_STOPS", "True").lower() == "true"

    # --- Strategy Constants ---
    SYMBOL = "XRP/USDT"
    TIMEFRAME_PRIMARY = "5m"
    TIMEFRAME_CONFIRM = "1m"
    TIMEFRAME_CONTEXT = "1h"
    
    # Note: All strategy-specific parameters have been moved to PairConfig.
    # Use get_pair_config(symbol) to access them.
    
    # Execution Defaults (System-wide fallback)
    ORDER_BOOK_DEPTH = 10
    LIMIT_ORDER_BUFFER = 0.005
    ORDER_TIMEOUT_SECONDS = 3
    
    # Live Data Buffers
    LIVE_DATA_BUFFER_SIZE = 5000
    LIVE_DATA_BUFFER_SIZE_CONFIRM = 15000

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
        symbol="BTC/USDT",
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
