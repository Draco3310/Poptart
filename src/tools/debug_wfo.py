import logging
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config import PAIR_CONFIGS
from src.core.backtest_engine import BacktestEngine

# Setup Logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DebugWFO")


def run_debug() -> None:
    pair = "SOLUSDT"
    start_date = "2024-01-01"
    end_date = "2024-01-07"  # 1 week

    logger.info(f"Starting Debug Backtest for {pair} ({start_date} to {end_date})")

    # Simulate WFO settings
    overrides = {
        # "ML_ENABLED": False, # Removed hardcoded disable. Controlled by Config.
        "EMA_PERIOD_FAST": 20,
        "EMA_PERIOD_SLOW": 200,
        "ADX_THRESHOLD": 30,
    }

    pair_config = PAIR_CONFIGS[pair]

    start_time = time.time()

    try:
        engine = BacktestEngine(pair_config, overrides)
        result = engine.run(start_date, end_date)

        duration = time.time() - start_time
        logger.info(f"Backtest Complete in {duration:.2f}s")
        logger.info(f"Trades: {len(result.trades)}")
        logger.info(f"PnL: {result.metrics.get('pnl_percent', 0.0):.2f}%")

    except Exception as e:
        logger.error(f"Backtest Failed: {e}", exc_info=True)


if __name__ == "__main__":
    run_debug()
