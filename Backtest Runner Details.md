# Backtest Runner Details

## Overview

The **Backtest Runner** is the simulation engine for the Poptart Gal Friday V2 trading bot. It allows for rigorous testing of trading strategies and ML models against historical data without risking real capital. It is designed to replicate the live trading environment as closely as possible, including fee structures, slippage, and data processing pipelines.

## Architecture

The backtesting system is built around the `BacktestRunner` class, which orchestrates the interaction between data, strategies, and the simulated exchange.

### Core Components

1.  **`BacktestRunner` (`src/backtest_runner.py`)**:
    *   **Role**: The main controller. It iterates through historical data, updates the simulated exchange, calculates features, and executes strategy logic.
    *   **Key Responsibility**: Manages the simulation loop, time synchronization between 5m and 1m data, and result reporting.

2.  **`SimulatedKrakenExchange` (`src/simulated_exchange.py`)**:
    *   **Role**: A drop-in replacement for the real `KrakenExchange` class.
    *   **Key Responsibility**: Tracks balances, executes orders with simulated slippage and fees, and maintains a record of all trades.
    *   **Simulation Details**:
        *   **Fees**: Defaults to 0.26% (taker) and 0.16% (maker), configurable.
        *   **Slippage**: Simulates market impact based on volatility.

3.  **`FeatureEngine` (`src/core/feature_engine.py`)**:
    *   **Role**: The centralized calculation engine for technical indicators.
    *   **Key Responsibility**: Ensures that the indicators used in backtesting are *identical* to those used in live trading. It computes RSI, Bollinger Bands, MACD, ATR, and ML features.

4.  **`StrategySelector` (`src/strategies/strategy_selector.py`)**:
    *   **Role**: The decision-making logic.
    *   **Key Responsibility**: Selects the best strategy based on current market regime and ML predictions.

## Data Flow

1.  **Ingestion**: Historical OHLCV data (5-minute and 1-minute intervals) is loaded from CSV files or the database.
2.  **Synchronization**: The runner aligns the 5m and 1m data streams. For every 5-minute candle, it iterates through the corresponding five 1-minute candles to allow for precise entry/exit timing (Hybrid Timeframe Logic).
3.  **Enrichment**: The `FeatureEngine` calculates technical indicators on the growing dataset to prevent look-ahead bias.
4.  **Decision**:
    *   The `StrategySelector` analyzes the enriched data.
    *   It queries specific strategies (e.g., `MeanReversionMTF`) and ML models.
    *   It returns a signal (`LONG`, `SHORT`, `EXIT`, or `None`) along with "Decision Context" (why the decision was made).
5.  **Execution**: Valid signals are sent to the `SimulatedKrakenExchange` for execution.

## Logging & Reporting

The backtest runner produces comprehensive logs to aid in strategy tuning and debugging.

### 1. Trade Log (`data/backtest_trades.csv`)
A standard record of all executed trades.
*   **Columns**: `entry_time`, `exit_time`, `entry_price`, `exit_price`, `side`, `size`, `pnl`, `pnl_percent`, `exit_reason`.

### 2. Decision Context Log (`data/backtest_decision_log.csv`)
A detailed log of *every* trade decision, capturing the "why" behind the action. This is crucial for debugging bad trades.
*   **Context Captured**:
    *   **Indicators**: RSI, Bollinger Band positions, ADX, MACD.
    *   **ML Scores**: Confidence scores from the ML predictors.
    *   **Regime**: The detected market regime (e.g., Trending, Ranging).
    *   **Strategy Logic**: Specific conditions that triggered the signal.

### 3. Summary Report (`backtesting_results/backtest_report_YYYYMMDD_HHMMSS.txt`)
A high-level summary of the backtest performance.
*   **Metrics**: Total PnL, Win Rate, Max Drawdown, Sharpe Ratio, Profit Factor.

## Usage

To run a backtest, execute the `src/backtest_runner.py` script. Ensure you have historical data available in the `data/` directory.

```bash
# Run the backtest runner
python -m src.backtest_runner
```

### Configuration
Key parameters can be adjusted in `src/config.py` or directly in the `BacktestRunner` initialization:
*   `initial_balance`: Starting capital (default: $10,000).
*   `fee_rate`: Trading fee per transaction.
*   `slippage`: Estimated price slippage.

## Strategy Tuning Workflow

1.  **Run Backtest**: Execute the runner to generate logs.
2.  **Analyze Decision Log**: Open `data/backtest_decision_log.csv` to review specific trade setups. Look for losing trades and check the indicator values at the time of entry.
3.  **Adjust Logic**: Modify the strategy files (e.g., `src/strategies/mean_reversion_mtf.py`) to filter out identified bad setups (e.g., "Don't buy if ADX > 50").
4.  **Iterate**: Rerun the backtest to verify improvements.
