# Poptart Gal Friday V2 - XRP Sentinel

**Poptart Gal Friday V2** is an automated trading bot designed for the Kraken exchange, specifically targeting the **XRP/USDT** pair. It features a **Modular Plugin Architecture** that separates core execution logic from trading strategies and machine learning predictors.

## 🏗 System Architecture

The system is divided into three main components:

1.  **Kernel (`src/core/`, `src/main.py`)**:
    *   Handles the main execution loop, data ingestion (Kraken), database management, and notifications (Telegram).
    *   Designed to be stable and rarely modified.

2.  **Brain (`src/core/feature_engine.py`)**:
    *   A centralized engine that computes technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.) and prepares data for ML models.
    *   Ensures all strategies and models work with the same enriched dataset.

3.  **Plugins (`src/strategies/`, `src/predictors/`)**:
    *   **Strategies**: Pluggable trading logic (e.g., `MeanReversionMTF`). Strategies consume enriched data and ML scores to generate signals.
    *   **Predictors**: Machine Learning models (XGBoost, Random Forest, LSTM) that provide confidence scores to filter or weight strategy signals.

### ⏱️ Hybrid Timeframe Logic
The system now employs a **Hybrid Timeframe Architecture** to maximize signal quality and precision:
*   **Primary Analysis (5m)**: Technical indicators (RSI, Bollinger Bands, EMA) are calculated on 5-minute candles to reduce noise and identify robust trends.
*   **Confirmation (1m)**: Entry signals generated on the 5m timeframe must be confirmed by 1-minute price action (e.g., touching bands) to ensure optimal entry timing.

## 🚀 Installation & Setup

### Prerequisites

*   [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
*   **OR** Python 3.12+ (for local development)
*   Kraken API Keys (API Key & Private Key)
*   Telegram Bot Token & Chat ID

### 1. Clone the Repository

```bash
git clone <repository-url>
cd poptart-gal-friday-v2-brain
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env  # If .env.example exists, otherwise create new
```

Add your credentials to `.env`:

```ini
# Kraken API
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

# Telegram Notification
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Risk Management
RISK_PER_TRADE=0.02  # 2% risk per trade

# Database Configuration
DB_PATH=data/sentinel.db
```

## 🐳 Running with Docker (Recommended)

The easiest way to run the bot is using Docker Compose.

1.  **Build and Run:**

    ```bash
    docker-compose up --build -d
    ```

2.  **View Logs:**

    ```bash
    docker-compose logs -f
    ```

3.  **Stop the Bot:**

    ```bash
    docker-compose down
    ```

## 🛠 Local Development

If you prefer to run the bot locally without Docker:

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Bot:**

    ```bash
    python -m src.main
    ```

## 📈 Backtesting

The project includes a robust backtesting engine to simulate strategies against historical data.

*   **Detailed Documentation**: See [Backtest Runner Details.md](Backtest Runner Details.md) for architecture, configuration, and usage instructions.
*   **Run Backtest**: `python -m src.backtest_runner`
*   **Output**: Detailed trade logs (including decision context) are saved to `data/backtest_decision_log.csv`.

## 🧪 Testing & Quality Assurance

The project includes a comprehensive testing suite using **Pytest**, **Ruff** (linting), and **Mypy** (type checking).

### Running the Test Suite

To run the full pipeline (Linting -> Type Checking -> Tests):

```bash
./run_suite.sh
```

Or run individual components:

*   **Tests:** `python -m pytest tests/ -v`
*   **Linting:** `ruff check .`
*   **Type Checking:** `mypy src/`

## 📂 Project Structure

```
src/
├── core/
│   └── feature_engine.py    # Centralized indicator calculation
├── predictors/              # ML Model plugins
│   ├── base_predictor.py    # Abstract base class
│   ├── lstm_model.py        # LSTM implementation
│   ├── orchestrator.py      # Manages model ensemble
│   ├── random_forest.py     # Random Forest implementation
│   └── xgboost_model.py     # XGBoost implementation
├── strategies/              # Trading strategy plugins
│   └── mean_reversion_mtf.py
├── config.py                # Configuration management
├── database.py              # Database interactions
├── exchange.py              # Kraken API wrapper (with DRY_RUN safety)
├── main.py                  # Entry point
└── notifier.py              # Telegram notifier
tests/                       # Test Suite
├── conftest.py              # Fixtures (Mock Exchange, Market Data)
├── test_functional.py       # FeatureEngine tests
├── test_security.py         # Security & Safety tests
└── test_strategy.py         # Strategy logic tests
pyproject.toml               # Tool configuration (Ruff, Mypy, Pytest)
run_suite.sh                 # Test execution script
