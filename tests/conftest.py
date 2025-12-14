from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.exchange import KrakenExchange


@pytest.fixture
def ohlcv_fixture() -> pd.DataFrame:
    """
    Generates synthetic OHLCV data for testing (Default 5m for backward compatibility).
    Includes 3000 periods of data to satisfy lookback requirements (EMA 200 on 1h requires ~2400 5m candles).
    """
    return generate_ohlcv(periods=3000, freq="5min")


@pytest.fixture
def ohlcv_5m_fixture() -> pd.DataFrame:
    """Generates synthetic 5m OHLCV data."""
    return generate_ohlcv(periods=3000, freq="5min")


@pytest.fixture
def ohlcv_1m_fixture() -> pd.DataFrame:
    """Generates synthetic 1m OHLCV data."""
    return generate_ohlcv(periods=15000, freq="1min")


def generate_ohlcv(periods: int, freq: str) -> pd.DataFrame:
    """Helper to generate synthetic OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=periods, freq=freq)

    # Generate synthetic price movement (Random Walk)
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, periods)
    price = 100 * (1 + returns).cumprod()

    data = {
        "timestamp": dates,
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price * (1 + np.random.normal(0, 0.002, periods)),  # Slight variation for close
        "volume": np.random.randint(1000, 5000, periods).astype(float),
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def mock_kraken() -> MagicMock:
    """
    Mocks the KrakenExchange class to prevent real API calls.
    Uses AsyncMock for async methods.
    """
    mock_exchange = MagicMock(spec=KrakenExchange)

    # Mock Async Methods
    mock_exchange.initialize = AsyncMock()
    mock_exchange.close = AsyncMock()
    mock_exchange.fetch_ohlcv = AsyncMock()
    mock_exchange.fetch_confirm_ohlcv = AsyncMock()

    # Mock Balance
    mock_exchange.fetch_balance = AsyncMock(return_value=1000.0)

    # Mock Ticker
    mock_exchange.get_market_price = AsyncMock(return_value=1.0)

    # Mock Order Creation
    mock_exchange.create_entry_order = AsyncMock(
        return_value={"id": "mock_order_123", "status": "closed", "filled": 100.0}
    )
    mock_exchange.create_exit_order = AsyncMock(
        return_value={"id": "mock_exit_123", "status": "closed", "filled": 100.0}
    )

    return mock_exchange
