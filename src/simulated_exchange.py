import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class SimulatedKrakenExchange:
    """
    Simulated Kraken Exchange for Backtesting.
    Replays historical data and simulates order execution.
    """

    def __init__(
        self,
        data_path: str,
        initial_balance: float = 1000.0,
        fee_rate: float = 0.001,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        base_currency: str = "XRP",
    ):
        self.fee_rate = fee_rate
        self.base_currency = base_currency

        # Load data with mixed header detection
        try:
            # First, peek at the first row to guess format
            peek = pd.read_csv(data_path, nrows=1, header=None)
            if peek.empty:
                raise ValueError("Data file is empty")
            first_row = peek.iloc[0]

            # Heuristic: If first row contains strings like 'timestamp' or 'open', it has a header.
            # If it contains numbers, it's likely headerless.
            has_header = False
            first_col_str = str(first_row[0]).lower()
            second_col_str = str(first_row[1]).lower()

            if isinstance(first_row[0], str) and (
                "timestamp" in first_col_str or "open_time" in first_col_str or "date" in first_col_str
            ):
                has_header = True
            elif isinstance(first_row[1], str) and "open" in second_col_str:
                has_header = True

            if has_header:
                self.data = pd.read_csv(data_path)
                self.data.columns = [c.lower().strip() for c in self.data.columns]

                # Rename open_time to timestamp if present (Binance format)
                if "open_time" in self.data.columns:
                    self.data.rename(columns={"open_time": "timestamp"}, inplace=True)
            else:
                # Kraken format: timestamp, open, high, low, close, volume, count
                # We only need the first 6
                self.data = pd.read_csv(
                    data_path, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "count"]
                )
                # Drop extra columns if any
                self.data = self.data[["timestamp", "open", "high", "low", "close", "volume"]]

            # Parse timestamp
            # Try to infer format (int ms or ISO string)
            if pd.api.types.is_numeric_dtype(self.data["timestamp"]):
                # Assume ms if large int, s if small? Usually crypto data is ms.
                # Heuristic: if max > 3000000000 (year 2065 in seconds), it's probably ms
                if self.data["timestamp"].max() > 3000000000:
                    self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit="ms")
                else:
                    self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit="s")
            else:
                self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])

        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise

        # Ensure numeric types for OHLCV and downcast to float32
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce").astype("float32")

        # Drop rows with NaNs in critical columns
        self.data.dropna(subset=numeric_cols, inplace=True)

        # Filter by Date Range
        if start_date:
            self.data = self.data[self.data["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            self.data = self.data[self.data["timestamp"] <= pd.to_datetime(end_date)]

        self.data = self.data.sort_values("timestamp").reset_index(drop=True)

        if self.data.empty:
            logger.error(f"No data found for range {start_date} to {end_date}")
            raise ValueError("Data is empty after filtering.")

        # Pre-convert to numpy arrays for fast access
        self.timestamp_arr = self.data["timestamp"].values
        self.open_arr = self.data["open"].values
        self.high_arr = self.data["high"].values
        self.low_arr = self.data["low"].values
        self.close_arr = self.data["close"].values
        self.volume_arr = self.data["volume"].values

        self.current_index = 0
        self.balance = {"USDT": initial_balance, self.base_currency: 0.0}
        self.initial_balance: float = initial_balance

        # Track orders and trades
        self.orders: list[Dict[str, Any]] = []
        self.trades: list[Dict[str, Any]] = []

        logger.info(f"Simulated Exchange initialized with {len(self.data)} candles.")
        logger.info(f"Initial Balance: {initial_balance} USDT")

    def reset(self) -> None:
        """Resets the simulation to the beginning."""
        self.current_index = 0
        self.balance = {"USDT": self.initial_balance, self.base_currency: 0.0}
        self.orders = []
        self.trades = []

    def step(self) -> bool:
        """Advances the simulation by one candle."""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            return True
        return False

    def get_current_candle(self) -> pd.Series:
        """Returns the current candle."""
        return self.data.iloc[self.current_index]

    def get_current_time(self) -> datetime:
        """Returns the timestamp of the current candle."""
        # Use fast numpy access
        return pd.Timestamp(self.timestamp_arr[self.current_index])

    def get_current_time_fast(self) -> pd.Timestamp:
        """Fast access to current timestamp."""
        return pd.Timestamp(self.timestamp_arr[self.current_index])

    def get_current_close_fast(self) -> float:
        """Fast access to current close price."""
        return float(self.close_arr[self.current_index])

    def get_current_low_fast(self) -> float:
        """Fast access to current low price."""
        return float(self.low_arr[self.current_index])

    def get_current_high_fast(self) -> float:
        """Fast access to current high price."""
        return float(self.high_arr[self.current_index])

    def fetch_ohlcv(self, limit: int = 300, timeframe: str = Config.TIMEFRAME_PRIMARY) -> pd.DataFrame:
        """
        Returns historical data up to the current simulation step.
        Mimics ccxt.fetch_ohlcv.
        Resamples data if timeframe is different from base data.
        """
        # Determine lookback based on timeframe
        # Assuming base data is 1m
        lookback_multiplier = 1
        if timeframe == "5m":
            lookback_multiplier = 5
        elif timeframe == "15m":
            lookback_multiplier = 15
        elif timeframe == "1h":
            lookback_multiplier = 60
        elif timeframe == "4h":
            lookback_multiplier = 240
        elif timeframe == "1d":
            lookback_multiplier = 1440

        # We need enough raw data to generate 'limit' candles of the target timeframe
        raw_limit = limit * lookback_multiplier

        start_idx = max(0, self.current_index - raw_limit + 1)
        subset = self.data.iloc[start_idx : self.current_index + 1].copy()

        if timeframe == "1m":
            return subset

        # Resample
        subset.set_index("timestamp", inplace=True)

        # Define aggregation rules
        agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

        # Map timeframe to pandas offset alias if needed
        # Pandas 2.2+ prefers 'min' over 'm' for minutes
        pd_timeframe = timeframe.replace("m", "min") if timeframe.endswith("m") else timeframe

        # Resample to target timeframe
        resampled = subset.resample(pd_timeframe).agg(agg_rules)  # type: ignore

        # Drop incomplete last candle if it's not the end of the interval?
        # In a real simulation, we might see the "forming" candle.
        # For now, let's keep it.

        # Drop NaNs (gaps)
        resampled = resampled.dropna()

        # Reset index to get timestamp back as column
        resampled.reset_index(inplace=True)

        # Return only the requested limit
        return resampled.iloc[-limit:]

    def get_market_price(self) -> float:
        """Returns the CLOSE price of the current candle."""
        return float(self.data.iloc[self.current_index]["close"])

    def fetch_balance(self, currency: str = "USDT") -> float:
        """Returns the simulated balance."""
        return self.balance.get(currency, 0.0)

    def fetch_confirm_ohlcv(self, timeframe: str = Config.TIMEFRAME_CONFIRM, limit: int = 5) -> pd.DataFrame:
        """Fetches recent OHLCV data for confirmation (default 1m)."""
        return self.fetch_ohlcv(limit=limit, timeframe=timeframe)

    def create_entry_order(self, side: str, qty: float, current_price: float, client_oid: str) -> Dict[str, Any]:
        """
        Simulates a Limit Order Entry.
        In backtesting, we assume fill at the current price (or next open).
        For simplicity and "close to real world" approximation without tick data:
        We fill at the provided 'current_price' immediately.
        """
        # Normalize side
        side = side.upper()
        if side == "LONG":
            side = "BUY"
        elif side == "SHORT":
            side = "SELL"

        # Validate balance
        cost = qty * current_price
        fee = cost * self.fee_rate

        if side == "BUY":
            total_cost = cost + fee
            if total_cost > self.balance["USDT"]:
                logger.warning(f"Insufficient USDT for BUY. Have {self.balance['USDT']}, need {total_cost}")
                raise Exception("Insufficient funds")
            self.balance["USDT"] -= total_cost
            self.balance[self.base_currency] += qty
        elif side == "SELL":
            if qty > self.balance[self.base_currency]:
                logger.warning(
                    f"Insufficient {self.base_currency} for SELL. Have {self.balance[self.base_currency]}, need {qty}"
                )
                raise Exception("Insufficient funds")
            self.balance[self.base_currency] -= qty
            # Fee is deducted from proceeds
            self.balance["USDT"] += cost - fee

        # Record Trade
        trade = {
            "id": f"sim_order_{len(self.orders)}",
            "timestamp": self.get_current_time(),
            "symbol": Config.SYMBOL,
            "side": side,
            "price": current_price,
            "amount": qty,
            "cost": cost,
            "fee": fee,
            "client_oid": client_oid,
            "status": "closed",
            "filled": qty,
        }
        self.orders.append(trade)
        self.trades.append(trade)

        logger.info(f"SIMULATED ORDER: {side} {qty} @ {current_price} | Balance: {self.balance}")

        return trade

    def get_current_orderbook(self) -> Dict[str, Any]:
        """
        Generates a Synthetic Order Book based on 'Noisy Money Flow'.

        Logic:
        1. Spread: Derived from Volatility (High - Low) / Close.
        2. Depth: Derived from Volume.
        3. OBI: Derived from Money Flow Multiplier + Gaussian Noise.
        """
        import random

        candle = self.get_current_candle()
        close = float(candle["close"])
        high = float(candle["high"])
        low = float(candle["low"])
        volume = float(candle["volume"])

        # 1. Spread Simulation
        # Base spread 0.01% + Volatility Component
        volatility = (high - low) / close if close > 0 else 0
        spread_pct = 0.0001 + (volatility * 0.1)  # Scale down volatility impact

        # 2. Depth Simulation
        # Base depth on volume
        depth_vol = volume * 0.5  # Assume 50% of volume is visible in top book

        # 3. OBI Simulation (Money Flow + Noise)
        # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
        # Range: -1 to 1
        range_len = high - low
        if range_len > 0:
            mfm = ((close - low) - (high - close)) / range_len
        else:
            mfm = 0.0

        # Add Noise (Std Dev 0.2)
        noise = random.gauss(0, 0.2)
        simulated_obi = mfm + noise

        # Clamp OBI to -1 to 1
        simulated_obi = max(-0.99, min(0.99, simulated_obi))

        # Calculate Bid/Ask Volume from OBI
        # OBI = (Bid - Ask) / (Bid + Ask)
        # Let Total = Bid + Ask = depth_vol
        # Bid - Ask = OBI * Total
        # Bid + Ask = Total
        # 2 * Bid = Total * (1 + OBI) -> Bid = Total * (1 + OBI) / 2
        # 2 * Ask = Total * (1 - OBI) -> Ask = Total * (1 - OBI) / 2

        bid_vol = depth_vol * (1 + simulated_obi) / 2
        ask_vol = depth_vol * (1 - simulated_obi) / 2

        # Construct Order Book
        best_bid = close * (1 - spread_pct / 2)
        best_ask = close * (1 + spread_pct / 2)

        return {"bids": [[best_bid, bid_vol]], "asks": [[best_ask, ask_vol]]}

    def create_exit_order(
        self, side: str, qty: float, reason: str, related_oid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulates a Market Order Exit.
        Fills at current market price.
        """
        price = self.get_market_price()

        # Validate balance
        cost = qty * price
        fee = cost * self.fee_rate

        if side.upper() == "BUY":  # Closing a short (not implemented in strategy but good to have)
            total_cost = cost + fee
            if total_cost > self.balance["USDT"]:
                raise Exception("Insufficient funds")
            self.balance["USDT"] -= total_cost
            self.balance[self.base_currency] += qty
        elif side.upper() == "SELL":  # Closing a long
            if qty > self.balance[self.base_currency]:
                # Allow small precision errors? No, be strict.
                if qty > self.balance[self.base_currency] * 1.0001:
                    raise Exception(
                        f"Insufficient {self.base_currency}. Have {self.balance[self.base_currency]}, need {qty}"
                    )
                qty = self.balance[self.base_currency]  # Cap at max

            self.balance[self.base_currency] -= qty
            # Fee is deducted from proceeds
            self.balance["USDT"] += cost - fee

        extra = {}
        if related_oid:
            extra["related_oid"] = related_oid

        trade = {
            "id": f"sim_exit_{len(self.orders)}",
            "timestamp": self.get_current_time(),
            "symbol": Config.SYMBOL,
            "side": side,
            "price": price,
            "amount": qty,
            "cost": cost,
            "fee": fee,
            "reason": reason,
            "status": "closed",
            "filled": qty,
            "extra": extra if extra else None,
        }
        self.orders.append(trade)
        self.trades.append(trade)

        logger.info(f"SIMULATED EXIT: {side} {qty} @ {price} ({reason}) | Balance: {self.balance}")

        return trade
