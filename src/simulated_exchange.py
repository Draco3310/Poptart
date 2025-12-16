import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class SimulatedKrakenExchange:
    """
    Simulated Kraken Exchange for Backtesting.
    Replays historical data and simulates order execution for multiple pairs.
    """

    def __init__(
        self,
        data_map: Dict[str, pd.DataFrame],
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
    ):
        """
        Args:
            data_map: Dictionary mapping symbol (e.g. "XRP/USDT") to its DataFrame.
                      DataFrames must be aligned by timestamp.
            initial_balance: Starting USDT balance.
            fee_rate: Trading fee rate (0.1% default).
        """
        self.fee_rate = fee_rate
        self.data_map = data_map
        self.symbols = list(data_map.keys())
        
        if not self.symbols:
            raise ValueError("No data provided to SimulatedExchange")

        # Validate alignment
        first_len = len(data_map[self.symbols[0]])
        for sym in self.symbols:
            if len(data_map[sym]) != first_len:
                logger.warning(f"Data length mismatch for {sym}: {len(data_map[sym])} vs {first_len}")

        self.current_index = 0
        self.balance = {"USDT": initial_balance}
        for sym in self.symbols:
            base = sym.split("/")[0]
            self.balance[base] = 0.0
            
        self.initial_balance = initial_balance

        # Track orders and trades
        self.orders: list[Dict[str, Any]] = []
        self.trades: list[Dict[str, Any]] = []

        # Cache for resampled dataframes
        # Structure: {symbol: {timeframe: dataframe}}
        self.resampled_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._precalculate_resampled_data()

        logger.info(f"Simulated Exchange initialized with {len(self.symbols)} pairs.")
        logger.info(f"Initial Balance: {initial_balance} USDT")

    def _precalculate_resampled_data(self) -> None:
        """Pre-calculates resampled dataframes for common timeframes to avoid O(N^2) overhead."""
        logger.info("Pre-calculating resampled dataframes...")
        common_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        
        agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

        for symbol in self.symbols:
            self.resampled_cache[symbol] = {}
            df = self.data_map[symbol].copy()
            df.set_index("timestamp", inplace=True)
            
            for tf in common_timeframes:
                try:
                    pd_tf = tf.replace("m", "min") if tf.endswith("m") else tf
                    resampled = df.resample(pd_tf).agg(agg_rules)
                    resampled = resampled.dropna()
                    resampled.reset_index(inplace=True)
                    self.resampled_cache[symbol][tf] = resampled
                except Exception as e:
                    logger.warning(f"Failed to pre-calculate {tf} for {symbol}: {e}")

    def reset(self) -> None:
        """Resets the simulation to the beginning."""
        self.current_index = 0
        self.balance = {"USDT": self.initial_balance}
        for sym in self.symbols:
            base = sym.split("/")[0]
            self.balance[base] = 0.0
        self.orders = []
        self.trades = []

    def step(self) -> bool:
        """Advances the simulation by one candle."""
        # Check if any symbol has data left
        # We assume aligned data, so checking one is enough
        if self.current_index < len(self.data_map[self.symbols[0]]) - 1:
            self.current_index += 1
            return True
        return False

    def get_current_time(self) -> pd.Timestamp:
        """Returns the timestamp of the current candle."""
        # Use first symbol as reference
        return self.data_map[self.symbols[0]].iloc[self.current_index]["timestamp"]

    async def fetch_ohlcv(
        self, limit: int = 300, timeframe: str = Config.TIMEFRAME_PRIMARY, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Returns historical data up to the current simulation step.
        Mimics ccxt.fetch_ohlcv (Async).
        """
        target_symbol = symbol or Config.SYMBOL
        if target_symbol not in self.data_map:
            logger.error(f"Symbol {target_symbol} not found in simulation data.")
            return pd.DataFrame()

        df = self.data_map[target_symbol]

        current_time = self.get_current_time()

        if timeframe == "1m":
            # For 1m, we can just slice the original dataframe
            # We need 'limit' rows ending at current_index
            start_idx = max(0, self.current_index - limit + 1)
            return df.iloc[start_idx : self.current_index + 1].copy()

        # Check cache
        if symbol in self.resampled_cache and timeframe in self.resampled_cache[symbol]:
            cached_df = self.resampled_cache[symbol][timeframe]
            
            # Optimization: Use searchsorted instead of boolean indexing
            # Boolean indexing is O(N), which becomes O(N^2) over the simulation loop.
            # searchsorted is O(log N).
            
            # Find the insertion point for current_time to include it
            # side='right' returns index i such that a[:i] <= x < a[i:]
            idx = cached_df["timestamp"].searchsorted(current_time, side='right')
            
            # We want rows up to idx (exclusive of idx because it's 'right', so it points to next element)
            # But wait, if current_time matches exactly, 'right' puts it after.
            # So cached_df.iloc[:idx] includes the match.
            # If current_time is between candles, 'right' puts it after the previous one?
            # Example: timestamps [10:00, 10:05]. current_time 10:03.
            # searchsorted(10:03, right) -> index 1 (10:05).
            # So [:1] gives [10:00]. Correct.
            # Example: current_time 10:05.
            # searchsorted(10:05, right) -> index 2 (after 10:05).
            # So [:2] gives [10:00, 10:05]. Correct.
            
            if idx == 0:
                return pd.DataFrame(columns=cached_df.columns)
                
            start_idx = max(0, idx - limit)
            return cached_df.iloc[start_idx:idx].copy()

        # Fallback: On-the-fly resampling (Slow!)
        logger.warning(f"Timeframe {timeframe} not cached for {symbol}. Performing slow on-the-fly resampling.")
        
        # Determine lookback based on timeframe
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            if unit == 'm':
                lookback_multiplier = value
            elif unit == 'h':
                lookback_multiplier = value * 60
            elif unit == 'd':
                lookback_multiplier = value * 1440
            else:
                lookback_multiplier = 1
        except ValueError:
            lookback_multiplier = 1

        raw_limit = limit * lookback_multiplier
        start_idx = max(0, self.current_index - raw_limit + 1)
        subset = df.iloc[start_idx : self.current_index + 1].copy()
        
        subset.set_index("timestamp", inplace=True)
        agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        pd_timeframe = timeframe.replace("m", "min") if timeframe.endswith("m") else timeframe
        resampled = subset.resample(pd_timeframe).agg(agg_rules)
        resampled = resampled.dropna()
        resampled.reset_index(inplace=True)
        
        return resampled.iloc[-limit:]

    async def fetch_confirm_ohlcv(
        self, symbol: Optional[str] = None, timeframe: str = Config.TIMEFRAME_CONFIRM, limit: int = 5
    ) -> pd.DataFrame:
        """Fetches recent OHLCV data for confirmation (default 1m)."""
        return await self.fetch_ohlcv(limit=limit, timeframe=timeframe, symbol=symbol)

    async def get_market_price(self, symbol: Optional[str] = None) -> float:
        """Returns the CLOSE price of the current candle."""
        target_symbol = symbol or Config.SYMBOL
        if target_symbol not in self.data_map:
            return 0.0
        return float(self.data_map[target_symbol].iloc[self.current_index]["close"])

    async def fetch_balance(self, currency: str = "USDT") -> float:
        """Returns the simulated balance."""
        return self.balance.get(currency, 0.0)
    
    async def fetch_total_equity(self) -> float:
        """Calculates total equity (USDT + Value of Holdings)."""
        equity = self.balance.get("USDT", 0.0)
        for sym in self.symbols:
            base = sym.split("/")[0]
            qty = self.balance.get(base, 0.0)
            if qty > 0:
                price = await self.get_market_price(sym)
                equity += qty * price
        return equity

    async def fetch_order_book(
        self, symbol: Optional[str] = None, limit: int = Config.ORDER_BOOK_DEPTH
    ) -> Dict[str, Any]:
        """
        Generates a Synthetic Order Book based on 'Noisy Money Flow'.
        """
        target_symbol = symbol or Config.SYMBOL
        if target_symbol not in self.data_map:
            return {"bids": [], "asks": []}

        candle = self.data_map[target_symbol].iloc[self.current_index]
        close = float(candle["close"])
        high = float(candle["high"])
        low = float(candle["low"])
        volume = float(candle["volume"])

        # 1. Spread Simulation
        volatility = (high - low) / close if close > 0 else 0
        spread_pct = 0.0001 + (volatility * 0.1)

        # 2. Depth Simulation
        depth_vol = volume * 0.5

        # 3. OBI Simulation
        range_len = high - low
        if range_len > 0:
            mfm = ((close - low) - (high - close)) / range_len
        else:
            mfm = 0.0

        noise = random.gauss(0, 0.2)
        simulated_obi = max(-0.99, min(0.99, mfm + noise))

        bid_vol = depth_vol * (1 + simulated_obi) / 2
        ask_vol = depth_vol * (1 - simulated_obi) / 2

        best_bid = close * (1 - spread_pct / 2)
        best_ask = close * (1 + spread_pct / 2)

        return {"bids": [[best_bid, bid_vol]], "asks": [[best_ask, ask_vol]]}

    async def create_entry_order(
        self, side: str, qty: float, current_price: float, client_oid: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simulates a Limit Order Entry."""
        target_symbol = symbol or Config.SYMBOL
        base_currency = target_symbol.split("/")[0]
        
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
                # In simulation, we can raise or just return failed order. 
                # Raising mimics exchange error.
                # But let's just cap it? No, strategy should handle sizing.
                # We'll assume strict check.
                return {"id": "failed", "status": "rejected"}
                
            self.balance["USDT"] -= total_cost
            self.balance[base_currency] += qty
        elif side == "SELL":
            if qty > self.balance[base_currency]:
                logger.warning(f"Insufficient {base_currency} for SELL.")
                return {"id": "failed", "status": "rejected"}
                
            self.balance[base_currency] -= qty
            self.balance["USDT"] += cost - fee

        # Record Trade
        trade = {
            "id": f"sim_order_{len(self.orders)}",
            "timestamp": self.get_current_time(),
            "symbol": target_symbol,
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

        logger.info(f"SIMULATED ORDER: {side} {qty} {target_symbol} @ {current_price} | Balance: {self.balance}")

        return trade

    async def create_exit_order(
        self, side: str, qty: float, reason: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simulates a Market Order Exit."""
        target_symbol = symbol or Config.SYMBOL
        base_currency = target_symbol.split("/")[0]
        
        price = await self.get_market_price(target_symbol)

        # Validate balance
        cost = qty * price
        fee = cost * self.fee_rate

        if side.upper() == "BUY":  # Closing a short
            total_cost = cost + fee
            if total_cost > self.balance["USDT"]:
                 return {"id": "failed", "status": "rejected"}
            self.balance["USDT"] -= total_cost
            self.balance[base_currency] += qty
        elif side.upper() == "SELL":  # Closing a long
            if qty > self.balance[base_currency]:
                # Allow tiny precision error
                if qty > self.balance[base_currency] * 1.0001:
                     return {"id": "failed", "status": "rejected"}
                qty = self.balance[base_currency]

            self.balance[base_currency] -= qty
            self.balance["USDT"] += cost - fee

        trade = {
            "id": f"sim_exit_{len(self.orders)}",
            "timestamp": self.get_current_time(),
            "symbol": target_symbol,
            "side": side,
            "price": price,
            "amount": qty,
            "cost": cost,
            "fee": fee,
            "reason": reason,
            "status": "closed",
            "filled": qty,
        }
        self.orders.append(trade)
        self.trades.append(trade)

        logger.info(f"SIMULATED EXIT: {side} {qty} {target_symbol} @ {price} ({reason}) | Balance: {self.balance}")

        return trade
    
    async def fetch_recent_trades(self, symbol: Optional[str] = None, limit: int = 1) -> list[Dict[str, Any]]:
        """Fetches recent trades (mocked)."""
        return []
