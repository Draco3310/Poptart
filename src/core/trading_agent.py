import asyncio
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config, PairConfig, get_data_path, get_model_path
from src.core.feature_engine import FeatureEngine
from src.core.risk_manager import RiskManager
from src.database import Database
from src.exchange import KrakenExchange
from src.notifier import TelegramNotifier
from src.predictors.orchestrator import PredictorOrchestrator
from src.strategies.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)

class TradingAgent:
    """
    Autonomous Trading Agent for a specific cryptocurrency pair.
    Encapsulates state, data ingestion, feature engineering, and strategy execution.
    """

    def __init__(
        self,
        pair_config: PairConfig,
        exchange: KrakenExchange,
        notifier: TelegramNotifier,
        risk_manager: RiskManager,
        db: Optional[Database] = None,
        run_id: Optional[str] = None
    ):
        self.config = pair_config
        self.symbol = pair_config.symbol
        self.exchange = exchange
        self.notifier = notifier
        self.risk_manager = risk_manager
        self.db = db
        self.run_id = run_id
        self.logger = logging.getLogger(f"Agent-{self.symbol}")

        # dedicated components
        self.feature_engine = FeatureEngine()
        self.strategy_selector = self._init_strategy_selector()
        self.predictor_orchestrator = self._init_predictors()

        # State
        self.position: Optional[Dict[str, Any]] = None
        self.historic_data = pd.DataFrame()
        self.historic_confirm_data = pd.DataFrame()
        self.historic_btc_data = pd.DataFrame() # For context
        self.last_dca_time: Optional[pd.Timestamp] = None

    def _init_strategy_selector(self) -> StrategySelector:
        """Initializes the strategy selector with pair-specific models."""
        symbol_clean = self.symbol.replace("/", "")
        regime_path = get_model_path(symbol_clean, "rf_regime", ext=".joblib")

        # Skip Regime Model for BTC (SmartDCA doesn't use it)
        if "BTC" in self.symbol:
            regime_path = None

        if not Config.ML_ENABLED:
            regime_path = None

        return StrategySelector(regime_model_path=regime_path)

    def _init_predictors(self) -> PredictorOrchestrator:
        """Initializes ML predictors for this pair."""
        orchestrator = PredictorOrchestrator()

        # Skip ML for BTC/USDT as it uses SmartDCA
        if "BTC" in self.symbol:
            return orchestrator

        if Config.ML_ENABLED:
            symbol_clean = self.symbol.replace("/", "")
            xgb_path = get_model_path(symbol_clean, "xgb_tp_sl_H1", ext=".model")
            rf_path = get_model_path(symbol_clean, "rf_tp_sl_H1", ext=".joblib")

            predictor_config = [
                {"type": "xgboost", "path": xgb_path, "weight": 0.5},
                {"type": "rf", "path": rf_path, "weight": 0.5},
            ]
            orchestrator.load_predictors(predictor_config)

        return orchestrator

    async def initialize(self) -> None:
        """Loads historic data and recovers position."""
        self.logger.info(f"Initializing Agent for {self.symbol}...")

        # 1. Recover Position
        self.position = await self._recover_position()
        if self.position:
            msg = f"🔄 {self.symbol} Position Recovered: {self.position['qty']} @ {self.position['entry_price']}"
            await self.notifier.send_message(msg)

        # 2. Load Historic Data (Critical for MTF)
        self.historic_data = self._load_historic_data(self.symbol, Config.TIMEFRAME_PRIMARY)
        self.historic_confirm_data = self._load_historic_data(self.symbol, Config.TIMEFRAME_CONFIRM)
        self.historic_btc_data = self._load_historic_data("BTC/USDT", Config.TIMEFRAME_PRIMARY)

        self.logger.info(f"Initialization Complete. Historic Rows: {len(self.historic_data)}")

    async def tick(self) -> None:
        """Executes one trading cycle."""
        try:
            # A. Fetch Data
            self.logger.debug(f"Fetching data for {self.symbol}...")

            # Primary Data
            raw_data = await self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=Config.TIMEFRAME_PRIMARY,
                limit=3000
            )

            # Merge with Historic (Optimized Rolling Buffer)
            if not self.historic_data.empty:
                # Append new data only
                last_ts = self.historic_data["timestamp"].iloc[-1]
                new_rows = raw_data[raw_data["timestamp"] > last_ts]
                if not new_rows.empty:
                    self.historic_data = pd.concat([self.historic_data, new_rows])
                    
                # Maintain Buffer
                if len(self.historic_data) > Config.LIVE_DATA_BUFFER_SIZE:
                    self.historic_data = self.historic_data.iloc[-Config.LIVE_DATA_BUFFER_SIZE:]
                
                raw_data = self.historic_data.copy()
            else:
                self.historic_data = raw_data

            # Confirmation Data
            confirm_data = await self.exchange.fetch_confirm_ohlcv(symbol=self.symbol)

            if not self.historic_confirm_data.empty:
                last_ts = self.historic_confirm_data["timestamp"].iloc[-1]
                new_rows = confirm_data[confirm_data["timestamp"] > last_ts]
                if not new_rows.empty:
                    self.historic_confirm_data = pd.concat([self.historic_confirm_data, new_rows])
                
                if len(self.historic_confirm_data) > Config.LIVE_DATA_BUFFER_SIZE_CONFIRM:
                    self.historic_confirm_data = self.historic_confirm_data.iloc[-Config.LIVE_DATA_BUFFER_SIZE_CONFIRM:]
                
                confirm_data = self.historic_confirm_data.copy()
            else:
                self.historic_confirm_data = confirm_data

            # BTC Data (Context)
            btc_data = await self.exchange.fetch_ohlcv(
                symbol="BTC/USDT",
                timeframe=Config.TIMEFRAME_PRIMARY
            )

            if not self.historic_btc_data.empty:
                last_ts = self.historic_btc_data["timestamp"].iloc[-1]
                new_rows = btc_data[btc_data["timestamp"] > last_ts]
                if not new_rows.empty:
                    self.historic_btc_data = pd.concat([self.historic_btc_data, new_rows])
                
                if len(self.historic_btc_data) > Config.LIVE_DATA_BUFFER_SIZE:
                    self.historic_btc_data = self.historic_btc_data.iloc[-Config.LIVE_DATA_BUFFER_SIZE:]
                
                btc_data = self.historic_btc_data.copy()
            else:
                self.historic_btc_data = btc_data

            if raw_data.empty:
                self.logger.warning("No primary data fetched.")
                return

            # B. Feature Engineering
            context: Dict[str, Any] = {"pair_config": self.config}
            if not btc_data.empty:
                btc_copy = btc_data.copy()
                if "timestamp" in btc_copy.columns:
                    btc_copy.set_index("timestamp", inplace=True)
                context["btc_df"] = btc_copy

            enriched_data = self.feature_engine.compute_features(raw_data, context=context)

            if enriched_data.empty or len(enriched_data) < 50:
                self.logger.warning("Insufficient data after feature engineering.")
                return

            enriched_confirm = self.feature_engine.compute_features(confirm_data)

            # C. Regime & ML
            # Fetch real-time data for execution
            order_book = await self.exchange.fetch_order_book(symbol=self.symbol)
            l2_features = self.feature_engine.compute_l2_features(order_book)
            current_price = await self.exchange.get_market_price(symbol=self.symbol)

            # DCA Balances
            current_base_bal = 0.0
            current_equity = 0.0
            if self.config.enable_dca_mode:
                base_currency = self.symbol.replace("USDT", "").replace("/", "")
                current_base_bal = await self.exchange.fetch_balance(base_currency)
                usdt_bal = await self.exchange.fetch_balance("USDT")
                current_equity = usdt_bal + (current_base_bal * current_price)

            await self.process_tick(
                enriched_data,
                enriched_confirm,
                l2_features,
                current_price,
                current_base_bal,
                current_equity
            )

        except Exception as e:
            self.logger.error(f"Error in tick: {e}", exc_info=True)

    async def process_tick(
        self,
        enriched_data: pd.DataFrame,
        enriched_confirm: pd.DataFrame,
        l2_features: Dict[str, Any],
        current_price: float,
        current_base_bal: float,
        current_equity: float
    ) -> Dict[str, Any]:
        """
        Core logic for analyzing data and executing trades.
        Separated for easier backtesting with pre-computed features.
        """
        # 1. Regime & ML
        # Check for pre-computed values (Optimization)
        if "regime" in enriched_data.columns and "ml_score" in enriched_data.columns:
            # Use the last row's pre-computed values
            last_row = enriched_data.iloc[-1]
            regime_name = last_row["regime"]
            ml_score = last_row["ml_score"]
            # Update state for consistency
            # self.strategy_selector.last_regime = ... (Enum conversion needed if we use it elsewhere)
        else:
            # Real-time Inference
            regime_enum = self.strategy_selector.regime_classifier.predict(
                enriched_data, self.strategy_selector.last_regime
            )
            self.strategy_selector.last_regime = regime_enum
            regime_name = regime_enum.name

            ml_score = None
            if Config.ML_ENABLED:
                ml_score = self.predictor_orchestrator.get_ensemble_score(enriched_data, regime=regime_name)

        # 2. Strategy Analysis
        dca_config = {}
        if self.config.enable_dca_mode:
            dca_config = {
                "target_allocation": self.config.dca_target_allocation,
                "dip_threshold_rsi": self.config.dca_dip_threshold_rsi,
                "notional_per_trade": self.config.dca_notional_per_trade,
            }

        analysis = self.strategy_selector.analyze(
            enriched_data,
            pair_config=self.config,
            ml_score=ml_score,
            confirm_df=enriched_confirm,
            l2_features=l2_features,
            regime=regime_name,
            current_price=current_price,
            enable_mean_reversion=self.config.enable_mean_reversion,
            enable_trend_following=self.config.enable_trend_following,
            enable_dca_mode=self.config.enable_dca_mode,
            dca_config=dca_config,
            current_balance_btc=current_base_bal,
            current_equity_usdt=current_equity
        )

        # Log Decision to DB
        if self.db and self.run_id:
            self.db.log_decision(self.run_id, analysis)

        signal = analysis["signal"]
        multiplier = analysis.get("size_multiplier", 0.0)
        atr = analysis.get("atr", 0.0)

        # 3. Execution
        await self._handle_execution(signal, analysis, current_price, atr, multiplier, regime_name)
        
        return analysis

    async def _handle_execution(
        self, signal: str, analysis: Dict, current_price: float, atr: float, multiplier: float, regime_name: str
    ):
        """Handles entry and exit logic."""

        # 1. Exits
        if self.position:
            await self._check_exit(analysis, current_price)

        # 2. Entries
        # Allow entry if no position OR if DCA mode is enabled
        can_enter = not self.position or self.config.enable_dca_mode

        # Check DCA Interval
        if self.config.enable_dca_mode and self.config.dca_interval_minutes:
            # Get timestamp from analysis (injected in backtest) or use current time
            current_ts = analysis.get("timestamp", pd.Timestamp.now())
            
            if self.last_dca_time:
                elapsed = (current_ts - self.last_dca_time).total_seconds() / 60.0
                if elapsed < self.config.dca_interval_minutes:
                    can_enter = False

        if can_enter and signal == "LONG":
            await self._execute_entry(signal, analysis, current_price, atr, multiplier, regime_name)
            if self.config.enable_dca_mode:
                self.last_dca_time = analysis.get("timestamp", pd.Timestamp.now())

    async def _check_exit(self, analysis: Dict, current_price: float):
        """Checks and executes exits."""
        if not self.position:
            return

        sl_price = self.position.get("stop_loss", 0.0)
        exit_signal = False
        exit_reason = ""

        # Hard SL
        if self.position["side"] == "LONG" and current_price <= sl_price and sl_price > 0:
            exit_signal = True
            exit_reason = "Stop Loss"

        # Strategy Exit
        pos_strategy = self.position.get("strategy", "MeanReversion")
        updates = {}

        if pos_strategy == "TrendFollowing":
            updates = self.strategy_selector.trend_following.get_exit_updates(self.position, analysis, self.config)
        elif pos_strategy == "BTCSmartDCA":
            updates = self.strategy_selector.btc_dca.get_exit_updates(self.position, analysis, self.config)
        else:
            updates = self.strategy_selector.mean_reversion.get_exit_updates(self.position, analysis, self.config)

        if updates.get("exit_signal"):
            exit_signal = True
            exit_reason = updates.get("exit_reason", "Strategy Exit")

        # Trailing Stop Update
        if updates.get("new_sl"):
            if updates["new_sl"] > self.position["stop_loss"]:
                self.position["stop_loss"] = updates["new_sl"]
                self.logger.info(f"Trailing Stop Updated: {self.position['stop_loss']:.4f}")

        # Execute Exit
        if exit_signal:
            self.logger.info(f"Executing Exit: {exit_reason}")
            await self.exchange.create_exit_order("SELL", self.position["qty"], exit_reason, symbol=self.symbol)
            await self.notifier.notify_exit(f"{self.symbol} {exit_reason}", current_price, 0.0)
            self.position = None

        # Signal Flip (Long -> Short not supported, but Long -> Exit is)
        elif analysis["signal"] == "SHORT":
             self.logger.info("Signal Flip (Long -> Short). Exiting Long.")
             await self.exchange.create_exit_order("SELL", self.position["qty"], "Signal Flip", symbol=self.symbol)
             await self.notifier.notify_exit(f"{self.symbol} Signal Flip", current_price, 0.0)
             self.position = None

    async def _execute_entry(
        self, signal: str, analysis: Dict, current_price: float, atr: float, multiplier: float, regime_name: str
    ):
        """Executes entry orders."""
        # Use Total Equity for Risk Management (Circuit Breaker & Sizing)
        total_equity = await self.exchange.fetch_total_equity()

        if self.risk_manager.check_circuit_breaker(total_equity):
            self.logger.warning("Circuit Breaker Active. Skipping Entry.")
            return

        # Available Liquidity
        usdt_balance = await self.exchange.fetch_balance("USDT")

        qty = 0.0
        if self.config.enable_dca_mode:
            # Get shortfall from analysis context
            shortfall_usdt = analysis.get("decision_context", {}).get("shortfall_usdt", 0.0)
            qty = self.strategy_selector.btc_dca.calculate_position_size(self.config, current_price, shortfall_usdt)
        else:
            # Size based on Total Equity
            qty = self.risk_manager.calculate_size(total_equity, current_price, atr, self.config, multiplier, regime_name)

        # Cap at available USDT
        if (qty * current_price) > usdt_balance:
            self.logger.info(f"Capping size to available USDT: {usdt_balance:.2f}")
            qty = usdt_balance / current_price * 0.99

        if qty > 0:
            client_oid = f"sentinel_{self.symbol.replace('/', '')}_{int(asyncio.get_event_loop().time())}"
            order = await self.exchange.create_entry_order(signal, qty, current_price, client_oid, symbol=self.symbol)

            sl_dist = atr * self.config.atr_multiplier
            stop_loss = current_price - sl_dist
            
            # Disable SL for DCA Mode
            if self.config.enable_dca_mode:
                stop_loss = 0.0

            if self.position and self.config.enable_dca_mode:
                # Accumulate
                new_qty = self.position["qty"] + qty
                current_val = self.position["qty"] * self.position["entry_price"]
                new_val = qty * current_price
                new_avg_price = (current_val + new_val) / new_qty

                self.position["qty"] = new_qty
                self.position["entry_price"] = new_avg_price
                self.position["stop_loss"] = 0.0
            else:
                self.position = {
                    "side": "LONG",
                    "qty": qty,
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "tp1_hit": False,
                    "trade_id": order.get("id", client_oid),
                    "strategy": analysis.get("active_strategy", "MeanReversion")
                }

            await self.notifier.notify_entry(f"{self.symbol} {signal}", current_price, qty, stop_loss)
        else:
            self.logger.warning("Calculated size is 0. Skipping.")

    def _load_historic_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Loads historic data from CSV."""
        try:
            # Use get_data_path to resolve correct path (handles mapping)
            file_path = get_data_path(symbol, timeframe)
            
            if not file_path.exists():
                # Fallback: Try directory search if exact match fails
                # This handles cases where file naming might differ slightly or we want newest
                from src.config import SYMBOL_MAP
                short_name = SYMBOL_MAP.get(symbol, symbol.split("/")[0])
                data_dir = f"data/{short_name}"
                
                if not os.path.exists(data_dir):
                    return pd.DataFrame()

                files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and timeframe in f]
                if not files:
                    return pd.DataFrame()

                # Sort by modification time (newest first)
                files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
                file_path = os.path.join(data_dir, files[0])
            
            self.logger.info(f"Loading historic data from {file_path}...")
            df = pd.read_csv(file_path)

            df.columns = [c.lower() for c in df.columns]
            if "timestamp" not in df.columns and "date" in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if "open_time" in df.columns:
                df.drop(columns=["open_time"], inplace=True)

            df.sort_values("timestamp", inplace=True)
            return df

        except Exception as e:
            self.logger.error(f"Failed to load historic data: {e}")
            return pd.DataFrame()

    async def _recover_position(self) -> Optional[Dict[str, Any]]:
        """Recovers position from exchange."""
        try:
            base_currency = self.symbol.replace("USDT", "").replace("/", "")
            qty = await self.exchange.fetch_balance(base_currency)

            if qty > 1.0: # Threshold
                self.logger.info(f"Recovering position: Found {qty} {base_currency}")
                trades = await self.exchange.fetch_recent_trades(symbol=self.symbol, limit=1)
                entry_price = 0.0
                if trades:
                    last_trade = trades[0]
                    if last_trade["side"] == "buy":
                        entry_price = float(last_trade["price"])
                    else:
                        entry_price = await self.exchange.get_market_price(symbol=self.symbol)
                else:
                    entry_price = await self.exchange.get_market_price(symbol=self.symbol)

                return {
                    "side": "LONG",
                    "qty": qty,
                    "entry_price": entry_price,
                    "stop_loss": 0.0,
                    "tp1_hit": False,
                    "trade_id": "recovered",
                    "strategy": "Unknown"
                }
            return None
        except Exception as e:
            self.logger.error(f"Failed to recover position: {e}")
            return None
