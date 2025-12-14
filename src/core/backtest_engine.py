import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from src.config import Config, PairConfig, get_data_path, get_model_path
from src.core.backtest_result import BacktestResult
from src.core.feature_engine import FeatureEngine
from src.core.risk_manager import RiskManager
from src.predictors.orchestrator import PredictorOrchestrator
from src.simulated_exchange import SimulatedKrakenExchange
from src.strategies.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Core simulation engine for backtesting.
    Decoupled from reporting and file I/O.
    """

    def __init__(
        self,
        pair_config: PairConfig,
        config_overrides: Optional[Dict[str, Any]] = None,
        btc_data: Optional[pd.DataFrame] = None,
    ):
        self.pair_config = pair_config
        self.config_overrides = config_overrides or {}
        self.btc_data = btc_data
        self.logger = logging.getLogger(f"BacktestEngine_{pair_config.symbol}")

    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> BacktestResult:
        """
        Executes the backtest simulation.
        """
        # 1. Apply Config Overrides (Patching Global Config)
        # NOTE: In a threaded environment, this is unsafe. Use multiprocessing.
        original_config = {}

        # Apply Pair Config Defaults
        Config.SYMBOL = self.pair_config.symbol
        Config.ATR_MULTIPLIER = self.pair_config.atr_multiplier
        Config.MAX_VOLATILITY_THRESHOLD = self.pair_config.max_volatility_threshold
        Config.MIN_TRADE_COOLDOWN_MINUTES = self.pair_config.cooldown_minutes
        Config.ML_TREND_LONG_THRESHOLD = self.pair_config.ml_threshold_trend
        Config.EMA_PERIOD_SLOW = self.pair_config.ema_period_slow
        Config.EMA_PERIOD_FAST = self.pair_config.ema_period_fast
        # Use Strategy Threshold if defined, else fallback to Regime Threshold
        Config.ADX_THRESHOLD = self.pair_config.adx_threshold_strategy or self.pair_config.adx_threshold

        # Trend Strategy Params
        if hasattr(self.pair_config, "trend_trailing_stop_type"):
            Config.TREND_TRAILING_STOP_TYPE = self.pair_config.trend_trailing_stop_type
        if hasattr(self.pair_config, "trend_trailing_stop_multiplier"):
            Config.TREND_TRAILING_STOP_MULTIPLIER = self.pair_config.trend_trailing_stop_multiplier
        if hasattr(self.pair_config, "trend_max_extension"):
            Config.TREND_MAX_EXTENSION = self.pair_config.trend_max_extension
        if hasattr(self.pair_config, "trend_rsi_max"):
            Config.TREND_RSI_MAX = self.pair_config.trend_rsi_max

        Config.DRY_RUN = False

        # Apply Explicit Overrides
        for key, value in self.config_overrides.items():
            applied = False
            # 1. Override Global Config
            if hasattr(Config, key):
                original_config[key] = getattr(Config, key)
                setattr(Config, key, value)
                applied = True

            # 2. Override Pair Config (Instance)
            if hasattr(self.pair_config, key):
                setattr(self.pair_config, key, value)
                applied = True

            if not applied:
                self.logger.warning(f"Config key {key} not found in Config or PairConfig, skipping override.")

        # Determine Data Path
        data_path = str(get_data_path(self.pair_config.symbol, "1m"))
        if not os.path.exists(data_path):
            self.logger.error(f"Data file not found: {data_path}")
            return self._create_empty_result(start_date, end_date)

        # Initialize Components
        try:
            random.seed(42)
            np.random.seed(42)

            # Calculate Data Start Date (Warmup Buffer)
            # We need at least 24h (Volume Profile) + 200 candles (EMA)
            # AND ema200_1h requires 200 hours (~8.3 days).
            # 10 days buffer should be safe.
            data_start_date = None
            if start_date:
                dt_start = pd.to_datetime(start_date)
                dt_data_start = dt_start - pd.Timedelta(days=10)
                data_start_date = dt_data_start.strftime("%Y-%m-%d %H:%M:%S")
                self.logger.info(f"Adjusting data load start to {data_start_date} for warmup.")

            base_currency = self.pair_config.symbol.replace("USDT", "")
            exchange = SimulatedKrakenExchange(
                data_path,
                initial_balance=10000.0,
                start_date=data_start_date,
                end_date=end_date,
                base_currency=base_currency,
            )
            feature_engine = FeatureEngine()
            predictor_orchestrator = PredictorOrchestrator()

            if Config.ML_ENABLED:
                self._load_ml_models(predictor_orchestrator)

            # Dynamic Regime Model Path
            symbol = self.pair_config.symbol.replace("/", "")
            regime_path = get_model_path(symbol, "rf_regime", ext=".joblib")
            strategy = StrategySelector(regime_model_path=regime_path)
            risk_manager = RiskManager()

        except Exception as e:
            self.logger.critical(f"Initialization failed: {e}")
            return self._create_empty_result(start_date, end_date)

        # Pre-compute Features
        self.logger.info("Pre-computing features...")
        enriched_1m, enriched_5m = self._compute_features(exchange, feature_engine)

        # Pre-compute BTC Trend if available
        btc_trend_series = None
        if self.btc_data is not None:
            self.logger.info("Pre-computing BTC Trend...")
            # Resample to Daily to calculate 20-Day SMA
            btc_daily = self.btc_data.set_index("timestamp").resample("D").agg({"close": "last"}).dropna()
            btc_daily["sma20"] = btc_daily["close"].rolling(window=20).mean()
            btc_daily["bullish"] = btc_daily["close"] > btc_daily["sma20"]

            # Reindex to 1m to match simulation steps (Forward Fill)
            # This is memory intensive but fast for lookup
            # Alternatively, we can lookup by date in the loop
            btc_trend_series = btc_daily["bullish"]

        # Main Loop Variables
        steps = 0
        warmup_period = 300
        current_position = None
        equity_curve = []
        decision_log: List[Dict[str, Any]] = []
        last_date = None
        last_entry_time: Optional[pd.Timestamp] = None
        last_5m_start: Optional[pd.Timestamp] = None
        cached_analysis: Optional[Dict[str, Any]] = None
        entry_fired_for_block: Optional[pd.Timestamp] = None

        # DCA Config
        dca_config = {}
        if self.pair_config.enable_dca_mode:
            dca_config = {
                "target_allocation": self.pair_config.dca_target_allocation,
                "dip_threshold_rsi": self.pair_config.dca_dip_threshold_rsi,
                "notional_per_trade": self.pair_config.dca_notional_per_trade,
            }

        initial_hold_qty = exchange.initial_balance / exchange.data.iloc[0]["close"]

        # --- Simulation Loop ---
        while exchange.step():
            steps += 1
            current_time = exchange.get_current_time_fast()
            current_close = exchange.get_current_close_fast()
            current_low = exchange.get_current_low_fast()

            # Equity Tracking
            current_date = current_time.date()
            if last_date is None:
                last_date = current_date
            if current_date != last_date:
                usdt_bal = exchange.fetch_balance("USDT")
                base_bal = exchange.fetch_balance(base_currency)
                total_equity = usdt_bal + (base_bal * current_close)
                benchmark_equity = initial_hold_qty * current_close
                equity_curve.append({"timestamp": last_date, "equity": total_equity, "benchmark": benchmark_equity})
                last_date = current_date

            if exchange.current_index < warmup_period:
                continue

            try:
                # Strategy Analysis (Cached per 5m block)
                target_5m_start = current_time.floor("5min") - pd.Timedelta(minutes=5)
                if target_5m_start not in enriched_5m.index:
                    continue

                if last_5m_start != target_5m_start:
                    # New 5m bar
                    # Lookup BTC Trend
                    btc_bullish = True  # Default to True if no BTC data
                    if btc_trend_series is not None:
                        # Lookup by Date (Daily resolution)
                        # We use the previous day's close for the signal to avoid lookahead bias?
                        # Or current day? 20-Day MA usually implies "Close > MA".
                        # If we are intraday, we should use the *previous* daily close and MA?
                        # Or calculate rolling 20-Day MA on hourly data?
                        # The user specified "BTC > BTC 20-day MA".
                        # Let's use the value for the *current date*
                        # (which is effectively "so far today" or "yesterday's close" depending on resampling).
                        # Safest is to use `current_time.floor('D')`
                        current_day = current_time.floor("D")
                        if current_day in btc_trend_series.index:
                            btc_bullish = bool(btc_trend_series.loc[current_day])
                        else:
                            # Fallback to previous day if current day not finished/available
                            prev_day = current_day - pd.Timedelta(days=1)
                            if prev_day in btc_trend_series.index:
                                btc_bullish = bool(btc_trend_series.loc[prev_day])

                    cached_analysis = self._analyze_step(
                        target_5m_start,
                        current_time,
                        enriched_5m,
                        enriched_1m,
                        strategy,
                        predictor_orchestrator,
                        feature_engine,
                        exchange,
                        dca_config,
                        current_close,
                        btc_bullish=btc_bullish,
                    )
                    last_5m_start = target_5m_start
                    entry_fired_for_block = None

                if cached_analysis is None:
                    continue

                signal = cached_analysis["signal"]
                analysis_for_exit = dict(cached_analysis)
                analysis_for_exit["close"] = current_close

                # Execution Logic
                # 1. Exits
                if current_position:
                    current_position = self._handle_exits(
                        current_position,
                        current_low,
                        current_close,
                        current_time,
                        exchange,
                        strategy,
                        analysis_for_exit,
                        decision_log,
                        cached_analysis,
                    )

                # 2. Entries
                if signal == "LONG" and entry_fired_for_block != last_5m_start:
                    entered, trade_info = self._handle_entry(
                        current_position,
                        current_time,
                        current_close,
                        exchange,
                        risk_manager,
                        cached_analysis,
                        last_entry_time,
                        decision_log,
                        dca_config,
                        strategy,
                    )
                    if entered:
                        current_position = trade_info
                        entry_fired_for_block = last_5m_start
                        last_entry_time = current_time

                elif signal == "SHORT" and current_position and current_position["side"] == "LONG":
                    # Signal Flip Exit
                    self._close_position(
                        current_position,
                        current_close,
                        current_time,
                        "Signal Flip",
                        exchange,
                        decision_log,
                        cached_analysis,
                    )
                    current_position = None

                # Log interesting skips for diagnosis
                elif signal is None:
                    ctx = cached_analysis.get("decision_context", {})
                    reason = ctx.get("reason_string", "")
                    # Log if it passed the basic trend check but failed a filter
                    if reason and "Misaligned" not in reason and "Counter-Trend" not in reason:
                        self._log_decision("SKIP", current_time, cached_analysis, decision_log)

            except Exception as e:
                self.logger.error(f"Error in step {steps}: {e}")

        # --- Finalize ---
        # Restore Config
        for key, value in original_config.items():
            setattr(Config, key, value)

        # Calculate Metrics
        final_equity = self._calculate_final_equity(exchange, base_currency)
        metrics = self._calculate_metrics(exchange, equity_curve, final_equity, initial_hold_qty)

        return BacktestResult(
            run_id=f"{self.pair_config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pair=self.pair_config.symbol,
            start_date=start_date,
            end_date=end_date,
            config=self.config_overrides,
            metrics=metrics,
            trades=exchange.trades,
            equity_curve=equity_curve,
            decision_log=decision_log,
        )

    def _load_ml_models(self, orchestrator: PredictorOrchestrator) -> None:
        models_to_load: List[Dict[str, Any]] = []
        symbol = self.pair_config.symbol.replace("/", "")

        # Dynamic Paths
        xgb_path = get_model_path(symbol, "xgb_tp_sl_H1", ext=".model")
        rf_path = get_model_path(symbol, "rf_tp_sl_H1", ext=".joblib")

        if os.path.exists(xgb_path):
            models_to_load.append({"type": "xgboost", "path": xgb_path, "weight": 0.5})

        if os.path.exists(rf_path):
            models_to_load.append({"type": "rf", "path": rf_path, "weight": 0.5})

        if models_to_load:
            orchestrator.load_predictors(models_to_load)
        else:
            self.logger.warning(f"No ML models found for {symbol}. ML will be disabled effectively.")

    def _compute_features(
        self, exchange: SimulatedKrakenExchange, feature_engine: FeatureEngine
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        full_1m = exchange.data.copy()
        full_1m.set_index("timestamp", inplace=True)

        # Prepare Context (BTC Data)
        context = {}
        if self.btc_data is not None:
            # Ensure BTC data is indexed by timestamp
            btc_df = self.btc_data.copy()
            if "timestamp" in btc_df.columns:
                btc_df.set_index("timestamp", inplace=True)
            context["btc_df"] = btc_df

        enriched_1m = feature_engine.compute_features(full_1m, context=context)

        agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        full_5m = full_1m.resample("5min").agg(agg_rules).dropna()  # type: ignore

        # For 5m features, we should ideally resample BTC data to 5m too?
        # BetaFeatureBlock handles merging. If we pass 1m BTC data to 5m SOL data,
        # the merge will fail or be sparse unless we resample BTC.
        # Let's resample BTC if present.
        context_5m = {}
        if "btc_df" in context:
            btc_5m = context["btc_df"].resample("5min").agg(agg_rules).dropna()  # type: ignore
            context_5m["btc_df"] = btc_5m

        enriched_5m = feature_engine.compute_features(full_5m, context=context_5m)
        return enriched_1m, enriched_5m

    def _analyze_step(
        self,
        target_5m_start: pd.Timestamp,
        current_time: pd.Timestamp,
        enriched_5m: pd.DataFrame,
        enriched_1m: pd.DataFrame,
        strategy: StrategySelector,
        predictor: PredictorOrchestrator,
        feature_engine: FeatureEngine,
        exchange: SimulatedKrakenExchange,
        dca_config: Dict[str, Any],
        current_close: float,
        btc_bullish: bool = True,
    ) -> Optional[Dict[str, Any]]:
        loc_5m = int(enriched_5m.index.get_loc(target_5m_start))  # type: ignore
        start_loc = max(0, loc_5m - 200 + 1)
        df_slice = enriched_5m.iloc[start_loc : loc_5m + 1]

        if current_time not in enriched_1m.index:
            return None
        loc_1m = int(enriched_1m.index.get_loc(current_time))  # type: ignore
        start_loc_1m = max(0, loc_1m - 5)
        confirm_slice = enriched_1m.iloc[start_loc_1m:loc_1m]

        # Use strategy's last regime for hysteresis
        regime_enum = strategy.regime_classifier.predict(
            df_slice, strategy.last_regime, adx_threshold=self.pair_config.adx_threshold
        )
        regime_name = regime_enum.name

        # Update strategy state (so it stays in sync)
        strategy.last_regime = regime_enum

        ml_score = None
        if Config.ML_ENABLED:
            ml_score = predictor.get_ensemble_score(df_slice, regime=regime_name)

        order_book = exchange.get_current_orderbook()
        l2_features = feature_engine.compute_l2_features(order_book)

        current_base_bal = 0.0
        current_equity = 0.0
        if self.pair_config.enable_dca_mode:
            base_currency = self.pair_config.symbol.replace("USDT", "")
            current_base_bal = exchange.fetch_balance(base_currency)
            usdt_bal = exchange.fetch_balance("USDT")
            current_equity = usdt_bal + (current_base_bal * current_close)

        analysis = strategy.analyze(
            df_slice,
            ml_score,
            confirm_df=confirm_slice,
            l2_features=l2_features,
            enable_mean_reversion=self.pair_config.enable_mean_reversion,
            enable_trend_following=self.pair_config.enable_trend_following,
            enable_dca_mode=self.pair_config.enable_dca_mode,
            dca_config=dca_config,
            current_price=current_close,
            current_balance_btc=current_base_bal,
            current_equity_usdt=current_equity,
            regime=regime_name,  # Pass explicit regime to avoid re-calculation
            adx_threshold=self.pair_config.adx_threshold,  # Pass ADX threshold for heuristic
            btc_bullish=btc_bullish,  # Pass BTC Trend Context
        )

        if "decision_context" in analysis:
            analysis["decision_context"]["regime"] = regime_name
            analysis["decision_context"]["active_strategy"] = analysis.get("active_strategy")

        return analysis

    def _handle_exits(
        self,
        position: Dict[str, Any],
        current_low: float,
        current_close: float,
        current_time: pd.Timestamp,
        exchange: SimulatedKrakenExchange,
        strategy: StrategySelector,
        analysis: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
        cached_analysis: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        # Stop Loss
        # Check for Close-Only Stop Logic
        close_only = getattr(Config, "CLOSE_ONLY_STOPS", False)

        sl_hit = False
        exit_price = position["stop_loss"]

        if close_only:
            if position["side"] == "LONG" and current_close <= position["stop_loss"]:
                sl_hit = True
                exit_price = current_close  # Exit at Close
        else:
            if position["side"] == "LONG" and current_low <= position["stop_loss"]:
                sl_hit = True
                exit_price = position["stop_loss"]  # Exit at Level (Simulated)

        if sl_hit:
            self._close_position(
                position, exit_price, current_time, "Stop Loss", exchange, decision_log, cached_analysis
            )
            return None

        # Strategy Updates
        pos_strategy = position.get("strategy", "MeanReversion")
        updates = {}
        if pos_strategy == "TrendFollowing":
            updates = strategy.trend_following.get_exit_updates(position, analysis)
        elif pos_strategy == "BTCSmartDCA":
            updates = strategy.btc_dca.get_exit_updates(position, analysis)
        else:
            updates = strategy.mean_reversion.get_exit_updates(position, analysis)

        if updates.get("exit_signal"):
            self._close_position(
                position,
                current_close,
                current_time,
                updates.get("exit_reason", "Strategy Exit"),
                exchange,
                decision_log,
                cached_analysis,
            )
            return None

        # Partial Exits
        exit_qty = 0.0
        exit_reason = ""
        if updates.get("exit_qty"):
            exit_qty = updates["exit_qty"]
            exit_reason = updates.get("exit_reason", "Partial Exit")
        elif updates.get("tp1") and not position["tp1_hit"]:
            exit_qty = position["qty"] * Config.TP1_RATIO
            exit_reason = "TP1"
            position["tp1_hit"] = True

        if exit_qty > 0:
            exit_qty = min(exit_qty, position["qty"])
            trade = exchange.create_exit_order("SELL", exit_qty, reason=exit_reason)

            # Calculate PnL for partial
            entry_price = position["entry_price"]
            pnl = (current_close - entry_price) * exit_qty
            pnl_percent = ((current_close - entry_price) / entry_price) * 100

            trade["pnl"] = float(pnl)
            trade["pnl_percent"] = float(pnl_percent)
            trade["entry_price"] = entry_price

            self._log_decision(
                "EXIT_PARTIAL_LONG",
                current_time,
                cached_analysis,
                decision_log,
                trade_info={
                    "trade_id": position.get("trade_id"),
                    "execution_price": current_close,
                    "execution_qty": exit_qty,
                    "exit_reason": exit_reason,
                },
            )
            position["qty"] -= exit_qty
            if position["qty"] <= 1e-8:
                return None

        # Trailing Stop
        if updates.get("new_sl"):
            if updates["new_sl"] > position["stop_loss"]:
                position["stop_loss"] = updates["new_sl"]

        return position

    def _handle_entry(
        self,
        current_position: Optional[Dict[str, Any]],
        current_time: pd.Timestamp,
        current_close: float,
        exchange: SimulatedKrakenExchange,
        risk_manager: RiskManager,
        analysis: Dict[str, Any],
        last_entry_time: Optional[pd.Timestamp],
        decision_log: List[Dict[str, Any]],
        dca_config: Dict[str, Any],
        strategy: StrategySelector,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        can_enter = not current_position or self.pair_config.enable_dca_mode
        if not can_enter:
            return False, None

        # Cooldown
        min_cooldown = pd.Timedelta(minutes=getattr(Config, "MIN_TRADE_COOLDOWN_MINUTES", 0))
        if self.pair_config.enable_dca_mode and self.pair_config.dca_interval_minutes:
            min_cooldown = pd.Timedelta(minutes=self.pair_config.dca_interval_minutes)

        if last_entry_time and current_time - last_entry_time < min_cooldown:
            return False, None

        usdt_balance = exchange.fetch_balance("USDT")
        entry_price = float(exchange.get_current_candle()["open"])
        atr = analysis.get("atr", 0.0)

        if usdt_balance <= 10.0:
            return False, None

        qty = 0.0
        executed = False
        risk_decision = {}
        gate = {"allowed": True, "reason": "Allowed"}

        if self.pair_config.enable_dca_mode:
            # DCA Sizing
            qty = strategy.btc_dca.calculate_position_size(dca_config, entry_price)
            executed = qty > 0 and (qty * entry_price) <= usdt_balance
            risk_decision = {
                "qty": qty,
                "reason": "DCA Buy",
                "circuit_breaker": False,
                "base_qty": qty,
                "max_vol_qty": qty,
                "max_spot_qty": qty,
            }
            gate = {"allowed": True, "reason": "DCA Bypass"}
        else:
            # Standard Risk Manager
            regime = analysis.get("regime", "CHOP")
            multiplier = analysis.get("size_multiplier", 0.0)

            risk_decision = risk_manager.calculate_size_with_meta(usdt_balance, entry_price, atr, multiplier, regime)

            l2_features = analysis.get("l2_features")
            gate = risk_manager.check_trade_gating_with_meta(l2_features)

            qty = float(cast(float, risk_decision["qty"]))
            executed = bool(cast(bool, gate["allowed"])) and qty > 0

        action = "ENTRY_LONG" if executed else "ENTRY_LONG_RISK_FILTERED"
        trade_info = None

        if executed:
            client_oid = f"backtest_{int(time.time() * 1000)}"
            exchange.create_entry_order("BUY", qty, entry_price, client_oid)

            # Calculate Initial Stop Loss
            if self.pair_config.enable_dca_mode:
                stop_loss = 0.0
            else:
                sl_dist = atr * Config.ATR_MULTIPLIER
                stop_loss = entry_price - sl_dist

            # Update or Create Position
            if current_position and self.pair_config.enable_dca_mode:
                # Accumulate
                new_qty = current_position["qty"] + qty
                new_avg_price = (
                    (current_position["qty"] * current_position["entry_price"]) + (qty * entry_price)
                ) / new_qty

                current_position["qty"] = new_qty
                current_position["entry_price"] = new_avg_price
                current_position["stop_loss"] = 0.0
                trade_info = current_position  # Return updated position
            else:
                trade_info = {
                    "side": "LONG",
                    "qty": qty,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "tp1_hit": False,
                    "trade_id": client_oid,
                    "strategy": analysis.get("active_strategy", "MeanReversion"),
                }

            self._log_decision(
                action,
                current_time,
                analysis,
                decision_log,
                trade_info={
                    "trade_id": client_oid,
                    "execution_price": entry_price,
                    "execution_qty": qty,
                    "exit_reason": None,
                },
            )

            return True, trade_info

        # Log filtered entry
        self._log_decision(action, current_time, analysis, decision_log, trade_info=None)
        return False, None

    def _close_position(
        self,
        position: Dict[str, Any],
        price: float,
        time: pd.Timestamp,
        reason: str,
        exchange: SimulatedKrakenExchange,
        decision_log: List[Dict[str, Any]],
        analysis: Dict[str, Any],
    ) -> None:
        trade = exchange.create_exit_order("SELL", position["qty"], reason=reason, related_oid=position.get("trade_id"))

        # Calculate PnL
        entry_price_val = float(cast(float, position["entry_price"]))
        qty_val = float(cast(float, position["qty"]))
        pnl = (price - entry_price_val) * qty_val
        pnl_percent = ((price - entry_price_val) / entry_price_val) * 100

        trade["pnl"] = float(pnl)
        trade["pnl_percent"] = float(pnl_percent)
        trade["entry_price"] = entry_price_val

        self._log_decision(
            "EXIT_LONG",
            time,
            analysis,
            decision_log,
            trade_info={
                "trade_id": position.get("trade_id"),
                "execution_price": price,
                "execution_qty": position["qty"],
                "exit_reason": reason,
            },
        )

    def _log_decision(
        self,
        action: str,
        timestamp: pd.Timestamp,
        analysis: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
        trade_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Simplified logging for result object
        ctx = analysis.get("decision_context", {})

        # Capture extra features for analytics
        extra_features = {}
        exclude_keys = ["regime", "ml_score", "close", "rsi", "adx", "ema200", "ema200_1h", "atr", "reason_string"]
        for k, v in ctx.items():
            if k not in exclude_keys:
                extra_features[k] = v

        import json

        extra_json = json.dumps(extra_features, default=str)

        row = {
            "timestamp": timestamp,
            "action": action,
            "regime": ctx.get("regime"),
            "ml_score": ctx.get("ml_score"),
            "close": ctx.get("close"),
            "rsi": ctx.get("rsi"),
            "trade_info": trade_info,
            # Expanded Context for Debugging
            "adx": ctx.get("adx"),
            "ema200": ctx.get("ema200"),
            "ema200_1h": ctx.get("ema200_1h"),
            "atr": ctx.get("atr"),
            "reason_string": ctx.get("reason_string"),
            "extra_features": extra_json,
        }
        decision_log.append(row)

    def _calculate_final_equity(self, exchange: SimulatedKrakenExchange, base_currency: str) -> float:
        usdt = exchange.fetch_balance("USDT")
        base = exchange.fetch_balance(base_currency)
        price = exchange.get_market_price()
        return usdt + (base * price)

    def _calculate_metrics(
        self,
        exchange: SimulatedKrakenExchange,
        equity_curve: List[Dict[str, Any]],
        final_equity: float,
        initial_hold_qty: float,
    ) -> Dict[str, Any]:
        pnl = final_equity - exchange.initial_balance
        pnl_percent = (pnl / exchange.initial_balance) * 100

        if not equity_curve:
            return {"pnl": pnl, "pnl_percent": pnl_percent}

        df = pd.DataFrame(equity_curve).set_index("timestamp")
        df["returns"] = df["equity"].pct_change().fillna(0)

        excess_returns = df["returns"] - (0.04 / 365.0)
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(365) if excess_returns.std() != 0 else 0

        return {
            "final_equity": final_equity,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "sharpe_ratio": sharpe,
            "total_trades": len(exchange.trades),
        }

    def _create_empty_result(self, start: Optional[str], end: Optional[str]) -> BacktestResult:
        return BacktestResult(
            run_id="failed", pair=self.pair_config.symbol, start_date=start, end_date=end, config=self.config_overrides
        )
