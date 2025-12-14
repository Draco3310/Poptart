import asyncio
import logging

from src.config import Config, get_model_path
from src.core.feature_engine import FeatureEngine
from src.core.risk_manager import RiskManager
from src.exchange import KrakenExchange
from src.notifier import TelegramNotifier
from src.predictors.orchestrator import PredictorOrchestrator
from src.strategies.strategy_selector import StrategySelector

# Setup Logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("XRP-Sentinel-V2")


async def monitor_balance(exchange: KrakenExchange) -> None:
    """Background task to log balance every 60 seconds."""
    try:
        while True:
            try:
                balance = await exchange.fetch_balance("USDT")
                logger.warning(f"💰 BALANCE CHECK: {balance:.2f} USDT")
            except Exception as e:
                logger.error(f"Failed to fetch balance: {e}")
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass


async def main() -> None:
    logger.info("Starting XRP-Sentinel V2 (Async)...")

    # 1. Initialize Components
    try:
        Config.validate()
        # Mypy doesn't know validate() ensures these are not None
        assert Config.KRAKEN_API_KEY is not None
        assert Config.KRAKEN_API_SECRET is not None
        assert Config.TELEGRAM_TOKEN is not None
        assert Config.TELEGRAM_CHAT_ID is not None

        exchange = KrakenExchange(Config.KRAKEN_API_KEY, Config.KRAKEN_API_SECRET)
        await exchange.initialize()

        notifier = TelegramNotifier(Config.TELEGRAM_TOKEN, Config.TELEGRAM_CHAT_ID)
        await notifier.notify_startup()

        feature_engine = FeatureEngine()

        predictor_orchestrator = PredictorOrchestrator()

        # Dynamic Model Paths
        symbol = Config.SYMBOL.replace("/", "")  # XRPUSDT

        # TP/SL Models
        xgb_path = get_model_path(symbol, "xgb_tp_sl_H1", ext=".model")
        rf_path = get_model_path(symbol, "rf_tp_sl_H1", ext=".joblib")

        predictor_config = [
            {"type": "xgboost", "path": xgb_path, "weight": 0.5},
            {"type": "rf", "path": rf_path, "weight": 0.5},
        ]
        # Check if models exist before loading to avoid crash in prototype
        predictor_orchestrator.load_predictors(predictor_config)

        # Regime Model
        regime_path = get_model_path(symbol, "rf_regime", ext=".joblib")
        strategy = StrategySelector(regime_model_path=regime_path)
        risk_manager = RiskManager()

        # Start Balance Monitor
        balance_task = asyncio.create_task(monitor_balance(exchange))

        logger.info("All components initialized successfully.")
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 2. Main Loop
    try:
        while True:
            try:
                # Sleep Logic (Simple placeholder for "Smart Sleep")
                logger.debug("Waiting for next candle close...")
                await asyncio.sleep(10)  # Reduced for testing loop. Real: 60*60

                # A. Fetch Data (Primary & Confirmation)
                logger.debug(f"Fetching Primary Data ({Config.TIMEFRAME_PRIMARY})...")
                raw_data = await exchange.fetch_ohlcv(timeframe=Config.TIMEFRAME_PRIMARY)

                logger.debug(f"Fetching Confirmation Data ({Config.TIMEFRAME_CONFIRM})...")
                confirm_data = await exchange.fetch_confirm_ohlcv()

                # Fetch BTC Data for Beta Features
                logger.debug("Fetching BTC Data (Context)...")
                btc_data = await exchange.fetch_ohlcv(timeframe=Config.TIMEFRAME_PRIMARY, symbol="BTC/USDT")

                if raw_data.empty:
                    logger.warning("No primary data fetched. Retrying...")
                    continue

                # B. Feature Engineering
                # Prepare Context
                context = {}
                if not btc_data.empty:
                    if "timestamp" in btc_data.columns:
                        btc_data.set_index("timestamp", inplace=True)
                    context["btc_df"] = btc_data

                logger.debug("Computing Features (Primary)...")
                enriched_data = feature_engine.compute_features(raw_data, context=context)

                logger.debug("Computing Features (Confirmation)...")
                # We need indicators on 1m data for the strategy to check bands
                # For confirmation (1m), we might need 1m BTC data too if we want beta features there.
                # But usually confirmation is just price action relative to bands.
                # We'll pass context anyway if available (resampled or raw depending on timeframe match).
                # If confirm_data is 1m and btc_data is 5m, BetaFeatureBlock might fail alignment or produce NaNs.
                # For now, we skip context for confirmation to keep it simple/fast, unless needed.
                enriched_confirm = feature_engine.compute_features(confirm_data)

                # C. Regime Detection & ML Prediction
                # Detect Regime first to inform ML weighting
                regime_enum = strategy.regime_classifier.predict(enriched_data, strategy.last_regime)
                strategy.last_regime = regime_enum
                regime_name = regime_enum.name
                logger.debug(f"Detected Regime: {regime_name}")

                ml_score = None
                if Config.ML_ENABLED:
                    logger.debug("Running ML Predictors...")
                    ml_score = predictor_orchestrator.get_ensemble_score(enriched_data, regime=regime_name)
                    if ml_score is not None:
                        logger.debug(f"ML Ensemble Score: {ml_score:.4f}")
                    else:
                        logger.debug("ML Ensemble Score: None (Neutral/Offline)")
                else:
                    logger.debug("ML Prediction disabled.")

                # D. Strategy Analysis
                logger.debug("Analyzing Strategy Signals...")
                # Note: StrategySelector will re-detect regime internally or we could pass it if we refactored analyze.
                # For now, let it re-detect (cheap operation) or trust it matches.
                analysis = strategy.analyze(enriched_data, ml_score, confirm_df=enriched_confirm, regime=regime_name)

                signal = analysis["signal"]
                multiplier = analysis.get("size_multiplier", 0.0)

                if signal:
                    regime = analysis.get("regime", "UNKNOWN")
                    logger.info(f"SIGNAL GENERATED: {signal} (Regime: {regime}, Size Multiplier: {multiplier})")
                    score_str = f"{ml_score:.2f}" if ml_score is not None else "N/A"
                    await notifier.notify_signal(signal, regime, score_str, multiplier)

                    # E. Execution
                    try:
                        # 1. Get Balance
                        balance = await exchange.fetch_balance("USDT")
                        logger.debug(f"Current USDT Balance: {balance:.2f}")

                        # --- Circuit Breaker: Max Drawdown Check ---
                        if risk_manager.check_circuit_breaker(balance):
                            await notifier.send_message("🛑 CIRCUIT BREAKER: Trading Paused.")
                            await asyncio.sleep(300)  # Pause for 5 minutes
                            continue
                        # -------------------------------------------

                        # 2. Calculate Size
                        entry_price = analysis["close"]
                        atr = analysis["atr"]
                        regime = analysis.get("regime", "CHOP")

                        # Use Risk Manager for Sizing (Volatility Targeting)
                        qty = risk_manager.calculate_size(balance, entry_price, atr, multiplier, regime)

                        logger.debug(f"Calculated Position Size: {qty:.4f} XRP")

                        if qty > 0:
                            if Config.DRY_RUN:
                                logger.info(
                                    f"DRY RUN: Would place {signal} order for {qty:.4f} XRP @ ~{entry_price:.4f}"
                                )
                                await notifier.send_message(f"🧪 DRY RUN: {signal} {qty:.4f} XRP")
                            else:
                                # 3. Place Order
                                client_oid = f"sentinel_{int(asyncio.get_event_loop().time())}"
                                order = await exchange.create_entry_order(signal, qty, entry_price, client_oid)
                                logger.info(f"Order Placed: {order['id']}")

                                # Calculate estimated SL for notification
                                sl_dist = atr * Config.ATR_MULTIPLIER
                                stop_loss = entry_price - sl_dist if signal == "LONG" else entry_price + sl_dist
                                await notifier.notify_entry(signal, entry_price, qty, stop_loss)
                        else:
                            logger.warning("Calculated quantity is 0. Check risk settings or balance.")

                    except Exception as exec_err:
                        logger.error(f"Execution Failed: {exec_err}")
                        await notifier.notify_error(f"Execution Failed: {exec_err}")

                else:
                    logger.debug("No Signal.")

            except asyncio.CancelledError:
                logger.info("Main loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

    finally:
        logger.info("Cleaning up resources...")
        if "balance_task" in locals():
            balance_task.cancel()
            try:
                await balance_task
            except asyncio.CancelledError:
                pass
        await exchange.close()
        await notifier.notify_shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
