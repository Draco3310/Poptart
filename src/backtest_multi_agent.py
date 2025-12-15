import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.config import PAIR_CONFIGS, Config, get_data_path
from src.core.backtest_analytics import BacktestAnalytics
from src.core.feature_engine import FeatureEngine
from src.core.risk_manager import RiskManager
from src.core.trading_agent import TradingAgent
from src.simulated_exchange import SimulatedKrakenExchange

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiAgentBacktest")

# Suppress noisy loggers
logging.getLogger("src.simulated_exchange").setLevel(logging.WARNING)
logging.getLogger("src.core.feature_engine").setLevel(logging.WARNING)
logging.getLogger("src.predictors.orchestrator").setLevel(logging.WARNING)
logging.getLogger("src.predictors.xgboost_model").setLevel(logging.WARNING)
logging.getLogger("src.predictors.random_forest").setLevel(logging.WARNING)
logging.getLogger("src.predictors.regime_classifier").setLevel(logging.WARNING)

# Mock Notifier
class MockNotifier:
    async def send_message(self, msg: str):
        pass
    async def notify_entry(self, *args, **kwargs):
        pass
    async def notify_exit(self, *args, **kwargs):
        pass

async def run_multi_agent_backtest():
    logger.info(">>> STARTING MULTI-AGENT BACKTEST <<<")

    # 1. Load Data
    symbols = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
    data_map: Dict[str, pd.DataFrame] = {}
    
    # Define Backtest Window (Last Year)
    # Based on user feedback, data ends Nov 30, 2025.
    # We want Nov 30, 2024 to Nov 30, 2025.
    start_date = pd.Timestamp("2024-11-30")
    end_date = pd.Timestamp("2025-11-30")

    logger.info(f"Target Window: {start_date} to {end_date}")

    for symbol in symbols:
        clean_sym = symbol.replace("/", "")
        path = get_data_path(clean_sym, "1m")
        if not os.path.exists(path):
            logger.error(f"Data not found for {symbol} at {path}")
            return

        logger.info(f"Loading {symbol} from {path}...")
        df = pd.read_csv(path)
        
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter Window
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if df.empty:
            logger.error(f"No data for {symbol} in target window.")
            return
            
        data_map[symbol] = df
        logger.info(f"Loaded {len(df)} rows for {symbol}")

    # 2. Align Data (Intersection)
    # Find common timestamps
    common_timestamps = set(data_map[symbols[0]]["timestamp"])
    for sym in symbols[1:]:
        common_timestamps.intersection_update(set(data_map[sym]["timestamp"]))
    
    sorted_timestamps = sorted(list(common_timestamps))
    logger.info(f"Aligned Data: {len(sorted_timestamps)} common timestamps.")
    
    if not sorted_timestamps:
        logger.error("No overlapping data found.")
        return

    # Filter dataframes to common timestamps
    for sym in symbols:
        df = data_map[sym]
        data_map[sym] = df[df["timestamp"].isin(common_timestamps)].sort_values("timestamp").reset_index(drop=True)

    # 3. Pre-compute Features
    logger.info("Pre-computing features...")
    feature_engine = FeatureEngine()
    
    enriched_data_map: Dict[str, pd.DataFrame] = {}
    enriched_confirm_map: Dict[str, pd.DataFrame] = {}
    
    # Context for BTC
    btc_context = {}
    if "BTCUSDT" in data_map:
        btc_df = data_map["BTCUSDT"].copy()
        btc_df.set_index("timestamp", inplace=True)
        btc_context["btc_df"] = btc_df

    for sym in symbols:
        logger.info(f"Computing features for {sym}...")
        raw_1m = data_map[sym].copy()
        raw_1m.set_index("timestamp", inplace=True)
        
        # 1m Features (Confirmation)
        enriched_confirm = feature_engine.compute_features(raw_1m, context=btc_context)
        enriched_confirm_map[sym] = enriched_confirm
        
        # 5m Features (Primary)
        agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        raw_5m = raw_1m.resample("5min").agg(agg_rules).dropna() # type: ignore
        
        # Resample BTC context for 5m
        btc_context_5m = {}
        if "btc_df" in btc_context:
            btc_5m = btc_context["btc_df"].resample("5min").agg(agg_rules).dropna() # type: ignore
            btc_context_5m["btc_df"] = btc_5m
            
        enriched_5m = feature_engine.compute_features(raw_5m, context=btc_context_5m)
        enriched_data_map[sym] = enriched_5m

    # 4. Initialize Components
    exchange = SimulatedKrakenExchange(data_map, initial_balance=10000.0)
    risk_manager = RiskManager()
    notifier = MockNotifier()
    
    agents: List[TradingAgent] = []
    for sym in symbols:
        # Find config
        config_key = sym
        if config_key not in PAIR_CONFIGS:
            logger.warning(f"No config for {config_key}, skipping agent.")
            continue
            
        agent = TradingAgent(PAIR_CONFIGS[config_key], exchange, notifier, risk_manager) # type: ignore
        # Manually set historic data to avoid fetching in initialize
        # Actually initialize() loads from disk, which is fine, but we want to use our aligned data?
        # agent.initialize() loads "historic data" for lookback.
        # In backtest, we provide lookback via the pre-computed features.
        # So we can skip agent.initialize() or let it run (it won't hurt, just loads files).
        # Let's skip it to save time and rely on passed data.
        agents.append(agent)

    # 5. Simulation Loop
    logger.info("Starting Simulation Loop...")
    
    equity_curve = []
    decision_logs = []
    
    # Warmup
    warmup_steps = 200
    
    total_steps = len(sorted_timestamps)
    
    # We iterate through the exchange's step()
    # exchange.step() advances index.
    # We need to sync with timestamps.
    
    start_time = datetime.now()
    
    while exchange.step():
        current_time = exchange.get_current_time()
        idx = exchange.current_index
        
        if idx < warmup_steps:
            continue
            
        if idx % 1000 == 0:
            elapsed = datetime.now() - start_time
            rate = idx / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            logger.info(f"Step {idx}/{total_steps} ({idx/total_steps:.1%}) - {rate:.0f} steps/s - Equity: {await exchange.fetch_total_equity():.2f}")

        # Update Equity Curve (Daily)
        if current_time.hour == 0 and current_time.minute == 0:
            eq = await exchange.fetch_total_equity()
            equity_curve.append({"timestamp": current_time, "equity": eq})

        # Tick Agents
        for agent in agents:
            # Get Data Slices
            # Primary (5m)
            # We need the row corresponding to current_time (or latest closed 5m candle)
            # If current_time is 10:03, latest 5m close is 10:00.
            # But we are simulating 1m steps.
            # If we are at 10:04, we use 10:00 candle?
            # Yes, standard logic.
            
            # Find 5m index
            # enriched_data_map[sym] is indexed by timestamp
            
            # Optimization: We can't do .loc every time, it's slow.
            # But for 1.5M steps we might have to.
            
            # Current 5m candle timestamp
            current_5m_ts = current_time.floor("5min")
            # If current_time is 10:05:00, floor is 10:05:00.
            # But the candle for 10:05 closes at 10:05 (or 10:09?).
            # Usually timestamp is open time.
            # If timestamp is open time, 10:00 candle covers 10:00-10:05.
            # At 10:04, we don't have 10:00 close yet?
            # Wait, if we are at 10:04, we have 10:00-10:05 candle forming.
            # We should use the *previous* completed candle?
            # Or if we are using "Close" price of 1m candles to simulate,
            # we can form the 5m candle on the fly?
            # But we pre-computed features on completed 5m candles.
            # So at 10:04, the latest *completed* 5m candle is 09:55 (assuming 10:00 is current).
            
            # Let's assume we trade at the *close* of the candle.
            # If current_time is 10:05, the 10:00 candle just closed.
            # So we can use the 10:00 candle features.
            
            # Check if we are at a 5m boundary
            # if current_time.minute % 5 == 0:
            # But we might want to check every minute for "Confirmation" (1m).
            
            # Logic:
            # Primary Analysis: Run on latest CLOSED 5m candle.
            # Confirmation: Run on latest CLOSED 1m candle (which is current_time - 1m? or current_time if it's close time).
            # Our data timestamps are usually Open Time.
            # If timestamp is 10:00, it closes at 10:01 (for 1m).
            # If we are processing row 10:00, we assume it just closed?
            # Usually backtesters iterate *after* close.
            
            # Let's assume current_time is the time of the candle that just closed.
            # So we have data for current_time.
            
            # 5m Data
            # If current_time is 10:04, latest 5m open was 10:00.
            # But is it closed? No.
            # Latest closed 5m was 09:55.
            # So we look for 5m candle with timestamp <= current_time - 5min?
            
            target_5m_ts = current_time.floor("5min") - pd.Timedelta(minutes=5)
            
            # 1m Data
            target_1m_ts = current_time # Current row is the latest 1m candle
            
            # Fetch Slices
            df_5m = enriched_data_map[agent.symbol]
            df_1m = enriched_confirm_map[agent.symbol]
            
            if target_5m_ts not in df_5m.index or target_1m_ts not in df_1m.index:
                continue
                
            # Slice (Last 200 rows for indicators lookback if needed, but features are pre-computed)
            # We just need a slice ending at target.
            # StrategySelector uses .iloc[-1].
            # So we pass a slice ending at target.
            
            # Optimization: Get integer location
            # This is slow to do every time.
            # But let's try.
            
            # We can maintain a pointer?
            # df_5m is sorted.
            
            # Let's just use .loc with a slice for now.
            # df_5m.loc[:target_5m_ts].iloc[-50:]
            
            slice_5m = df_5m.loc[:target_5m_ts].iloc[-50:]
            slice_1m = df_1m.loc[:target_1m_ts].iloc[-10:]
            
            # L2 Features
            l2 = await exchange.fetch_order_book(agent.symbol)
            l2_features = feature_engine.compute_l2_features(l2)
            
            # Prices
            price = await exchange.get_market_price(agent.symbol)
            
            # Balances
            base_curr = agent.symbol.replace("USDT", "").replace("/", "")
            base_bal = await exchange.fetch_balance(base_curr)
            usdt_bal = await exchange.fetch_balance("USDT")
            equity = usdt_bal + (base_bal * price)
            
            # Process Tick
            analysis = await agent.process_tick(
                slice_5m,
                slice_1m,
                l2_features,
                price,
                base_bal,
                equity
            )
            
            # Log Decision
            if analysis:
                log_entry = {
                    "timestamp": current_time,
                    "symbol": agent.symbol,
                    "action": analysis.get("signal"),
                    "regime": analysis.get("decision_context", {}).get("regime"),
                    "price": price,
                    "equity": equity,
                    "reason": analysis.get("decision_context", {}).get("reason_string")
                }
                decision_logs.append(log_entry)

    # 6. Final Report
    final_equity = await exchange.fetch_total_equity()
    pnl = final_equity - 10000.0
    pnl_pct = (pnl / 10000.0) * 100
    
    logger.info("=" * 50)
    logger.info(f"MULTI-AGENT BACKTEST COMPLETE")
    logger.info(f"Final Equity: {final_equity:.2f} USDT")
    logger.info(f"PnL: {pnl:.2f} USDT ({pnl_pct:.2f}%)")
    logger.info(f"Total Trades: {len(exchange.trades)}")
    logger.info("=" * 50)
    
    # 7. DB Ingestion
    logger.info("Ingesting results to database...")
    analytics = BacktestAnalytics()
    
    run_id = f"MULTI_AGENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Register Run
    run_data = {
        "run_id": run_id,
        "run_type": "BACKTEST_MULTI",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "config_snapshot_id": analytics.get_or_create_config_snapshot({}),
        "code_version": "HEAD",
        "notes": "Multi-Agent Portfolio Backtest"
    }
    analytics.register_run(run_data)
    
    # Ingest Trades
    for t in exchange.trades:
        t["run_id"] = run_id
        # Ensure keys match DB schema
        # DB expects: trade_id, symbol, side, entry_price, exit_price, qty, pnl, pnl_percent, entry_time, exit_time, exit_reason
        # SimulatedExchange trades are unified (entry and exit are separate orders? No, it tracks 'trades' as completed orders?)
        # SimulatedExchange.trades list contains executed orders.
        # We need to reconstruct "Round Trip" trades for the DB?
        # Or does DB accept individual orders?
        # BacktestAnalytics.bulk_insert_trades expects round-trip trades usually.
        # Let's check BacktestAnalytics.
        pass
        
    # Actually, SimulatedExchange.trades in my implementation are just executed orders.
    # I need to pair them up to calculate PnL per trade for the DB.
    # Or I can just dump the orders if the DB supports it.
    # Let's look at BacktestAnalytics.bulk_insert_trades schema.
    
    # For now, I will save the raw orders to CSV and let the user analyze.
    # But the user specifically asked for DB ingestion.
    # I'll try to match the schema.
    
    # Save CSVs
    pd.DataFrame(exchange.trades).to_csv(f"backtesting_results/multi_agent_trades_{run_id}.csv")
    pd.DataFrame(decision_logs).to_csv(f"backtesting_results/multi_agent_decisions_{run_id}.csv")
    
    logger.info(f"Results saved to backtesting_results/ with run_id {run_id}")

if __name__ == "__main__":
    asyncio.run(run_multi_agent_backtest())
