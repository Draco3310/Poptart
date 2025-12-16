import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from src.config import Config
from src.core.trading_agent import TradingAgent
from src.simulated_exchange import SimulatedKrakenExchange
from src.core.backtest_analytics import BacktestAnalytics
from src.core.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

class AsyncBacktestEngine:
    """
    Asynchronous Event-Driven Backtest Engine.
    Orchestrates multiple TradingAgents in a unified simulation loop.
    Replaces the legacy synchronous BacktestEngine.
    """

    def __init__(self, agents: List[TradingAgent], exchange: SimulatedKrakenExchange):
        self.agents = agents
        self.exchange = exchange
        self.analytics = BacktestAnalytics()
        
        # State
        self.decision_logs: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Pre-computed Data
        self.enriched_data_map: Dict[str, pd.DataFrame] = {}
        self.enriched_confirm_map: Dict[str, pd.DataFrame] = {}
        self.feature_engine = FeatureEngine()

    async def precompute_features(self):
        """
        Pre-computes technical indicators and ML features for all agents.
        This avoids re-calculation in the hot loop.
        """
        logger.info("Pre-computing features for all agents...")
        
        # Context for BTC (if available in exchange data)
        btc_context = {}
        if "BTC/USDT" in self.exchange.data_map:
            btc_df = self.exchange.data_map["BTC/USDT"].copy()
            btc_df.set_index("timestamp", inplace=True)
            btc_context["btc_df"] = btc_df

        for agent in self.agents:
            symbol = agent.symbol
            if symbol not in self.exchange.data_map:
                logger.warning(f"No data found for {symbol}, skipping feature gen.")
                continue

            logger.info(f"Computing features for {symbol}...")
            raw_1m = self.exchange.data_map[symbol].copy()
            raw_1m.set_index("timestamp", inplace=True)

            # 1. Confirmation Features (1m)
            enriched_confirm = self.feature_engine.compute_features(raw_1m, context=btc_context)
            self.enriched_confirm_map[symbol] = enriched_confirm

            # 2. Primary Features (5m)
            agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            raw_5m = raw_1m.resample("5min").agg(agg_rules).dropna()
            
            # Resample BTC context
            btc_context_5m = {}
            if "btc_df" in btc_context:
                btc_5m = btc_context["btc_df"].resample("5min").agg(agg_rules).dropna()
                btc_context_5m["btc_df"] = btc_5m

            enriched_5m = self.feature_engine.compute_features(raw_5m, context=btc_context_5m)
            
            # 3. Batch ML Inference (Optional Optimization)
            # TODO: Implement batch ML inference here to speed up 'process_tick'
            
            self.enriched_data_map[symbol] = enriched_5m

    async def run(self, warmup_steps: int = 200):
        """
        Executes the main simulation loop.
        """
        logger.info("Starting Simulation Loop...")
        
        # Assume aligned data, use first symbol for length
        first_symbol = self.exchange.symbols[0]
        total_steps = len(self.exchange.data_map[first_symbol])
        start_time = datetime.now()
        
        while self.exchange.step():
            current_time = self.exchange.get_current_time()
            idx = self.exchange.current_index
            
            if idx < warmup_steps:
                continue
                
            # Progress Log
            if idx % 5000 == 0:
                elapsed = datetime.now() - start_time
                rate = idx / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                equity = await self.exchange.fetch_total_equity()
                logger.info(f"Step {idx}/{total_steps} ({idx/total_steps:.1%}) - {rate:.0f} steps/s - Equity: {equity:.2f}")

            # Track Equity (Daily)
            if current_time.hour == 0 and current_time.minute == 0:
                eq = await self.exchange.fetch_total_equity()
                self.equity_curve.append({"timestamp": current_time, "equity": eq})

            # Tick Agents
            for agent in self.agents:
                await self._tick_agent(agent, current_time)

        # Finalize
        await self._finalize_run()

    async def _tick_agent(self, agent: TradingAgent, current_time: pd.Timestamp):
        """
        Processes a single agent for the current timestep.
        """
        symbol = agent.symbol
        
        # Data Slicing
        # We assume current_time is the close of the 1m candle.
        # Primary (5m): Use latest CLOSED 5m candle.
        target_5m_ts = current_time.floor("5min") - pd.Timedelta(minutes=5)
        target_1m_ts = current_time

        df_5m = self.enriched_data_map.get(symbol)
        df_1m = self.enriched_confirm_map.get(symbol)

        if df_5m is None or df_1m is None:
            return

        # Check availability & Create Slices using searchsorted (O(log N))
        # This avoids creating large intermediate copies with .loc[:ts]
        
        # 5m Slice
        # Find position where target_5m_ts would be inserted (right side to include it)
        pos_5m = df_5m.index.searchsorted(target_5m_ts, side='right')
        if pos_5m < 200: # Need at least 200 bars
            return
        
        # Verify the timestamp at pos-1 matches target (or is close enough if we allow gaps)
        # But strictly, we want the candle AT target_5m_ts to be the last one.
        # searchsorted 'right' returns index after the match.
        # So df_5m.index[pos_5m - 1] should be <= target_5m_ts.
        
        # Actually, simpler check:
        # If we want exact match:
        # if df_5m.index[pos_5m - 1] != target_5m_ts: return
        
        # But let's stick to the logic: "latest CLOSED 5m candle".
        # If target_5m_ts exists, pos_5m will be index+1.
        # If it doesn't exist (gap), pos_5m will be the next available.
        # We want the slice ending at or before target_5m_ts.
        
        slice_5m = df_5m.iloc[pos_5m-200:pos_5m]
        
        # 1m Slice
        pos_1m = df_1m.index.searchsorted(target_1m_ts, side='right')
        if pos_1m < 20:
            return
            
        slice_1m = df_1m.iloc[pos_1m-20:pos_1m]

        # Market Data
        l2 = await self.exchange.fetch_order_book(symbol)
        l2_features = self.feature_engine.compute_l2_features(l2)
        price = await self.exchange.get_market_price(symbol)
        
        # Balances
        base_curr = symbol.replace("USDT", "").replace("/", "")
        base_bal = await self.exchange.fetch_balance(base_curr)
        usdt_bal = await self.exchange.fetch_balance("USDT")
        equity = usdt_bal + (base_bal * price)

        # Agent Logic
        analysis = await agent.process_tick(
            slice_5m,
            slice_1m,
            l2_features,
            price,
            base_bal,
            equity
        )

        # Logging
        if analysis:
            log_entry = {
                "timestamp": current_time,
                "symbol": symbol,
                "action": analysis.get("signal"),
                "regime": analysis.get("decision_context", {}).get("regime"),
                "price": price,
                "equity": equity,
                "reason": analysis.get("decision_context", {}).get("reason_string"),
                "trade_info": None # Can be populated if trade occurred
            }
            self.decision_logs.append(log_entry)

    async def _finalize_run(self):
        """
        Generates report and ingests to DB.
        """
        final_equity = await self.exchange.fetch_total_equity()
        pnl = final_equity - self.exchange.initial_balance
        
        logger.info("=" * 50)
        logger.info(f"ASYNC BACKTEST COMPLETE")
        logger.info(f"Final Equity: {final_equity:.2f} USDT")
        logger.info(f"PnL: {pnl:.2f} USDT")
        logger.info(f"Total Trades: {len(self.exchange.trades)}")
        logger.info("=" * 50)

        # DB Ingestion
        run_id = f"ASYNC_BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Register Run
        strat_id = self.analytics.get_or_create_strategy_version("MultiAgent", "v2.0", "HEAD")
        
        run_data = {
            "run_id": run_id,
            "run_type": "BACKTEST_ASYNC",
            "start_date": str(self.exchange.data_map[self.exchange.symbols[0]].iloc[0]["timestamp"]),
            "end_date": str(self.exchange.data_map[self.exchange.symbols[0]].iloc[-1]["timestamp"]),
            "strategy_version_id": strat_id,
            "feature_engine_version_id": None,
            "data_source_id": None,
            "config_snapshot_id": self.analytics.get_or_create_config_snapshot({}),
            "code_version": "HEAD",
            "report_path": f"backtesting_results/report_{run_id}.txt",
            "debug_log_path": f"backtesting_results/debug_{run_id}.log",
            "notes": "Async Engine Run"
        }
        self.analytics.register_run(run_data)
        
        # Ingest Trades
        # Transform SimulatedExchange trades to DB schema
        db_trades = []
        for t in self.exchange.trades:
            db_trades.append({
                "run_id": run_id,
                "client_oid": t.get("id"),
                "timestamp": t.get("timestamp"),
                "side": t.get("side"),
                "price": t.get("price"),
                "amount": t.get("amount"),
                "fee": t.get("fee", 0.0),
                "pnl": t.get("pnl", 0.0),
                "pnl_pct": t.get("pnl_percent", 0.0),
                "status": "FILLED",
                "order_type": "MARKET", # Sim is mostly market
                "extra": t.get("reason")
            })
        
        self.analytics.bulk_insert_trades(db_trades)
        
        # Ingest Decisions
        # Add run_id to logs
        for d in self.decision_logs:
            d["run_id"] = run_id
            
        self.analytics.bulk_insert_decisions(self.decision_logs)
        
        logger.info(f"Results saved with Run ID: {run_id}")
