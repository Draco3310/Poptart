import asyncio
import logging
from typing import Dict, List, Optional

from typing import Optional

from src.config import PAIR_CONFIGS, Config
from src.core.risk_manager import RiskManager
from src.core.trading_agent import TradingAgent
from src.database import Database
from src.exchange import KrakenExchange
from src.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Manages the lifecycle and execution of multiple TradingAgents.
    Handles initialization, parallel execution, rate limiting, and health monitoring.
    """

    def __init__(
        self,
        exchange: KrakenExchange,
        notifier: TelegramNotifier,
        risk_manager: RiskManager,
        db: Optional[Database] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.exchange = exchange
        self.notifier = notifier
        self.risk_manager = risk_manager
        self.db = db
        self.run_id = run_id
        self.agents: Dict[str, TradingAgent] = {}
        self.tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    async def initialize_agents(self) -> None:
        """Initializes agents for all configured pairs."""
        try:
            for symbol, pair_config in PAIR_CONFIGS.items():
                logger.info(f"Initializing Agent for {symbol}...")
                agent = TradingAgent(
                    pair_config, 
                    self.exchange, 
                    self.notifier, 
                    self.risk_manager,
                    db=self.db,
                    run_id=self.run_id
                )
                await agent.initialize()
                self.agents[symbol] = agent
            
            logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
        except Exception as e:
            logger.critical(f"Agent initialization failed: {e}")
            raise

    async def start(self) -> None:
        """Starts the orchestration loop."""
        logger.info("Starting Agent Orchestrator...")
        
        # Start agent loops
        for symbol, agent in self.agents.items():
            task = asyncio.create_task(self._run_agent_loop(agent))
            self.tasks.append(task)
            # Stagger start times to avoid initial API spike
            await asyncio.sleep(2)

        # Monitor tasks
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stops all agents."""
        logger.info("Stopping Agent Orchestrator...")
        self._stop_event.set()
        
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("All agents stopped.")

    async def _run_agent_loop(self, agent: TradingAgent) -> None:
        """
        Runs the main loop for a single agent.
        Includes error handling and rate limit management.
        """
        logger.info(f"Started loop for {agent.symbol}")
        
        while not self._stop_event.is_set():
            try:
                start_time = asyncio.get_event_loop().time()
                
                logger.debug(f"Ticking {agent.symbol}...")
                await agent.tick()
                
                # Calculate sleep time to maintain cycle interval
                # Default cycle is 60s (since we use 5m/1m candles, checking every ~10-60s is fine)
                # But `main.py` was sleeping 10s.
                # Let's use a configurable interval or default to 15s.
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(10.0, 15.0 - elapsed)
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info(f"Agent {agent.symbol} loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in agent {agent.symbol}: {e}", exc_info=True)
                # Backoff on error
                await asyncio.sleep(30)
