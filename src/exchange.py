import asyncio
import logging
from typing import Any, Dict, Optional, cast

import ccxt.async_support as ccxt
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class KrakenExchange:
    """
    Kraken Exchange Wrapper using ccxt (Async).
    Handles Marketable Limit Orders, Timeouts, and Data Fetching.
    """

    def __init__(self, api_key: str, api_secret: str):
        config = {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True, "options": {"defaultType": "spot"}}

        if Config.HTTP_PROXY or Config.HTTPS_PROXY:
            config["proxies"] = {"http": Config.HTTP_PROXY, "https": Config.HTTPS_PROXY}
            logger.info(f"Using Proxy: {Config.HTTP_PROXY}")

        self.exchange = ccxt.kraken(config)

    async def initialize(self) -> None:
        """Initializes the exchange connection and loads markets."""
        try:
            await self.exchange.load_markets()
            logger.info("Connected to Kraken (Async). Markets loaded.")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise

    async def close(self) -> None:
        """Closes the exchange connection."""
        await self.exchange.close()
        logger.info("Kraken connection closed.")

    async def fetch_ohlcv(
        self, limit: int = 300, timeframe: str = Config.TIMEFRAME_PRIMARY, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetches OHLCV data for the configured symbol and timeframe."""
        target_symbol = symbol or Config.SYMBOL
        try:
            ohlcv = await self.exchange.fetch_ohlcv(target_symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            raise

    async def fetch_confirm_ohlcv(self, timeframe: str = Config.TIMEFRAME_CONFIRM, limit: int = 5) -> pd.DataFrame:
        """Fetches recent OHLCV data for confirmation (default 1m)."""
        return await self.fetch_ohlcv(limit=limit, timeframe=timeframe)

    async def fetch_order_book(self, limit: int = Config.ORDER_BOOK_DEPTH) -> Dict[str, Any]:
        """
        Fetches the Order Book (Level 2 Data).
        Returns: {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}
        """
        try:
            order_book = await self.exchange.fetch_order_book(Config.SYMBOL, limit=limit)
            return cast(Dict[str, Any], order_book)
        except Exception as e:
            logger.error(f"Failed to fetch Order Book: {e}")
            return {"bids": [], "asks": []}

    async def get_market_price(self) -> float:
        """Fetches current ticker price."""
        try:
            ticker = await self.exchange.fetch_ticker(Config.SYMBOL)
            return float(cast(Any, ticker["last"]))
        except Exception as e:
            logger.error(f"Failed to fetch ticker: {e}")
            raise

    async def fetch_balance(self, currency: str = "USDT") -> float:
        """Fetches free balance for the specified currency."""
        try:
            balance = await self.exchange.fetch_balance()
            bal_dict = cast(Dict[str, Any], balance)
            return float(bal_dict.get(currency, {}).get("free", 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    async def create_entry_order(self, side: str, qty: float, current_price: float, client_oid: str) -> Dict[str, Any]:
        """
        Places a Marketable Limit Order (Aggressive Limit).
        Monitors for 3 seconds. If not fully filled, cancels remainder.
        """
        if Config.DRY_RUN:
            logger.warning("DRY_RUN enabled: Skipping real order placement.")
            return {
                "id": f"dry_run_{client_oid}",
                "status": "closed",
                "filled": qty,
                "price": current_price,
                "info": "DRY RUN",
            }

        if side.upper() == "BUY":
            limit_price = current_price * (1 + Config.LIMIT_ORDER_BUFFER)
        else:
            limit_price = current_price * (1 - Config.LIMIT_ORDER_BUFFER)

        # ccxt async methods for precision might be synchronous helpers or async?
        # price_to_precision is usually synchronous in ccxt base class.
        formatted_price = float(cast(str, self.exchange.price_to_precision(Config.SYMBOL, limit_price)))
        formatted_qty = float(cast(str, self.exchange.amount_to_precision(Config.SYMBOL, qty)))

        params = {"clientOrderId": client_oid}

        logger.info(f"Placing Entry {side} Limit Order: {formatted_qty} @ {formatted_price} (ClientOID: {client_oid})")

        try:
            order_side = cast(Any, side.lower())

            order = await self.exchange.create_order(
                symbol=Config.SYMBOL,
                type="limit",
                side=order_side,
                amount=formatted_qty,
                price=formatted_price,
                params=params,
            )
            order_dict = cast(Dict[str, Any], order)
            order_id = cast(str, order_dict["id"])

            # 3-second Timeout Logic
            await asyncio.sleep(Config.ORDER_TIMEOUT_SECONDS)

            # Refresh order status
            fetched_order = await self.exchange.fetch_order(order_id, symbol=Config.SYMBOL)
            fetched_dict = cast(Dict[str, Any], fetched_order)

            status = fetched_dict["status"]
            filled = float(fetched_dict.get("filled", 0.0))

            if status == "open" or status == "partially_filled":
                logger.info(f"Order {order_id} timed out (Status: {status}, Filled: {filled}). Cancelling remainder.")
                try:
                    await self.exchange.cancel_order(order_id, symbol=Config.SYMBOL)
                except Exception as cancel_err:
                    logger.warning(f"Could not cancel order {order_id}: {cancel_err}")

                final_order = await self.exchange.fetch_order(order_id, symbol=Config.SYMBOL)
                return cast(Dict[str, Any], final_order)

            return fetched_dict

        except Exception as e:
            logger.error(f"Entry Order Failed: {e}")
            raise

    async def create_exit_order(self, side: str, qty: float, reason: str) -> Dict[str, Any]:
        """
        Places a Market Order for Exits (Stop Loss, TP, etc.).
        """
        if Config.DRY_RUN:
            logger.warning(f"DRY_RUN enabled: Skipping exit order ({reason}).")
            return {
                "id": f"dry_run_exit_{int(asyncio.get_event_loop().time())}",
                "status": "closed",
                "filled": qty,
                "info": "DRY RUN",
            }

        formatted_qty = float(cast(str, self.exchange.amount_to_precision(Config.SYMBOL, qty)))

        logger.info(f"Placing Exit {side} Market Order: {formatted_qty} (Reason: {reason})")

        try:
            order_side = cast(Any, side.lower())
            order = await self.exchange.create_order(
                symbol=Config.SYMBOL, type="market", side=order_side, amount=formatted_qty
            )
            return cast(Dict[str, Any], order)
        except Exception as e:
            logger.error(f"Exit Order Failed: {e}")
            raise
