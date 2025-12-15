import asyncio
import logging
import time
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
            # Support for Sync (requests)
            config["proxies"] = {"http": Config.HTTP_PROXY, "https": Config.HTTPS_PROXY}
            # Support for Async (aiohttp) - ccxt often uses 'aiohttp_proxy' or 'proxy' depending on version
            # We set both to be safe
            config["aiohttp_proxy"] = Config.HTTP_PROXY
            # config["proxy"] = Config.HTTP_PROXY # Some versions use this
            logger.info(f"Using Proxy: {Config.HTTP_PROXY}")

        self.exchange = ccxt.kraken(config)
        # Force trust_env for aiohttp to pick up env vars if explicit config fails
        self.exchange.aiohttp_trust_env = True

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
        """
        Fetches OHLCV data for the configured symbol and timeframe.
        Supports pagination to fetch more than the exchange limit (e.g. > 720 for Kraken).
        """
        target_symbol = symbol or Config.SYMBOL
        # print(f"DEBUG: fetch_ohlcv called with limit={limit} for {target_symbol}")

        # Determine timeframe duration in seconds
        tf_seconds = 60 # default 1m
        if timeframe == "5m":
            tf_seconds = 300
        elif timeframe == "15m":
            tf_seconds = 900
        elif timeframe == "1h":
            tf_seconds = 3600

        # Calculate 'since' to fetch enough history
        # We add a buffer (1.5x) to ensure we cover gaps
        duration_ms = limit * tf_seconds * 1000
        since = int(time.time() * 1000) - duration_ms

        all_ohlcv = []

        try:
            while len(all_ohlcv) < limit:
                # Fetch batch
                # Kraken limit is usually 720. We ask for 'limit' but ccxt/exchange might truncate.
                # We use 'since' to paginate forward.
                logger.info(f"DEBUG: Fetching batch since={since}, limit={limit}")
                batch = await self.exchange.fetch_ohlcv(target_symbol, timeframe=timeframe, since=since, limit=limit)

                if not batch:
                    logger.info("DEBUG: Batch empty.")
                    break

                logger.info(f"DEBUG: Batch size: {len(batch)}")
                all_ohlcv.extend(batch)

                # Update 'since' to the last timestamp + 1ms to avoid duplicates
                last_ts = batch[-1][0]
                if last_ts <= since:
                    logger.info("DEBUG: Timestamp not advancing.")
                    break # Avoid infinite loop if no progress
                since = last_ts + 1

                # If we got fewer than requested (and likely fewer than exchange max), we are probably at the head
                if len(batch) < 720: # Assuming 720 is Kraken's max
                    logger.info("DEBUG: Batch < 720, assuming head.")
                    break

                # Safety break for too many requests
                if len(all_ohlcv) >= limit * 2:
                    logger.info("DEBUG: Safety break.")
                    break

                await asyncio.sleep(self.exchange.rateLimit / 1000)

            # Deduplicate just in case
            # Convert to DF
            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.drop_duplicates(subset="timestamp", keep="last", inplace=True)

            # Sort and take last 'limit'
            df.sort_values("timestamp", inplace=True)
            if len(df) > limit:
                df = df.iloc[-limit:]

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            raise

    async def fetch_confirm_ohlcv(
        self, symbol: Optional[str] = None, timeframe: str = Config.TIMEFRAME_CONFIRM, limit: int = 5
    ) -> pd.DataFrame:
        """Fetches recent OHLCV data for confirmation (default 1m)."""
        return await self.fetch_ohlcv(limit=limit, timeframe=timeframe, symbol=symbol)

    async def fetch_order_book(
        self, symbol: Optional[str] = None, limit: int = Config.ORDER_BOOK_DEPTH
    ) -> Dict[str, Any]:
        """
        Fetches the Order Book (Level 2 Data).
        Returns: {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}
        """
        target_symbol = symbol or Config.SYMBOL
        try:
            order_book = await self.exchange.fetch_order_book(target_symbol, limit=limit)
            return cast(Dict[str, Any], order_book)
        except Exception as e:
            logger.error(f"Failed to fetch Order Book for {target_symbol}: {e}")
            return {"bids": [], "asks": []}

    async def get_market_price(self, symbol: Optional[str] = None) -> float:
        """Fetches current ticker price."""
        target_symbol = symbol or Config.SYMBOL
        try:
            ticker = await self.exchange.fetch_ticker(target_symbol)
            return float(cast(Any, ticker["last"]))
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {target_symbol}: {e}")
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

    async def fetch_total_equity(self) -> float:
        """
        Calculates total equity (USDT + Value of Holdings).
        Only considers XRP, SOL, BTC to save API calls/complexity.
        """
        try:
            balance = await self.exchange.fetch_balance()
            bal_dict = cast(Dict[str, Any], balance)

            usdt = float(bal_dict.get("USDT", {}).get("total", 0.0))

            equity = usdt
            for coin in ["XRP", "SOL", "BTC"]:
                qty = float(bal_dict.get(coin, {}).get("total", 0.0))
                if qty > 0:
                    try:
                        ticker = await self.exchange.fetch_ticker(f"{coin}/USDT")
                        price = float(ticker["last"])
                        equity += qty * price
                    except Exception as e:
                        logger.warning(f"Could not fetch price for {coin}: {e}")

            return equity
        except Exception as e:
            logger.error(f"Failed to fetch total equity: {e}")
            return 0.0

    async def fetch_recent_trades(self, symbol: Optional[str] = None, limit: int = 1) -> list[Dict[str, Any]]:
        """Fetches recent trades to recover entry price."""
        target_symbol = symbol or Config.SYMBOL
        try:
            trades = await self.exchange.fetch_my_trades(target_symbol, limit=limit)
            return cast(list[Dict[str, Any]], trades)
        except Exception as e:
            logger.error(f"Failed to fetch recent trades for {target_symbol}: {e}")
            return []

    async def create_entry_order(
        self, side: str, qty: float, current_price: float, client_oid: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Places a Marketable Limit Order (Aggressive Limit).
        Monitors for 3 seconds. If not fully filled, cancels remainder.
        """
        target_symbol = symbol or Config.SYMBOL

        if Config.DRY_RUN:
            logger.warning(f"DRY_RUN enabled: Skipping real order placement for {target_symbol}.")
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
        formatted_price = float(cast(str, self.exchange.price_to_precision(target_symbol, limit_price)))
        formatted_qty = float(cast(str, self.exchange.amount_to_precision(target_symbol, qty)))

        params = {"clientOrderId": client_oid}

        logger.info(
            f"Placing Entry {side} Limit Order: {formatted_qty} {target_symbol} @ {formatted_price} "
            f"(ClientOID: {client_oid})"
        )

        try:
            order_side = cast(Any, side.lower())

            order = await self.exchange.create_order(
                symbol=target_symbol,
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
            fetched_order = await self.exchange.fetch_order(order_id, symbol=target_symbol)
            fetched_dict = cast(Dict[str, Any], fetched_order)

            status = fetched_dict["status"]
            filled = float(fetched_dict.get("filled", 0.0))

            if status == "open" or status == "partially_filled":
                logger.info(f"Order {order_id} timed out (Status: {status}, Filled: {filled}). Cancelling remainder.")
                try:
                    await self.exchange.cancel_order(order_id, symbol=target_symbol)
                except Exception as cancel_err:
                    logger.warning(f"Could not cancel order {order_id}: {cancel_err}")

                final_order = await self.exchange.fetch_order(order_id, symbol=target_symbol)
                return cast(Dict[str, Any], final_order)

            return fetched_dict

        except Exception as e:
            logger.error(f"Entry Order Failed: {e}")
            raise

    async def create_exit_order(
        self, side: str, qty: float, reason: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Places a Market Order for Exits (Stop Loss, TP, etc.).
        """
        target_symbol = symbol or Config.SYMBOL

        if Config.DRY_RUN:
            logger.warning(f"DRY_RUN enabled: Skipping exit order for {target_symbol} ({reason}).")
            return {
                "id": f"dry_run_exit_{int(asyncio.get_event_loop().time())}",
                "status": "closed",
                "filled": qty,
                "info": "DRY RUN",
            }

        formatted_qty = float(cast(str, self.exchange.amount_to_precision(target_symbol, qty)))

        logger.info(f"Placing Exit {side} Market Order: {formatted_qty} {target_symbol} (Reason: {reason})")

        try:
            order_side = cast(Any, side.lower())
            order = await self.exchange.create_order(
                symbol=target_symbol, type="market", side=order_side, amount=formatted_qty
            )
            return cast(Dict[str, Any], order)
        except Exception as e:
            logger.error(f"Exit Order Failed: {e}")
            raise
