import asyncio
import logging
import ssl

import aiohttp
import certifi

from src.config import Config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Handles Telegram notifications (Async Native).
    """

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    async def send_message(self, message: str) -> None:
        """Sends a message to the configured Telegram chat asynchronously."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat ID not configured. Skipping notification.")
            return

        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}

        try:
            await self._post_request(payload)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def _post_request(self, payload: dict) -> None:
        """Async POST request with SSL fallback."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        timeout = aiohttp.ClientTimeout(total=10)
        
        # Use trust_env=True to respect HTTP_PROXY/HTTPS_PROXY environment variables
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            try:
                async with session.post(self.base_url, json=payload, ssl=ssl_context) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.warning(f"Telegram API Error: {response.status} - {text}")
                    else:
                        # Success
                        return
            except (aiohttp.ClientSSLError, ssl.SSLError):
                # Retry without SSL verification
                logger.warning("Telegram SSL failed, retrying without verification...")
                try:
                    async with session.post(self.base_url, json=payload, ssl=False) as response:
                        if response.status != 200:
                            text = await response.text()
                            logger.warning(f"Telegram API Error (No SSL): {response.status} - {text}")
                except Exception as e:
                    logger.warning(f"Telegram Request Failed (No SSL): {e}")
            except Exception as e:
                logger.warning(f"Telegram Request Failed: {e}")

    async def notify_startup(self) -> None:
        msg = f"🚀 *XRP-Sentinel V3 Started*\nSymbol: `{Config.SYMBOL}`\nTimeframe: `{Config.TIMEFRAME_PRIMARY}`"
        await self.send_message(msg)

    async def notify_shutdown(self) -> None:
        await self.send_message("🛑 *XRP-Sentinel V3 Shutting Down*")

    async def notify_signal(self, signal: str, regime: str, score: str, multiplier: float) -> None:
        msg = f"🚀 *SIGNAL DETECTED*\nSignal: {signal}\nRegime: {regime}\nScore: {score}\nMult: {multiplier}"
        await self.send_message(msg)

    async def notify_entry(self, side: str, price: float, qty: float, stop_loss: float) -> None:
        msg = f"🔵 *ENTRY EXECUTED*\nSide: {side}\nPrice: {price}\nQty: {qty}\nInitial SL: {stop_loss}"
        await self.send_message(msg)

    async def notify_tp1(self, price: float, qty: float, new_sl: float) -> None:
        msg = f"🟢 *TP1 HIT (50% Closed)*\nPrice: {price}\nQty Closed: {qty}\nStop Loss Moved to Breakeven: {new_sl}"
        await self.send_message(msg)

    async def notify_exit(self, reason: str, price: float, qty: float, pnl: float = 0.0) -> None:
        msg = f"🔴 *POSITION CLOSED*\nReason: {reason}\nPrice: {price}\nQty: {qty}"
        await self.send_message(msg)

    async def notify_error(self, error_msg: str) -> None:
        await self.send_message(f"⚠️ *CRITICAL ERROR*\n`{error_msg}`")
