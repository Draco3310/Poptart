from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.exchange import KrakenExchange


class TestSecurity:
    """
    Security & Safety tests for the Kernel.
    """

    def test_credential_leak_prevention(self) -> None:
        """
        Test Case 1: Credential Leak Prevention

        Test Steps:
        1. Temporarily unset critical environment variables.
        2. Call Config.validate().
        3. Verify ValueError is raised.

        Expected Result:
        - Raises ValueError with message about missing keys.
        """
        # Save original state
        original_key = Config.KRAKEN_API_KEY

        # Simulate missing key
        Config.KRAKEN_API_KEY = None

        try:
            with pytest.raises(ValueError, match="Missing critical environment variables"):
                Config.validate()
        finally:
            # Restore state
            Config.KRAKEN_API_KEY = original_key

    def test_dry_run_enforcement(self, mock_kraken: MagicMock) -> None:
        """
        Test Case 2: Dry Run Enforcement
        Placeholder for legacy test structure.
        Actual implementation is in test_dry_run_safety_check.
        """
        pass

    @pytest.mark.asyncio
    async def test_dry_run_safety_check(self) -> None:
        """
        Test Case 2: Dry Run Enforcement (Modified)

        Test Steps:
        1. Force Config.DRY_RUN = True.
        2. Initialize KrakenExchange.
        3. Call create_entry_order.
        4. Assert that the underlying ccxt.create_order was NOT called.
        """
        # 1. Setup
        original_dry_run = Config.DRY_RUN
        Config.DRY_RUN = True

        try:
            # Initialize Exchange (Mocking the internal ccxt instance)
            # We patch ccxt.kraken to avoid network calls during __init__
            # Note: src.exchange imports ccxt.async_support as ccxt
            with patch("src.exchange.ccxt.kraken") as mock_kraken_cls:
                # Configure the mock to return a MagicMock instance
                mock_instance = MagicMock()
                mock_kraken_cls.return_value = mock_instance

                # We need valid keys to pass __init__ validation if any,
                # but KrakenExchange just passes them to ccxt.
                exchange = KrakenExchange("key", "secret")

                # 2. Execute
                # This should hit the DRY_RUN check and return immediately
                # Since it's async, we must await it
                result = await exchange.create_entry_order("BUY", 100, 1.0, "test_oid")

                # 3. Assert
                # Verify we got the dummy response
                assert result["info"] == "DRY RUN"
                assert result["status"] == "closed"

                # Verify REAL create_order was NOT called on the mock instance
                mock_instance.create_order.assert_not_called()

        finally:
            Config.DRY_RUN = original_dry_run
