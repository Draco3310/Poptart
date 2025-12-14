import logging
import os

import requests

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestConnection")

try:
    logger.info("Attempting to connect to https://api.kraken.com/0/public/Time")
    response = requests.get("https://api.kraken.com/0/public/Time", timeout=10)
    logger.info(f"Status Code: {response.status_code}")
    logger.info(f"Response: {response.text}")
except Exception as e:
    logger.error(f"Connection failed: {e}")
    # Check for proxy env vars
    logger.info("Environment Variables:")
    for k, v in os.environ.items():
        if "PROXY" in k.upper():
            logger.info(f"{k}: {v}")
