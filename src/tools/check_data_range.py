import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CheckDataRange")


def get_range(filepath: str) -> None:
    if not os.path.exists(filepath):
        logger.error(f"{filepath}: Not Found")
        return

    with open(filepath, "rb") as f:
        first_line = f.readline().decode().strip()
        f.seek(-1024, 2)  # Go to near end
        last_chunk = f.read().decode()
        last_line = last_chunk.strip().split("\n")[-1]

    logger.info(f"File: {filepath}")
    logger.info(f"Header: {first_line}")
    logger.info(f"Last Line: {last_line}")
    logger.info("-" * 20)


if __name__ == "__main__":
    get_range("data/XRP/XRPUSDT_1.csv")
    get_range("data/SOL/SOLUSDT_1.csv")
