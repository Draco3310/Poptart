import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Singleton registry for loaded ML models to avoid redundant disk I/O and memory usage.
    """
    _instance: Optional['ModelRegistry'] = None
    _cache: Dict[str, Any] = {}

    def __new__(cls) -> 'ModelRegistry':
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        return cls._cache.get(key)

    @classmethod
    def register(cls, key: str, model: Any) -> None:
        cls._cache[key] = model
        logger.debug(f"Registered model: {key}")

    @classmethod
    def clear(cls) -> None:
        """Clears the registry. Useful for testing."""
        cls._cache.clear()
        logger.info("Model registry cleared.")
