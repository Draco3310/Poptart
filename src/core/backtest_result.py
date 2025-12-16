from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BacktestResult:
    """
    Standardized container for backtest outputs.
    """

    # Metadata
    run_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    config: Dict[str, Any]
    pair: Optional[str] = None  # Legacy single-pair support
    pairs: List[str] = field(default_factory=list)  # Multi-pair support

    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    # Detailed Logs (Optional, for debugging)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the result."""
        return {
            "run_id": self.run_id,
            "pair": self.pair,
            "pairs": self.pairs,
            "metrics": self.metrics,
            "trades_count": len(self.trades),
            "final_equity": self.metrics.get("final_equity", 0.0),
            "pnl_percent": self.metrics.get("pnl_percent", 0.0),
        }
