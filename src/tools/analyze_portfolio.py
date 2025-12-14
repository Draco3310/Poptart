import argparse
import logging
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.core.backtest_analytics import BacktestAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("PortfolioAnalyzer")


def analyze_portfolio(xrp_run_id: str, btc_run_id: str, sol_run_id: str) -> None:
    analytics = BacktestAnalytics()

    runs = {"XRP": xrp_run_id, "BTC": btc_run_id, "SOL": sol_run_id}

    logger.info("Fetching trades for portfolio analysis...")

    all_trades = []

    for symbol, run_id in runs.items():
        trades = analytics.get_trades(run_id)
        if trades.empty:
            logger.warning(f"No trades found for {symbol} (Run {run_id})")
            continue

        # Ensure timestamp is datetime
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades["symbol"] = symbol
        all_trades.append(trades)

    if not all_trades:
        logger.error("No trades found for any pair.")
        return

    # Combine all trades
    df_trades = pd.concat(all_trades)
    df_trades.sort_values("timestamp", inplace=True)

    # Create Daily Equity Curves
    # We assume starting capital of $10,000 per pair (Total $30,000)
    initial_capital_per_pair = 10000.0

    # We need to reconstruct the equity curve.
    # Since we only have trades, we can sum PnL over time.
    # We'll resample to Daily.

    # Create a date range covering all trades
    start_date = df_trades["timestamp"].min().floor("D")
    end_date = df_trades["timestamp"].max().ceil("D")
    date_range = pd.date_range(start_date, end_date, freq="D")

    equity_curves = pd.DataFrame(index=date_range)

    for symbol in runs.keys():
        pair_trades = df_trades[df_trades["symbol"] == symbol].copy()
        if pair_trades.empty:
            equity_curves[symbol] = initial_capital_per_pair
            continue

        # Group by day and sum PnL
        daily_pnl = pair_trades.groupby(pair_trades["timestamp"].dt.floor("D"))["pnl"].sum()

        # Reindex to full range and fill 0
        daily_pnl = daily_pnl.reindex(date_range, fill_value=0.0)

        # Cumulative Sum + Initial Capital
        equity_curves[symbol] = initial_capital_per_pair + daily_pnl.cumsum()

    # Calculate Portfolio Equity
    equity_curves["Portfolio"] = equity_curves.sum(axis=1)

    # Calculate Drawdowns
    drawdowns = {}
    for col in equity_curves.columns:
        peak = equity_curves[col].cummax()
        dd = (equity_curves[col] - peak) / peak
        drawdowns[col] = dd.min()

    # Calculate Correlations of Daily Returns
    returns = equity_curves.pct_change().dropna()
    correlation_matrix = returns[["XRP", "BTC", "SOL"]].corr()

    # Output Results
    print("\n========================================")
    print("       PORTFOLIO ANALYSIS REPORT        ")
    print("========================================")

    print("\n--- Performance ---")
    for col in equity_curves.columns:
        final_eq = equity_curves[col].iloc[-1]
        start_eq = equity_curves[col].iloc[0]
        pnl = final_eq - start_eq
        pnl_pct = (pnl / start_eq) * 100
        print(f"{col:<10} | Start: ${start_eq:,.2f} | End: ${final_eq:,.2f} | PnL: ${pnl:,.2f} ({pnl_pct:.2f}%)")

    print("\n--- Max Drawdowns ---")
    for col, dd in drawdowns.items():
        print(f"{col:<10} | {dd:.2%}")

    print("\n--- Correlation Matrix (Daily Returns) ---")
    print(correlation_matrix)

    # Verification Check
    portfolio_dd = drawdowns["Portfolio"]
    sum([drawdowns[s] for s in runs.keys()])
    # Note: Sum of drawdowns is negative (e.g. -0.10 + -0.05 = -0.15).
    # We want Portfolio DD (e.g. -0.08) to be "better" (higher) than sum.
    # Or usually we compare Portfolio DD vs Weighted Average DD.
    # The prompt says: "Verify that the portfolio drawdown is lower than the sum of individual drawdowns."
    # This phrasing is slightly ambiguous. "Lower" usually means "smaller magnitude" (closer to 0) or "more negative"?
    # Usually diversification reduces risk, so Portfolio DD % < Max(Individual DD %).
    # Or Portfolio DD $ < Sum(Individual DD $).
    # Let's assume it means the magnitude is smaller than the worst individual, or significantly reduced.

    print("\n--- Verification ---")
    print(f"Portfolio Max DD: {portfolio_dd:.2%}")
    worst_individual = min(drawdowns.values())
    print(f"Worst Individual DD: {worst_individual:.2%}")

    if portfolio_dd > worst_individual:
        print("✅ Portfolio Drawdown is better (smaller magnitude) than the worst individual drawdown.")
    else:
        print("⚠️ Portfolio Drawdown is NOT better than the worst individual drawdown.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xrp", required=True, help="XRP Run ID")
    parser.add_argument("--btc", required=True, help="BTC Run ID")
    parser.add_argument("--sol", required=True, help="SOL Run ID")
    args = parser.parse_args()

    analyze_portfolio(args.xrp, args.btc, args.sol)
