import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from vnpy_lite.app.algo_trading.engine import BacktestEngine

def test_engine():
    engine = BacktestEngine()

    symbols = [f"Stock_{i}" for i in range(50)]
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    print("Running backtest...")
    trades = engine.run_backtest(symbols, start_date, end_date)

    print("\nMetrics:")
    print(engine.metrics)

    if not trades.empty:
        print("\nLast 5 Trades:")
        print(trades.tail())

        print("\nDaily Stats Head:")
        print(engine.daily_stats.head())
        print("\nDaily Stats Tail:")
        print(engine.daily_stats.tail())

if __name__ == "__main__":
    test_engine()
