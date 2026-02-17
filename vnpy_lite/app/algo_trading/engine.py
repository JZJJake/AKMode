import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from ...trader.data_loader import generate_mock_data
from .strategy import CompositeStrategy

class BacktestEngine:
    """
    Engine to run the backtest using vectorized strategy signals and
    sequential execution logic.
    """

    def __init__(self):
        self.strategy = CompositeStrategy()
        self.results = {}
        self.daily_stats = pd.DataFrame()
        self.metrics = {}
        self.trades_df = pd.DataFrame()

    def run_backtest(self,
                     symbols: List[str],
                     start_date: str,
                     end_date: str,
                     capital: float = 1_000_000,
                     stop_loss_pct: float = 0.05,
                     take_profit_pct: float = 0.10,
                     holding_period: int = 5):
        """
        Runs the backtest across multiple symbols.
        """
        all_trades = []

        print(f"Starting backtest for {len(symbols)} symbols...")

        for symbol in symbols:
            # 1. Load Data
            df = generate_mock_data(symbol, start_date, end_date)

            # 2. Generate Signals (Vectorized)
            df = self.strategy.generate_signals(df)

            # 3. Execution Loop (Stateful)
            trades = self._execute_strategy(df, symbol, stop_loss_pct, take_profit_pct, holding_period)
            all_trades.extend(trades)

        # 4. Calculate Aggregate Stats
        self.trades_df = pd.DataFrame(all_trades)
        if not self.trades_df.empty:
            self.calculate_daily_stats(self.trades_df, start_date, end_date, capital)
        else:
            self.daily_stats = pd.DataFrame(index=pd.bdate_range(start=start_date, end=end_date))
            self.daily_stats['balance'] = capital
            self.metrics = {
                "Total Return": "0.00%",
                "Win Rate": "0.00%",
                "Total Trades": 0,
                "Max Drawdown": "0.00"
            }

        return self.trades_df

    def _execute_strategy(self,
                          df: pd.DataFrame,
                          symbol: str,
                          sl_pct: float,
                          tp_pct: float,
                          hold_days: int) -> List[Dict]:
        """
        Executes trades for a single symbol based on signals.
        """
        trades = []
        holding = False
        entry_price = 0.0
        entry_date = None
        days_held = 0
        pending_entry = False

        # Iterate over rows
        for row in df.itertuples():
            current_date = row.Index

            # 1. Process Entry
            if pending_entry and not holding:
                entry_price = row.open
                entry_date = current_date
                holding = True
                days_held = 0
                pending_entry = False # Consumed

                # Check for immediate exit on same day?
                # Conservative: Yes.
                # If Low < SL, exit immediately.

                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)

                exit_price = 0.0
                exit_reason = ""

                if row.low <= sl_price:
                     exit_price = sl_price # Slippage ignored
                     exit_reason = "Stop Loss (Intraday)"
                elif row.high >= tp_price:
                     exit_price = tp_price
                     exit_reason = "Take Profit (Intraday)"

                if exit_price > 0:
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": exit_price,
                        "reason": exit_reason,
                        "pnl": pnl,
                        "return_pct": pnl * 100,
                        "days_held": 0
                    })
                    holding = False
                    entry_price = 0.0
                    entry_date = None
                    continue # Move to next day (cannot re-enter same day)

            # 2. Process Holding (if still holding after entry check)
            if holding:
                days_held += 1

                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)

                exit_price = 0.0
                exit_reason = ""

                if row.low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "Stop Loss"
                elif row.high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "Take Profit"
                elif days_held >= hold_days:
                    exit_price = row.close # Exit at Close of 5th day
                    exit_reason = "Time Exit"

                if exit_price > 0:
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": exit_price,
                        "reason": exit_reason,
                        "pnl": pnl,
                        "return_pct": pnl * 100,
                        "days_held": days_held
                    })
                    holding = False
                    entry_price = 0.0
                    entry_date = None

            # 3. Check for New Signal
            # Only if not holding (No Pyramiding)
            # If we just exited, `holding` is False, so we can check for signal to enter tomorrow.
            if not holding:
                if hasattr(row, 'signal') and row.signal:
                    pending_entry = True

        return trades

    def calculate_daily_stats(self, trades_df: pd.DataFrame, start_date: str, end_date: str, capital: float):
        """
        Calculates daily equity curve and metrics.
        """
        dates = pd.bdate_range(start=start_date, end=end_date)

        # Aggregate PnL by exit date
        if not trades_df.empty:
            # Assume 10% capital per trade for simplicity in simulation?
            # Or assume we invest 'capital' divided by N?
            # User didn't specify position sizing.
            # I'll assume a fixed bet size (e.g. 20% of capital) for PnL curve

            position_size = capital * 0.2 # 5 positions max roughly

            # Map trades to exit dates
            # Group by exit date
            grouped = trades_df.groupby('exit_date')['pnl'].sum()

            # Convert Index to Datetime if needed (it should be)
            # Align with dates
            aligned_pnl = grouped.reindex(dates, fill_value=0)

            # Daily PnL ($) = Sum(PnL%) * Position_Size
            daily_dollar_pnl = aligned_pnl * position_size

            # Cumulative
            self.daily_stats = pd.DataFrame(index=dates)
            self.daily_stats['daily_pnl'] = daily_dollar_pnl
            self.daily_stats['balance'] = capital + daily_dollar_pnl.cumsum()
            self.daily_stats['drawdown'] = self.daily_stats['balance'] - self.daily_stats['balance'].cummax()

            # Calculate metrics
            total_return = (self.daily_stats['balance'].iloc[-1] / capital) - 1
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)

            self.metrics = {
                "Total Return": f"{total_return*100:.2f}%",
                "Win Rate": f"{win_rate*100:.2f}%",
                "Total Trades": len(trades_df),
                "Max Drawdown": f"{self.daily_stats['drawdown'].min():.2f}"
            }
        else:
             self.daily_stats = pd.DataFrame(index=dates)
             self.daily_stats['balance'] = capital
