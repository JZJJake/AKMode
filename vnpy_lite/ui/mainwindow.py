import sys
from datetime import date, timedelta
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QDateEdit, QSpinBox,
                               QDoubleSpinBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QPalette

from ..app.algo_trading.engine import BacktestEngine
from .widget import ChartWidget
import pandas as pd

class BacktestWorker(QThread):
    finished_signal = Signal(object, object) # trades_df, engine
    progress_signal = Signal(str)

    def __init__(self, symbols, start_date, end_date, capital, sl_pct, tp_pct, hold_days):
        super().__init__()
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.hold_days = hold_days
        self.engine = BacktestEngine()

    def run(self):
        self.progress_signal.emit("Running backtest...")
        trades_df = self.engine.run_backtest(
            self.symbols,
            self.start_date,
            self.end_date,
            self.capital,
            self.sl_pct,
            self.tp_pct,
            self.hold_days
        )
        self.finished_signal.emit(trades_df, self.engine)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VN.PY Lite - Alpha Strategy Backtester")
        self.resize(1200, 800)

        # Apply Dark Theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #dcdcdc; }
            QWidget { background-color: #1e1e1e; color: #dcdcdc; }
            QLabel { color: #dcdcdc; font-size: 14px; }
            QPushButton { background-color: #3e3e3e; color: #ffffff; border: 1px solid #5e5e5e; padding: 5px; }
            QPushButton:hover { background-color: #4e4e4e; }
            QTableWidget { background-color: #2e2e2e; color: #dcdcdc; gridline-color: #4e4e4e; }
            QHeaderView::section { background-color: #3e3e3e; color: #dcdcdc; padding: 4px; }
            QLineEdit, QDateEdit, QSpinBox, QDoubleSpinBox { background-color: #2e2e2e; color: #dcdcdc; border: 1px solid #4e4e4e; }
        """)

        self.worker = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---------------------------
        # Left Panel (Config)
        # ---------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(250)

        left_layout.addWidget(QLabel("Backtest Configuration"))

        # Symbol Count
        left_layout.addWidget(QLabel("Symbol Count (Mock):"))
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 1000)
        self.spin_count.setValue(50)
        left_layout.addWidget(self.spin_count)

        # Dates
        left_layout.addWidget(QLabel("Start Date:"))
        self.date_start = QDateEdit()
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_start.setDate(date.today() - timedelta(days=365))
        left_layout.addWidget(self.date_start)

        left_layout.addWidget(QLabel("End Date:"))
        self.date_end = QDateEdit()
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setDate(date.today())
        left_layout.addWidget(self.date_end)

        # Parameters
        left_layout.addWidget(QLabel("Initial Capital:"))
        self.spin_capital = QDoubleSpinBox()
        self.spin_capital.setRange(10000, 100000000)
        self.spin_capital.setValue(1000000)
        left_layout.addWidget(self.spin_capital)

        left_layout.addWidget(QLabel("Stop Loss %:"))
        self.spin_sl = QDoubleSpinBox()
        self.spin_sl.setRange(0.01, 0.5)
        self.spin_sl.setSingleStep(0.01)
        self.spin_sl.setValue(0.05)
        left_layout.addWidget(self.spin_sl)

        left_layout.addWidget(QLabel("Take Profit %:"))
        self.spin_tp = QDoubleSpinBox()
        self.spin_tp.setRange(0.01, 1.0)
        self.spin_tp.setSingleStep(0.01)
        self.spin_tp.setValue(0.10)
        left_layout.addWidget(self.spin_tp)

        left_layout.addWidget(QLabel("Holding Days:"))
        self.spin_hold = QSpinBox()
        self.spin_hold.setRange(1, 50)
        self.spin_hold.setValue(5)
        left_layout.addWidget(self.spin_hold)

        # Run Button
        self.btn_run = QPushButton("Run Backtest")
        self.btn_run.clicked.connect(self.run_backtest)
        left_layout.addWidget(self.btn_run)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        left_layout.addStretch()

        # ---------------------------
        # Right Panel (Chart & Results)
        # ---------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Chart
        self.chart_widget = ChartWidget()
        right_layout.addWidget(self.chart_widget, stretch=2)

        # Metrics Table
        self.table_metrics = QTableWidget()
        self.table_metrics.setColumnCount(2)
        self.table_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.table_metrics, stretch=1)

        # Trades Table (Optional, maybe show last 100)
        self.table_trades = QTableWidget()
        self.table_trades.setColumnCount(5)
        self.table_trades.setHorizontalHeaderLabels(["Symbol", "Entry Date", "Exit Date", "PnL %", "Reason"])
        self.table_trades.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.table_trades, stretch=1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def run_backtest(self):
        # Disable button
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate

        # Get Parameters
        count = self.spin_count.value()
        symbols = [f"Stock_{i:03d}" for i in range(count)]
        start_date = self.date_start.date().toString("yyyy-MM-dd")
        end_date = self.date_end.date().toString("yyyy-MM-dd")
        capital = self.spin_capital.value()
        sl_pct = self.spin_sl.value()
        tp_pct = self.spin_tp.value()
        hold_days = self.spin_hold.value()

        # Start Thread
        self.worker = BacktestWorker(symbols, start_date, end_date, capital, sl_pct, tp_pct, hold_days)
        self.worker.finished_signal.connect(self.on_backtest_finished)
        self.worker.start()

    def on_backtest_finished(self, trades_df, engine):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update Metrics Table
        metrics = engine.metrics
        self.table_metrics.setRowCount(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            self.table_metrics.setItem(i, 0, QTableWidgetItem(k))
            self.table_metrics.setItem(i, 1, QTableWidgetItem(str(v)))

        # Update Trades Table
        if not trades_df.empty:
            # Show last 50 trades
            display_df = trades_df.tail(50).sort_values("exit_date", ascending=False)
            self.table_trades.setRowCount(len(display_df))
            for i, row in enumerate(display_df.itertuples()):
                self.table_trades.setItem(i, 0, QTableWidgetItem(row.symbol))
                self.table_trades.setItem(i, 1, QTableWidgetItem(str(row.entry_date)))
                self.table_trades.setItem(i, 2, QTableWidgetItem(str(row.exit_date)))

                item_pnl = QTableWidgetItem(f"{row.return_pct:.2f}%")
                if row.pnl > 0:
                    item_pnl.setForeground(QColor("red")) # In China Red is Up/Profit
                else:
                    item_pnl.setForeground(QColor("green")) # Green is Down/Loss

                self.table_trades.setItem(i, 3, item_pnl)
                self.table_trades.setItem(i, 4, QTableWidgetItem(row.reason))

            # Update Chart with ONE example stock that had a signal
            # Find a stock with signals
            # We need to access the engine's last processed data or re-load one.
            # But the engine doesn't store all DF in memory.
            # So let's re-generate data for the first stock in trades list to show it.
            first_symbol = trades_df['symbol'].iloc[0]
            start = self.date_start.date().toString("yyyy-MM-dd")
            end = self.date_end.date().toString("yyyy-MM-dd")

            # Load and process again to get indicators for charting
            from ..trader.data_loader import generate_mock_data
            df_chart = generate_mock_data(first_symbol, start, end)
            df_chart = engine.strategy.generate_signals(df_chart) # Add indicators

            # Filter chart to show interesting period? No, show all.
            self.chart_widget.update_chart(df_chart, trades_df[trades_df['symbol'] == first_symbol])

        else:
            QMessageBox.information(self, "No Trades", "No trades were generated with current parameters.")
