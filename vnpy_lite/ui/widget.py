import pandas as pd
import mplfinance as mpf
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = None

        # Initialize with empty chart
        self.figure = plt.figure(figsize=(10, 8), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()

    def update_chart(self, df: pd.DataFrame, trades: pd.DataFrame = None):
        """
        Updates the chart with new data.
        """
        # Clear previous
        self.layout.removeWidget(self.canvas)
        self.canvas.close()
        plt.close(self.figure)

        # Prepare Data
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Create AddPlots
        # 1. MA20 (on main panel)
        ap = []
        if 'close' in df.columns:
            ma20 = df['close'].rolling(window=20).mean()
            ap.append(mpf.make_addplot(ma20, panel=0, color='orange', width=1, ylabel='Price'))

        # 2. ZDP (Subpanel 1)
        if 'zdp' in df.columns and 'mazdp' in df.columns:
            ap.append(mpf.make_addplot(df['zdp'], panel=2, color='cyan', width=1, ylabel='ZDP'))
            ap.append(mpf.make_addplot(df['mazdp'], panel=2, color='yellow', width=1))
            # Add line at -4 and 0?
            # mpf doesn't support hlines in addplot easily, but acceptable.

        # 3. KDJ (Subpanel 2)
        if 'k' in df.columns and 'd' in df.columns and 'j' in df.columns:
            ap.append(mpf.make_addplot(df['k'], panel=3, color='white', width=1, ylabel='KDJ'))
            ap.append(mpf.make_addplot(df['d'], panel=3, color='yellow', width=1))
            ap.append(mpf.make_addplot(df['j'], panel=3, color='magenta', width=1))

        # 4. Signals (Markers)
        if 'signal' in df.columns:
            # Create a series of NaNs, populate with Low * 0.99 where signal is True
            signals = pd.Series(float('nan'), index=df.index)
            signals[df['signal']] = df['low'] * 0.98

            # Only add if there are signals
            if not signals.isna().all():
                ap.append(mpf.make_addplot(signals, type='scatter', markersize=50, marker='^', color='red', panel=0))

        # Trades (Markers for Entry/Exit)?
        # For simplicity, just show Signals.

        # Plot
        # panel_ratios = (Main, Volume, ZDP, KDJ)
        # Defaults: Main=0, Vol=1. We add 2 and 3.

        # Style
        style = mpf.make_mpf_style(base_mpf_style='nightclouds', rc={'axes.labelsize': 8})

        self.figure, axes = mpf.plot(df, type='candle', style=style,
                                     volume=True,
                                     addplot=ap,
                                     returnfig=True,
                                     panel_ratios=(4, 1, 1, 1),
                                     tight_layout=True,
                                     show_nontrading=False)

        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()
