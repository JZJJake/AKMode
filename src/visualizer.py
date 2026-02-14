import pandas as pd
import mplfinance as mpf
import os
import matplotlib.pyplot as plt

from app_config.settings import DATA_DIR

CHARTS_DIR = os.path.join(DATA_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def plot_prediction(code: str, target_date: str, df: pd.DataFrame, prob: float):
    """
    Plots the last 30 days K-Line for a stock with a marker on the target date.
    Adds a subplot for True Flow Ratio.

    Args:
        code: Stock code.
        target_date: The prediction date (YYYYMMDD).
        df: DataFrame containing OHLCV and True_Flow_Ratio. Must have DatetimeIndex.
        prob: Predicted probability.
    """
    if df.empty:
        return

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            return # Cannot plot without date index

    # Filter last 30 days up to target_date
    target_dt = pd.to_datetime(target_date)
    start_dt = target_dt - pd.Timedelta(days=45) # roughly 30 trading days

    plot_df = df.loc[start_dt:target_dt].copy()

    if plot_df.empty:
        return

    # Add subplot for Flow Ratio
    # By default, mpf has panel 0 (OHLC) and panel 1 (Volume).
    # We add True_Flow_Ratio as panel 2.

    ap = [
        mpf.make_addplot(plot_df['True_Flow_Ratio'], panel=2, color='purple', secondary_y=False, ylabel="Flow Ratio"),
    ]

    # Save path
    date_charts_dir = os.path.join(CHARTS_DIR, target_date)
    os.makedirs(date_charts_dir, exist_ok=True)
    save_path = os.path.join(date_charts_dir, f"{code}_prob_{prob:.2f}.png")

    # Custom Style
    s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 10})

    # Plot
    # Need to handle case where mplfinance expects 'Open', 'High', 'Low', 'Close', 'Volume' columns exactly.
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in plot_df.columns for col in required_cols):
        # Renaming might be needed if columns are lowercase, but my mock gen uses capitalized.
        pass

    try:
        mpf.plot(
            plot_df,
            type='candle',
            volume=True,
            addplot=ap,
            title=f"{code} Prediction: {prob:.2%}",
            style=s,
            savefig=dict(fname=save_path, dpi=100, bbox_inches='tight'),
            tight_layout=True
        )
        print(f"Chart saved to {save_path}")
    except Exception as e:
        print(f"Failed to plot chart for {code}: {e}")
