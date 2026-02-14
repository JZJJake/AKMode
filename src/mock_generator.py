import pandas as pd
import numpy as np
import os
import random
from app_config.settings import FEATURES_DIR

def generate_mock_stock_data(code: str, start_date: str = '2023-01-01', end_date: str = '2023-12-31', rally_prob: float = 0.05):
    """
    Generates synthetic OHLCV data for a stock with random walk + occasional rallies.

    Args:
        code: Stock code.
        start_date: Start date.
        end_date: End date.
        rally_prob: Probability of a 'pump' event (e.g., 3-day rally > 10%).
    """
    date_range = pd.date_range(start_date, end_date, freq='B') # Business days
    n_days = len(date_range)

    # Base price random walk
    initial_price = 10.0
    returns = np.random.normal(0, 0.02, n_days) # Daily return mean 0, std 2%
    prices = initial_price * np.exp(np.cumsum(returns))

    # Volume random walk (correlated with price somewhat)
    initial_vol = 10000
    vol_returns = np.random.normal(0, 0.5, n_days)
    volumes = initial_vol * np.exp(np.cumsum(vol_returns))
    volumes = np.abs(volumes) # Ensure positive

    # Flow Ratio: Random uniform (-0.5, 0.5) usually
    flow_ratios = np.random.uniform(-0.5, 0.5, n_days)

    # Inject Rallies (Labels=1)
    # If a rally occurs at day t, we artificially inflate t+1, t+2, t+3
    # Target: (Close[t+3] - Close[t]) / Close[t] >= 0.10

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({
        'Date': date_range,
        'Code': code,
        'Close': prices,
        'Volume': volumes,
        'True_Flow_Ratio': flow_ratios
    })

    # Generate OHLC (Open=Close[t-1], High/Low random around Close)
    df['Open'] = df['Close'].shift(1).fillna(initial_price)
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))

    # Inject Patterns
    # We iterate and inject a 15% jump over 3 days randomly
    for i in range(10, n_days - 10):
        if random.random() < rally_prob:
            # Rally starts at i
            # i+1: +5%
            # i+2: +5%
            # i+3: +5%
            # Total ~15%
            base_idx = i
            if base_idx + 3 >= n_days:
                continue

            # Current price at t
            p_t = df.loc[base_idx, 'Close']

            # Artificial pump
            df.loc[base_idx+1, 'Close'] = p_t * 1.05
            df.loc[base_idx+2, 'Close'] = p_t * 1.05 * 1.05
            df.loc[base_idx+3, 'Close'] = p_t * 1.05 * 1.05 * 1.05

            # Also increase volume and flow during rally
            df.loc[base_idx:base_idx+3, 'Volume'] *= 2.0
            df.loc[base_idx:base_idx+3, 'True_Flow_Ratio'] = np.random.uniform(0.3, 0.8, 4)

            # Recalculate OHLC for these days roughly
            for j in range(1, 4):
                idx = base_idx + j
                prev_c = df.loc[idx-1, 'Close']
                curr_c = df.loc[idx, 'Close']
                df.loc[idx, 'Open'] = prev_c
                df.loc[idx, 'High'] = max(prev_c, curr_c) * 1.01
                df.loc[idx, 'Low'] = min(prev_c, curr_c) * 0.99

    # Formatting
    df['Date'] = df['Date'].dt.strftime('%Y%m%d')

    # Save
    os.makedirs(FEATURES_DIR, exist_ok=True)
    file_path = os.path.join(FEATURES_DIR, f"{code}.parquet")
    df.to_parquet(file_path, engine='pyarrow', index=False)
    print(f"Generated mock data for {code}: {len(df)} rows.")

if __name__ == "__main__":
    generate_mock_stock_data("000001")
