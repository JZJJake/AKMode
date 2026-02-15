import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates mock OHLCV data with extended fields for strategy testing.
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    # 1. Price Generation (Random Walk)
    np.random.seed(hash(symbol) % 2**32)  # Consistent per symbol
    returns = np.random.normal(0, 0.02, n)
    price_start = np.random.uniform(10, 100)
    prices = price_start * (1 + returns).cumprod()

    opens = prices * (1 + np.random.normal(0, 0.005, n))
    closes = prices * (1 + np.random.normal(0, 0.005, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n)))

    # 2. Volume & Turnover
    volume = np.random.lognormal(16, 0.5, n)  # e.g., ~9M
    turnover = volume * closes  # Simple approximation

    # 3. Market Cap & Net Profit
    # Randomly assign a market cap regime for the stock
    is_small_cap = np.random.random() > 0.5
    if is_small_cap:
        market_cap = np.random.uniform(50, 340, n) # Under 350 Yi
    else:
        market_cap = np.random.uniform(360, 1000, n) # Over 350 Yi

    # Occasionally generate a loss-making stock (20% chance)
    is_profitable = np.random.random() > 0.2
    if is_profitable:
        net_profit = np.random.uniform(1, 50, n) # Positive profit
    else:
        net_profit = np.random.uniform(-10, -1, n) # Negative profit

    # 4. Out/In Volume (ZDP Logic)
    # Sine wave to force oscillation for ZDP signals
    # Period of sine wave: e.g., 50 days
    t = np.arange(n)
    # Use a faster sine wave to ensure we see cycles in short periods
    sine_wave = np.sin(t * 2 * np.pi / 20)

    # Add some noise to the sine wave
    sine_wave += np.random.normal(0, 0.1, n)
    sine_wave = np.clip(sine_wave, -1, 1)

    total_active_ratio = np.random.uniform(0.6, 0.9, n)
    # Map sine wave (-1 to 1) to buy ratio (0.1 to 0.9)
    # 0.5 + (-1 * 0.4) = 0.1
    # 0.5 + (1 * 0.4) = 0.9
    buy_ratio = 0.5 + (sine_wave * 0.4)

    out_volume = volume * total_active_ratio * buy_ratio
    in_volume = volume * total_active_ratio * (1 - buy_ratio)

    # Create DataFrame
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
        "turnover": turnover,
        "out_volume": out_volume,
        "in_volume": in_volume,
        "market_cap": market_cap,
        "net_profit": net_profit,
        "symbol": symbol
    }, index=dates)

    df.index.name = "date"

    return df

if __name__ == "__main__":
    # Test
    df = generate_mock_data("TestSymbol", "2023-01-01", "2023-06-01")
    print(df.head())
    print(df[['out_volume', 'in_volume', 'volume']].head())

    # Verify ZDP logic briefly
    raw_zdp = ((df['out_volume'] - df['in_volume']) / df['volume']) * 100
    print("\nRaw ZDP Head:\n", raw_zdp.head())
    print("\nMin ZDP:", raw_zdp.min())
    print("Max ZDP:", raw_zdp.max())
