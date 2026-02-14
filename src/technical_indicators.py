import pandas as pd
import numpy as np

def calculate_macd(df: pd.DataFrame, slow=26, fast=12, signal=9):
    """
    Calculates MACD Histogram.
    MACD = EMA(12) - EMA(26)
    Signal = EMA(MACD, 9)
    Hist = MACD - Signal
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd_hist

def calculate_kdj_j(df: pd.DataFrame, n=9, m1=3, m2=3):
    """
    Calculates KDJ_J indicator.
    RSV = (Close - Lowest_Low_n) / (Highest_High_n - Lowest_Low_n) * 100
    K = SMA(RSV, m1)
    D = SMA(K, m2)
    J = 3*K - 2*D
    """
    # Rolling min/max
    low_min = df['Low'].rolling(window=n).min()
    high_max = df['High'].rolling(window=n).max()

    # Avoid division by zero
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, 1)

    # EWM is closer to standard KDJ calculation than simple rolling mean, usually it's SMA(1/m1) * RSV + (m1-1)/m1 * Prev_K
    # Using Pandas ewm(com=m1-1, adjust=False) approximates the typical Wilder/EMA smoothing if m1=3 -> alpha=1/3
    # Standard KDJ uses SMA on RSV. Let's use EWM with alpha=1/m1.

    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d

    return j
