import pandas as pd
import numpy as np

class CompositeStrategy:
    """
    Vectorized implementation of the Capital Flow (ZDP) + KDJ Strategy.
    """

    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates ZDP and KDJ indicators.
        Returns a DataFrame with added columns.
        """
        # Work on a copy to avoid SettingWithCopy warnings
        df = df.copy()

        # ---------------------------
        # 1. ZDP (Capital Flow)
        # ---------------------------
        # RAW_ZDP = ((OUTVOL - INVOL) / VOL) * 100
        # Protect against zero volume
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_zdp = ((df['out_volume'] - df['in_volume']) / df['volume']) * 100
        raw_zdp = raw_zdp.fillna(0) # or method='ffill'

        # ZDP = MA(RAW_ZDP, 3)
        df['zdp'] = raw_zdp.rolling(window=3).mean()

        # MAZDP = MA(ZDP, 6)
        df['mazdp'] = df['zdp'].rolling(window=6).mean()

        # ---------------------------
        # 2. KDJ (N=9, M1=3, M2=3)
        # ---------------------------
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()

        # RSV
        # Avoid division by zero
        denom = high_9 - low_9
        denom = denom.replace(0, np.nan)

        rsv = (df['close'] - low_9) / denom * 100
        rsv = rsv.fillna(50) # Standard fallback

        # K = SMA(RSV, 3, 1) -> alpha = 1/3
        df['k'] = rsv.ewm(alpha=1/3, adjust=False).mean()

        # D = SMA(K, 3, 1) -> alpha = 1/3
        df['d'] = df['k'].ewm(alpha=1/3, adjust=False).mean()

        # J = 3K - 2D
        df['j'] = 3 * df['k'] - 2 * df['d']

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates Buy signals based on strategy logic.
        Returns the DataFrame with a 'signal' column (boolean).
        """
        # Calculate indicators
        df = self.calculate_indicators(df)

        # ---------------------------
        # Filter Conditions
        # ---------------------------
        # Market Cap < 350 (Yi)
        filter_cap = df['market_cap'] < 350

        # Net Profit > 0
        filter_profit = df['net_profit'] > 0

        # Turnover > 10 Million
        filter_turnover = df['turnover'] > 10_000_000

        # Combined Filter
        valid_universe = filter_cap & filter_profit & filter_turnover

        # ---------------------------
        # ZDP Logic
        # ---------------------------
        # Signal: (ZDP > MAZDP) AND (ZDP < 0) AND (MAZDP < -4)
        zdp_signal = (
            (df['zdp'] > df['mazdp']) &
            (df['zdp'] < 0) &
            (df['mazdp'] < -4)
        )

        # ---------------------------
        # KDJ Logic (V-Turn)
        # ---------------------------
        # Ref(J, 1) <= 30
        j_prev = df['j'].shift(1)
        j_prev2 = df['j'].shift(2)

        cond1 = j_prev <= 30

        # J > Ref(J, 1) (Turning up)
        cond2 = df['j'] > j_prev

        # Ref(J, 1) < Ref(J, 2) (Yesterday was local bottom)
        cond3 = j_prev < j_prev2

        kdj_signal = cond1 & cond2 & cond3

        # ---------------------------
        # Final Signal
        # ---------------------------
        df['signal'] = valid_universe & zdp_signal & kdj_signal

        return df

if __name__ == "__main__":
    # Test with mock data
    import sys
    import os

    # Add project root to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

    from vnpy_lite.trader.data_loader import generate_mock_data

    strategy = CompositeStrategy()

    # Generate data until we get a signal
    print("Searching for signals in mock data...")
    found_signal = False
    for i in range(20):
        symbol = f"Stock_{i}"
        df = generate_mock_data(symbol, "2023-01-01", "2024-01-01")
        df_res = strategy.generate_signals(df)

        signals = df_res[df_res['signal']]
        if not signals.empty:
            print(f"FOUND SIGNAL for {symbol}!")
            print(signals[['close', 'zdp', 'mazdp', 'j', 'market_cap']].head())

            # Show the previous days context for the first signal
            sig_date = signals.index[0]
            start_loc = df_res.index.get_loc(sig_date) - 2
            end_loc = df_res.index.get_loc(sig_date) + 1
            print("\nContext:")
            print(df_res.iloc[start_loc:end_loc][['j', 'zdp', 'mazdp', 'signal']])
            found_signal = True
            break

    if not found_signal:
        print("No signals found in 20 random stocks. Adjust mock parameters or strategy threshold.")
