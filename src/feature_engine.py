import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Feature engineering class for Chinese A-share market data.
    Focuses on generating features for XGBoost model to predict short-term price spikes.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame containing stock data.
        Expected columns: ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        Optional columns: ['buy_vol', 'sell_vol', 'outvol', 'invol']
        """
        # Work on a copy to avoid SettingWithCopy warnings on original df
        self.df = df.copy()

        # Ensure date is datetime if needed, though mostly we rely on index or order
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.sort_values('date', inplace=True)

        self.df.reset_index(drop=True, inplace=True)

    def generate_labels(self):
        """
        Generate the target label.
        Logic: Look ahead 3 days (t+1 to t+3).
        Condition: If (Max(High_t+1, High_t+2, High_t+3) / Close_t) - 1 >= 0.08, label = 1, else 0.
        """
        # Shift high prices to align t+1, t+2, t+3 with row t
        h1 = self.df['high'].shift(-1)
        h2 = self.df['high'].shift(-2)
        h3 = self.df['high'].shift(-3)

        # Calculate max high over the next 3 days
        # Use skipna=False to strictly require data for all 3 future days
        max_future_high = pd.concat([h1, h2, h3], axis=1).max(axis=1, skipna=False)

        # Calculate return
        future_return = (max_future_high / self.df['close']) - 1

        # Create label
        # We want to ensure we don't label the last few rows as 0 just because data is missing.
        # So we keep it as float/NaN first, drop NaNs later, then cast to int.
        self.df['label'] = np.where(future_return.isna(), np.nan, (future_return >= 0.08).astype(float))

    def _calculate_kdj(self):
        """
        Logic A: KDJ & RSV System
        RSV = (CLOSE - min(LOW, 9)) / (max(HIGH, 9) - min(LOW, 9)) * 100
        K = 2/3 * K_prev + 1/3 * RSV
        D = 2/3 * D_prev + 1/3 * K
        J = 3*K - 2*D
        """
        low_min = self.df['low'].rolling(window=9).min()
        high_max = self.df['high'].rolling(window=9).max()

        # RSV calculation
        # Handle potential division by zero if high_max == low_min
        rsv = (self.df['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50) # Default to 50 if range is 0 or NaN at start

        # Calculate K, D using EWM (alpha=1/3) which corresponds to SMA logic in Chinese software
        # adjust=False ensures the recursive calculation: y_t = (1-alpha)*y_{t-1} + alpha*x_t
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d

        self.df['kdj_k'] = k
        self.df['kdj_d'] = d
        self.df['kdj_j'] = j

        # Features to Extract
        self.df['kdj_j_diff'] = self.df['kdj_j'].diff()

        # kdj_v_turn: J turned upward from a low position (e.g., J_{t-1} < 30 AND J_t > J_{t-1})
        j_prev = self.df['kdj_j'].shift(1)
        # Using 30 as a threshold for "low position" as per common KDJ strategies, though "low position" can be subjective.
        # The prompt says "e.g., J_{t-1} < 30", so I'll stick to that example logic.
        self.df['kdj_v_turn'] = ((j_prev < 30) & (self.df['kdj_j'] > j_prev)).astype(int)

    def _calculate_ma20(self):
        """
        Logic B: Moving Average Trend (MA20)
        """
        ma20 = self.df['close'].rolling(window=20).mean()
        self.df['ma20'] = ma20

        ma20_prev = ma20.shift(1)

        # ma20_trend: (MA20_t - MA20_{t-1}) / MA20_{t-1}
        self.df['ma20_trend'] = (ma20 - ma20_prev) / ma20_prev

        # price_position: (Close_t - MA20_t) / MA20_t
        self.df['price_position'] = (self.df['close'] - ma20) / ma20

    def _calculate_zdp(self):
        """
        Logic C: Capital Flow (ZDP - The Core Feature)
        Prioritize buy_vol/sell_vol -> outvol/invol -> Fallback approximation
        """
        cols = self.df.columns

        if 'buy_vol' in cols and 'sell_vol' in cols:
            raw_zdp = (self.df['buy_vol'] - self.df['sell_vol']) / self.df['volume']
        elif 'outvol' in cols and 'invol' in cols:
            raw_zdp = (self.df['outvol'] - self.df['invol']) / self.df['volume']
        else:
            # Fallback: ((Close - Low) - (High - Close)) / (High - Low)
            # Simplifies to (2*Close - High - Low) / (High - Low)
            numerator = 2 * self.df['close'] - self.df['high'] - self.df['low']
            denominator = self.df['high'] - self.df['low']
            # Avoid division by zero
            raw_zdp = numerator / denominator.replace(0, np.nan)
            raw_zdp = raw_zdp.fillna(0) # If High=Low, force 0

        # Smooth: ZDP = MA(RAW_ZDP, 3)
        zdp = raw_zdp.rolling(window=3).mean()
        self.df['zdp_value'] = zdp

        # Signal line: MAZDP = MA(ZDP, 6)
        mazdp = zdp.rolling(window=6).mean()
        self.df['mazdp'] = mazdp

        # Features to Extract
        self.df['zdp_divergence'] = zdp - mazdp

        # zdp_golden_cross: True if ZDP crosses above MAZDP while ZDP < 0
        zdp_prev = zdp.shift(1)
        mazdp_prev = mazdp.shift(1)

        cross_above = (zdp > mazdp) & (zdp_prev <= mazdp_prev)
        condition_zdp_neg = (zdp < 0)

        self.df['zdp_golden_cross'] = (cross_above & condition_zdp_neg).astype(int)

    def _calculate_market_checks(self):
        """
        Logic D: Market Cap & Volume Checks
        """
        # vol_ratio: Volume_t / MA(Volume, 20)
        vol_ma20 = self.df['volume'].rolling(window=20).mean()
        self.df['vol_ratio'] = self.df['volume'] / vol_ma20

        # macd_dif: EMA(12) - EMA(26)
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd_dif'] = ema12 - ema26

    def process(self):
        """
        Execute the full feature engineering pipeline.
        Returns a clean DataFrame ready for XGBoost.
        """
        # Generate Label
        self.generate_labels()

        # Feature Engineering
        self._calculate_kdj()
        self._calculate_ma20()
        self._calculate_zdp()
        self._calculate_market_checks()

        # Data Cleaning
        # Drop rows with NaN values (generated by rolling windows)
        # The prompt says "Handle NaN values generated by rolling windows (drop them)."
        # Also need to drop the last 3 rows because label generation used shift(-3), so they might be NaN or invalid if we care about label correctness.
        # But wait, logic: "Ensure the output dataframe contains only numeric features and the label".

        # Drop columns that are not features or label (optional, but requested "ready for XGBoost")
        # I'll keep the date and code for reference if needed, but strictly for XGBoost usually we drop them.
        # However, usually we keep them as index or meta columns. The prompt says "contains only numeric features and the label".
        # I will keep the original OHLCV columns? "transform raw stock data into a feature-rich dataset".
        # Usually we keep the features. I'll drop the intermediate columns like 'kdj_k', 'kdj_d', 'ma20', 'mazdp' if they weren't requested as final features.
        # Requested features:
        # KDJ: kdj_j, kdj_j_diff, kdj_v_turn
        # MA20: ma20_trend, price_position
        # ZDP: zdp_value, zdp_divergence, zdp_golden_cross
        # Market: vol_ratio, macd_dif

        feature_cols = [
            'kdj_j', 'kdj_j_diff', 'kdj_v_turn',
            'ma20_trend', 'price_position',
            'zdp_value', 'zdp_divergence', 'zdp_golden_cross',
            'vol_ratio', 'macd_dif',
            'label'
        ]

        # Drop NaNs first from the calculated columns
        # Rolling 26 (MACD) is likely the largest window needed to be valid (or 20 for MA).
        # Actually MACD uses EMA, so it starts from the beginning but stabilizes later.
        # MA20 needs 20 rows.
        # Shift(-3) for label means the last 3 rows have unknown labels (or incomplete).
        # We should drop the last 3 rows because we don't have ground truth for them.

        # However, `generate_labels` logic: `max_future_high` will be NaN for the last few rows.
        # So `dropna()` will handle it if we select columns first.

        final_df = self.df[feature_cols].dropna()
        final_df['label'] = final_df['label'].astype(int)

        return final_df

def generate_mock_data(n_rows=100):
    """
    Generate mock stock data for testing.
    """
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='D')
    data = {
        'date': dates,
        'code': ['000001'] * n_rows,
        'open': np.random.uniform(10, 20, n_rows),
        'high': np.random.uniform(10, 20, n_rows),
        'low': np.random.uniform(10, 20, n_rows),
        'close': np.random.uniform(10, 20, n_rows),
        'volume': np.random.uniform(1000, 10000, n_rows),
        'amount': np.random.uniform(10000, 100000, n_rows),
        # Optional columns for ZDP testing
        'buy_vol': np.random.uniform(500, 5000, n_rows),
        'sell_vol': np.random.uniform(500, 5000, n_rows)
    }

    # Fix High/Low consistency
    df = pd.DataFrame(data)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df

if __name__ == "__main__":
    print("Generating mock data...")
    df = generate_mock_data(200)
    print(f"Mock data shape: {df.shape}")
    print(df.head())

    print("\nRunning Feature Engineering...")
    fe = FeatureEngineer(df)
    final_df = fe.process()

    print(f"\nFinal DataFrame shape: {final_df.shape}")
    print(final_df.head())
    print("\nColumns:", final_df.columns.tolist())

    print("\nCheck label distribution:")
    print(final_df['label'].value_counts())

    print("\nScript completed successfully.")
