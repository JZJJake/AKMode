import pandas as pd
import numpy as np
import os
import glob
from app_config.settings import FEATURES_DIR, DATA_DIR
from src.technical_indicators import calculate_macd, calculate_kdj_j

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

class FeatureEngine:
    """
    Processes Parquet features into model-ready tensors.
    Logic (Index-Based):
    1. Read Feature Files (sorted by date).
    2. Calculate Technicals (MACD, KDJ).
    3. Calculate Labels (3D Return > 10% using index shift).
    4. Apply NMS (Skip i+1, i+2, i+3 if i is positive).
    5. Construct Windows (i-5 to i).
    6. Normalize Window Data.
    7. Save X, y.
    """

    def __init__(self, window_size=6):
        self.window_size = window_size
        self.feature_files = glob.glob(os.path.join(FEATURES_DIR, "*.parquet"))

    def process_all(self):
        all_X = []
        all_y = []
        metadata = []

        for file_path in self.feature_files:
            code = os.path.basename(file_path).replace(".parquet", "")
            try:
                df = pd.read_parquet(file_path)
                df = df.sort_values('Date').reset_index(drop=True)

                # Calculate Technicals
                df['MACD_Hist'] = calculate_macd(df)
                df['KDJ_J'] = calculate_kdj_j(df)

                # Ensure float for Volume
                df['Volume'] = df['Volume'].astype(float)

                # Label Generation: Future 3D Return
                # (Close[t+3] - Close[t]) / Close[t]
                # Shift(-3) gets Close[t+3] at row t.
                future_close = df['Close'].shift(-3)
                df['Future_3D_Return'] = (future_close - df['Close']) / df['Close']
                df['Label'] = (df['Future_3D_Return'] >= 0.10).astype(int)

                # NMS Logic
                # Iterate through index i.
                # If i is positive label, skip until i + 3.

                skip_until = -1

                # Loop through valid indices where window and label can be computed
                # Start: window_size - 1 (need 5 prior days)
                # End: len(df) - 3 (need 3 future days for label)

                for i in range(self.window_size - 1, len(df) - 3):
                    if i <= skip_until:
                        continue

                    label = df.loc[i, 'Label']

                    # Window Extraction: [i-5, i-4, i-3, i-2, i-1, i] (Size 6)
                    window_df = df.iloc[i - self.window_size + 1 : i + 1].copy()

                    if len(window_df) != self.window_size:
                        continue

                    # Extract raw values for normalization
                    prices = window_df['Close'].values
                    volumes = window_df['Volume'].values
                    flows = window_df['True_Flow_Ratio'].values
                    macd_hists = window_df['MACD_Hist'].values
                    kdj_js = window_df['KDJ_J'].values

                    # Normalization
                    # 1. Norm_Price: (P_i / P_0) - 1
                    p0 = prices[0]
                    if p0 == 0: continue
                    norm_prices = (prices / p0) - 1.0

                    # 2. Norm_Vol: V_i / (Mean(Window_Vol) + 1e-9)
                    mean_vol = np.mean(volumes)
                    if mean_vol == 0: mean_vol = 1e-9
                    norm_vols = volumes / mean_vol

                    # 3. Flow Ratio (Raw)

                    # 4. MACD Hist (Normalized by Close price)
                    norm_macd = macd_hists / prices

                    # 5. KDJ_J (Normalized 0-1 range approx)
                    norm_kdj = kdj_js / 100.0

                    # Stack Channels
                    # Shape: (Time_Steps=6, Channels=5)
                    sample = np.stack([
                        norm_prices,
                        norm_vols,
                        flows,
                        norm_macd,
                        norm_kdj
                    ], axis=1) # (6, 5)

                    # Add to dataset
                    all_X.append(sample)
                    all_y.append(label)

                    # Metadata for validation
                    # Note: storing 'original_index' is crucial for verification
                    metadata.append({
                        'code': code,
                        'date': df.loc[i, 'Date'],
                        'original_index': i
                    })

                    # Apply NMS if positive label
                    if label == 1:
                        skip_until = i + 3

            except Exception as e:
                print(f"Error processing {code}: {e}")

        # Convert to numpy arrays
        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int32)

        # Save
        save_path = os.path.join(PROCESSED_DIR, "train_data.npz")
        np.savez(save_path, X=X, y=y, metadata=metadata)
        print(f"Saved processed data to {save_path}")
        print(f"X Shape: {X.shape}, y Shape: {y.shape}")

if __name__ == "__main__":
    fe = FeatureEngine()
    fe.process_all()
