import pandas as pd
import numpy as np
import torch
import joblib
import os
import glob
from datetime import datetime, timedelta
import argparse
import logging

from app_config.settings import MODELS_DIR, FEATURES_DIR, DATA_DIR
from src.model_defs import TimeSeriesNet
from src.technical_indicators import calculate_macd, calculate_kdj_j
from src.visualizer import plot_prediction

REPORTS_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

class InferenceEngine:
    """
    Daily Inference Pipeline.
    Loads trained models and runs prediction on a specific date using data from features/ directory.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Models
        self.nn_path = os.path.join(MODELS_DIR, "stage1_nn.pth")
        self.lgb_path = os.path.join(MODELS_DIR, "stage2_lgb.pkl")

        if not os.path.exists(self.nn_path) or not os.path.exists(self.lgb_path):
            raise FileNotFoundError("Models not found. Train first.")

        self.nn_model = TimeSeriesNet(input_size=5).to(self.device)
        self.nn_model.load_state_dict(torch.load(self.nn_path, map_location=self.device))
        self.nn_model.eval()

        self.lgb_model = joblib.load(self.lgb_path)

    def run_daily_inference(self, target_date_str: str):
        """
        Runs inference for all stocks on the target date.
        """
        print(f"Running Inference for {target_date_str}")

        feature_files = glob.glob(os.path.join(FEATURES_DIR, "*.parquet"))
        results = []

        for file_path in feature_files:
            code = os.path.basename(file_path).replace(".parquet", "")
            try:
                df = pd.read_parquet(file_path)
                # Ensure date is sorted datetime
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)

                # Check if target_date exists
                target_dt = pd.to_datetime(target_date_str)
                mask = df['Date'] == target_dt
                if not mask.any():
                    # No data for target date
                    continue

                target_idx = df.index[mask][0]

                # Need previous 5 days (total window size 6: T-5 to T)
                if target_idx < 5:
                    # Not enough history
                    continue

                # Calculate Technicals on Full DF (or window + lookback)
                # For simplicity, calculate on full, dropna handles start
                # Ideally, we should optimize this for daily runs, but full recalc is fast enough for <5000 stocks locally.

                # Ensure float
                df['Volume'] = df['Volume'].astype(float)

                # Technicals
                # MACD needs 26+9 days approx, KDJ needs 9 days.
                # If target_idx < 30, might be NaN.

                df['MACD_Hist'] = calculate_macd(df)
                df['KDJ_J'] = calculate_kdj_j(df)

                # Check if NaN at target_idx
                if pd.isna(df.loc[target_idx, 'MACD_Hist']) or pd.isna(df.loc[target_idx, 'KDJ_J']):
                    continue

                # Extract Window [target_idx - 5 : target_idx + 1]
                window_df = df.iloc[target_idx - 5 : target_idx + 1].copy()

                if len(window_df) != 6:
                    continue

                # Extract raw values
                prices = window_df['Close'].values
                volumes = window_df['Volume'].values
                flows = window_df['True_Flow_Ratio'].values
                macd_hists = window_df['MACD_Hist'].values
                kdj_js = window_df['KDJ_J'].values

                # Normalization
                p0 = prices[0]
                if p0 == 0: continue
                norm_prices = (prices / p0) - 1.0

                mean_vol = np.mean(volumes)
                if mean_vol == 0: mean_vol = 1e-9
                norm_vols = volumes / mean_vol

                norm_macd = macd_hists / prices
                norm_kdj = kdj_js / 100.0

                # Stack
                sample = np.stack([
                    norm_prices,
                    norm_vols,
                    flows,
                    norm_macd,
                    norm_kdj
                ], axis=1) # (6, 5)

                # Prepare Tensor
                X_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, 6, 5)

                # Stage 1: Embedding
                with torch.no_grad():
                    embedding = self.nn_model.get_embedding(X_tensor).cpu().numpy() # (1, 32)

                # Stage 2: Tabular Features
                # Mock: Random (In production, fetch real MarketCap/Turnover)
                tabular_feat = np.random.randn(1, 2).astype(np.float32)

                # Combine
                X_final = np.hstack([embedding, tabular_feat])

                # Predict
                prob = self.lgb_model.predict_proba(X_final)[0, 1]

                # Filter
                if prob >= 0.60:
                    res_entry = {
                        'Code': code,
                        'Date': target_date_str,
                        'Price': df.loc[target_idx, 'Close'],
                        'Flow_Ratio': df.loc[target_idx, 'True_Flow_Ratio'],
                        'Probability': prob
                    }
                    results.append(res_entry)

                    # Visualize
                    try:
                        plot_prediction(code, target_date_str, df, prob)
                    except Exception as e:
                        print(f"Plotting failed for {code}: {e}")

            except Exception as e:
                # print(f"Error inferencing {code}: {e}")
                continue

        # Generate Report
        if results:
            res_df = pd.DataFrame(results).sort_values('Probability', ascending=False)

            # Print to Console
            print("\n--- Daily Predictions (Prob >= 0.60) ---")
            print(res_df.to_string(index=False))

            # Save
            save_path = os.path.join(REPORTS_DIR, f"pred_{target_date_str}.csv")
            res_df.to_csv(save_path, index=False)
            print(f"\nReport saved to {save_path}")
        else:
            print("\nNo stocks met the prediction criteria (Prob >= 0.60).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKMode Daily Inference")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="Target Date (YYYYMMDD)")
    args = parser.parse_args()

    engine = InferenceEngine()
    engine.run_daily_inference(args.date)
