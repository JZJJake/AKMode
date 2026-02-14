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
        all_results = []

        for file_path in feature_files:
            code = os.path.basename(file_path).replace(".parquet", "")
            try:
                df = pd.read_parquet(file_path)

                # Handle Volume name mismatch (consistency with FeatureEngine)
                if 'Total_Volume' in df.columns and 'Volume' not in df.columns:
                    df.rename(columns={'Total_Volume': 'Volume'}, inplace=True)

                # Ensure date is sorted datetime
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)

                # Check if target_date exists
                target_dt = pd.to_datetime(target_date_str)
                mask = df['Date'] == target_dt
                if not mask.any():
                    continue

                target_idx = df.index[mask][0]

                # Need previous 5 days (total window size 6: T-5 to T)
                if target_idx < 5:
                    continue

                # Ensure float
                df['Volume'] = df['Volume'].astype(float)

                # Technicals
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
                tabular_feat = np.random.randn(1, 2).astype(np.float32)

                # Combine
                X_final = np.hstack([embedding, tabular_feat])

                # Predict
                prob = self.lgb_model.predict_proba(X_final)[0, 1]

                # Collect Result regardless of threshold
                res_entry = {
                    'Code': code,
                    'Date': target_date_str,
                    'Price': df.loc[target_idx, 'Close'],
                    'Flow_Ratio': df.loc[target_idx, 'True_Flow_Ratio'],
                    'Probability': prob
                }
                all_results.append(res_entry)

            except Exception as e:
                # print(f"Error inferencing {code}: {e}")
                continue

        # Process Results
        if all_results:
            res_df = pd.DataFrame(all_results).sort_values('Probability', ascending=False)

            # Print Top 5 Candidates Always
            print("\n--- Top 5 Candidates (Raw Output) ---")
            print(res_df.head(5).to_string(index=False))

            # Filter for Saving (Threshold >= 0.60)
            high_prob_df = res_df[res_df['Probability'] >= 0.60]

            if not high_prob_df.empty:
                save_path = os.path.join(REPORTS_DIR, f"pred_{target_date_str}.csv")
                high_prob_df.to_csv(save_path, index=False)
                print(f"\nSaved {len(high_prob_df)} predictions to {save_path}")

                # Visualize Top Predictions
                for _, row in high_prob_df.head(5).iterrows():
                    code = row['Code']
                    prob = row['Probability']
                    # Need df to plot, reload efficiently?
                    # For now just reload (slow but robust) or we could have cached it.
                    # Given constraints, reload.
                    try:
                        feature_path = os.path.join(FEATURES_DIR, f"{code}.parquet")
                        df = pd.read_parquet(feature_path)
                        # Ensure date column matches visualizer expectation
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        plot_prediction(code, target_date_str, df, prob)
                    except Exception as e:
                        print(f"Failed to plot {code}: {e}")
            else:
                print("\nNo stocks met the threshold (>= 0.60) for saving.")
        else:
            print("\nNo valid stocks found for inference date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKMode Daily Inference")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="Target Date (YYYYMMDD)")
    args = parser.parse_args()

    engine = InferenceEngine()
    engine.run_daily_inference(args.date)
