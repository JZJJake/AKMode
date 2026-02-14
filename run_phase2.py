from src.mock_generator import generate_mock_stock_data
from src.feature_engine import FeatureEngine
import numpy as np
import pandas as pd

def run_phase2():
    print("Step 1: Mock Data Generation")
    # Generate for 5 stocks
    codes = ["000001", "600000", "000002", "600001", "300059"]
    for code in codes:
        generate_mock_stock_data(code)

    print("\nStep 2: Feature Engineering & Tensor Construction")
    fe = FeatureEngine()
    fe.process_all()

    print("\nStep 3: Verification")
    data = np.load("data/processed/train_data.npz", allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = data['metadata']

    print(f"X Shape: {X.shape}") # Should be (N, 6, 5)
    print(f"y Shape: {y.shape}")
    print(f"Positive Labels: {np.sum(y)}")

    # Check NMS logic using metadata['original_index']
    # If y[k] == 1 and y[k+1] == 1 for same stock, then idx[k+1] - idx[k] >= 4

    # Convert metadata to DataFrame for easier grouping
    meta_df = pd.DataFrame(list(metadata))
    meta_df['label'] = y

    violation_count = 0

    for code, group in meta_df.groupby('code'):
        pos_samples = group[group['label'] == 1].sort_values('original_index')
        if len(pos_samples) > 1:
            indices = pos_samples['original_index'].values
            diffs = np.diff(indices)
            min_diff = np.min(diffs)

            if min_diff < 4:
                print(f"❌ NMS Violation for {code}: Min Diff = {min_diff}")
                violation_count += 1
            else:
                pass
                # print(f"✅ {code}: Min Diff = {min_diff}")

    if violation_count == 0:
        print("✅ NMS Logic Verified: No consecutive positive samples within 3 days (indices).")
    else:
        print(f"❌ NMS Logic Failed: {violation_count} stocks have violations.")

if __name__ == "__main__":
    run_phase2()
