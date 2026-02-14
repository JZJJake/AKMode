import pandas as pd
import logging
from src.etl import calculate_flow_metrics

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def verify_etl_logic():
    """
    Verification script to test the 'Unit Conversion' and 'Flow Ratio' logic
    using mocked raw API data.
    """
    logger.info("Starting ETL Verification Script...")

    # 1. Mock Data (Raw API Format)
    # Price: 12345 (12.345 Yuan)
    # Volume: 100 (100 Hands = 10000 Shares)
    # Status: 0 (Buy), 1 (Sell), 2 (Neutral)
    raw_data = [
        {"Time": "09:30:00", "Price": 12345, "Volume": 100, "Status": 0},  # Buy 100 Hands
        {"Time": "09:30:05", "Price": 12340, "Volume": 50,  "Status": 1},  # Sell 50 Hands
        {"Time": "09:30:10", "Price": 12345, "Volume": 50,  "Status": 2}   # Neutral 50 Hands
    ]

    logger.info("--- Step 1: Mock Raw Data ---")
    print(pd.DataFrame(raw_data))

    # 2. Simulate DataLoader Processing (Unit Conversion)
    logger.info("\n--- Step 2: Processing (Unit Conversion) ---")
    df = pd.DataFrame(raw_data)

    # Convert Price: Milli-Yuan -> Yuan
    df['Price'] = df['Price'] / 1000.0

    # Convert Volume: Hands -> Shares
    df['Volume'] = df['Volume'] * 100

    print("Processed DataFrame Head:")
    print(df[['Time', 'Price', 'Volume', 'Status']])

    # Verify Price
    expected_price = 12.345
    actual_price = df.iloc[0]['Price']
    if abs(actual_price - expected_price) < 1e-6:
        logger.info(f"✅ Price Verified: Expected {expected_price}, Got {actual_price}")
    else:
        logger.error(f"❌ Price Mismatch: Expected {expected_price}, Got {actual_price}")

    # Verify Volume
    expected_volume = 10000
    actual_volume = df.iloc[0]['Volume']
    if actual_volume == expected_volume:
        logger.info(f"✅ Volume Verified: Expected {expected_volume}, Got {actual_volume}")
    else:
        logger.error(f"❌ Volume Mismatch: Expected {expected_volume}, Got {actual_volume}")

    # 3. Verify Flow Ratio Calculation
    logger.info("\n--- Step 3: Flow Ratio Calculation ---")
    metrics = calculate_flow_metrics(df)

    # True Buy: 100 Hands * 100 = 10000
    # True Sell: 50 Hands * 100 = 5000
    # Neutral: 50 Hands * 100 = 5000
    # Total Volume: 20000
    # Flow Ratio: (10000 - 5000) / 20000 = 0.25

    print("Calculated Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    expected_ratio = 0.25
    actual_ratio = metrics['True_Flow_Ratio']

    if abs(actual_ratio - expected_ratio) < 1e-6:
        logger.info(f"✅ Flow Ratio Verified: Expected {expected_ratio}, Got {actual_ratio}")
    else:
        logger.error(f"❌ Flow Ratio Mismatch: Expected {expected_ratio}, Got {actual_ratio}")

if __name__ == "__main__":
    verify_etl_logic()
