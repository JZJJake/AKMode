import os
import pandas as pd
import numpy as np
import logging
from config.settings import RAW_TICKS_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)

def calculate_flow_metrics(tick_df: pd.DataFrame) -> dict:
    """
    Calculates the 'True Flow' metrics from tick data.
    Ref: https://github.com/oficcejo/tdx-api

    Logic:
    - True_Active_Buy (Status=0)
    - True_Active_Sell (Status=1)
    - Flow_Ratio = (Buy - Sell) / Total_Volume

    Note: tick_df must have columns ['Price', 'Volume', 'Status'] in proper units.
    """
    if tick_df.empty:
        return {}

    # Ensure numeric types (already handled in loader but safe to re-check)
    tick_df['Volume'] = pd.to_numeric(tick_df['Volume'], errors='coerce').fillna(0)
    tick_df['Status'] = pd.to_numeric(tick_df['Status'], errors='coerce').fillna(-1)

    # Calculate components
    # Status: 0=Buy, 1=Sell, 2=Neutral
    true_buy = tick_df[tick_df['Status'] == 0]['Volume'].sum()
    true_sell = tick_df[tick_df['Status'] == 1]['Volume'].sum()
    total_vol = tick_df['Volume'].sum()

    flow_ratio = 0.0
    if total_vol > 0:
        flow_ratio = (true_buy - true_sell) / total_vol

    return {
        'True_Active_Buy': true_buy,
        'True_Active_Sell': true_sell,
        'Total_Volume': total_vol,
        'True_Flow_Ratio': flow_ratio
    }

def save_raw_ticks(tick_df: pd.DataFrame, code: str, date: str):
    """
    Saves raw ticks to data/raw_ticks/{date}/{code}.parquet
    """
    if tick_df.empty:
        return

    # Create date directory
    date_dir = os.path.join(RAW_TICKS_DIR, date)
    os.makedirs(date_dir, exist_ok=True)

    file_path = os.path.join(date_dir, f"{code}.parquet")

    # Add Code column for context if read later
    df_to_save = tick_df.copy()
    df_to_save['Code'] = code

    try:
        df_to_save.to_parquet(file_path, engine='pyarrow', index=False)
    except Exception as e:
        logger.error(f"Failed to save raw ticks for {code} on {date}: {e}")

def update_features(feature_data: dict, code: str, date: str):
    """
    Updates the feature file for a specific stock: data/features/{code}.parquet
    """
    if not feature_data:
        return

    file_path = os.path.join(FEATURES_DIR, f"{code}.parquet")

    # Create new record
    new_record = feature_data.copy()
    new_record['Date'] = date
    new_record['Code'] = code
    new_df = pd.DataFrame([new_record])

    # Load existing or create new
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_parquet(file_path)
            # Check if date already exists to avoid duplicates
            if date in existing_df['Date'].values.astype(str):
                # Optionally update or skip. For now, let's skip/warn.
                # logger.warning(f"Data for {code} on {date} already exists in features.")
                # We might want to overwrite. Let's filter out old and append new.
                existing_df = existing_df[existing_df['Date'] != date]

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Sort by date
            combined_df = combined_df.sort_values('Date')
        except Exception as e:
            logger.error(f"Failed to read existing features for {code}: {e}")
            combined_df = new_df
    else:
        combined_df = new_df

    try:
        combined_df.to_parquet(file_path, engine='pyarrow', index=False)
    except Exception as e:
        logger.error(f"Failed to save features for {code}: {e}")

def process_daily_stock(tick_df: pd.DataFrame, code: str, date: str):
    """
    Orchestrates the ETL for a single stock.
    """
    # 1. Save Raw Ticks
    save_raw_ticks(tick_df, code, date)

    # 2. Calculate Features
    metrics = calculate_flow_metrics(tick_df)

    # 3. Update Feature Store
    update_features(metrics, code, date)
