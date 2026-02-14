import pandas as pd
import argparse
import logging
import time
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.etl import process_daily_stock
from src.utils import load_stock_list, save_stock_list

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_history(start_date, end_date):
    """
    Fetches historical data for all stocks in the list between start_date and end_date.
    Respects business days and adds a delay between requests.
    """
    loader = DataLoader()
    stock_list = load_stock_list()

    if not stock_list:
        logger.warning("Stock list is empty. Using default list.")
        stock_list = ["000001", "600000", "000002", "600001", "300059"]
        save_stock_list(stock_list)

    logger.info(f"Fetching history for {len(stock_list)} stocks from {start_date} to {end_date}")

    # Generate Business Days
    dates = pd.bdate_range(start=start_date, end=end_date)

    for date in dates:
        date_str = date.strftime("%Y%m%d")
        logger.info(f"Processing date: {date_str}")

        success_count = 0
        fail_count = 0

        for code in stock_list:
            try:
                # Fetch Tick Data
                tick_df = loader.fetch_history_ticks(code, date_str)

                if tick_df.empty:
                    # logger.debug(f"No data for {code} on {date_str}")
                    fail_count += 1
                    continue

                # Process & Save (Raw Ticks + Features)
                process_daily_stock(tick_df, code, date_str)
                success_count += 1

                # Tiny delay to avoid overwhelming the server
                # time.sleep(0.05)

            except Exception as e:
                logger.error(f"Failed to process {code} on {date_str}: {e}")
                fail_count += 1

        logger.info(f"Date {date_str} Summary: Success={success_count}, Failed/NoData={fail_count}")

        # Inter-day delay if needed
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Historical Data for AKMode")
    parser.add_argument("--start", type=str, required=True, help="Start Date (YYYYMMDD)")
    parser.add_argument("--end", type=str, required=True, help="End Date (YYYYMMDD)")

    args = parser.parse_args()

    fetch_history(args.start, args.end)
