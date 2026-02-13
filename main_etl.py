import argparse
import logging
import datetime
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from config.settings import LOG_FORMAT, LOG_LEVEL
from src.data_loader import DataLoader
from src.etl import process_daily_stock
from src.utils import load_stock_list, save_stock_list, get_stock_list_file

# Configure Logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def worker_task(code: str, date: str):
    """
    Task to run in a separate process.
    """
    try:
        loader = DataLoader() # New instance per process for thread/process safety if needed
        logger.info(f"Fetching data for {code} on {date}...")

        tick_df = loader.fetch_history_ticks(code, date)

        if tick_df.empty:
            logger.warning(f"No data for {code} on {date}. Skipping.")
            return f"{code}: No Data"

        process_daily_stock(tick_df, code, date)
        logger.info(f"Successfully processed {code} on {date}.")
        return f"{code}: Success"

    except Exception as e:
        logger.error(f"Error processing {code}: {e}")
        return f"{code}: Failed - {e}"

def main():
    parser = argparse.ArgumentParser(description="AKMode Daily ETL: Fetch Ticks & Process Features")
    parser.add_argument("--date", type=str, help="Date to process (YYYYMMDD). Defaults to today.", default=None)
    parser.add_argument("--workers", type=int, help="Number of concurrent workers.", default=os.cpu_count() or 1)

    args = parser.parse_args()

    target_date = args.date
    if not target_date:
        target_date = datetime.datetime.now().strftime("%Y%m%d")

    logger.info(f"Starting ETL for date: {target_date}")

    # Check for stock list
    stock_list = load_stock_list()
    if not stock_list:
        logger.warning("Stock list is empty or file missing.")
        # For demonstration, create a default list if missing
        default_stocks = ["000001", "600000", "300059"]
        save_stock_list(default_stocks)
        stock_list = default_stocks
        logger.info(f"Created default stock list with: {stock_list}")

    logger.info(f"Found {len(stock_list)} stocks to process.")

    start_time = time.time()

    # Use ProcessPoolExecutor for CPU-bound tasks (pandas processing)
    # Even though fetching is I/O bound, we are doing heavy processing afterwards.
    # Also, Python requests might block the GIL less, but pandas definitely holds it often.
    # Processes are safer for isolation.

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, code, target_date): code for code in stock_list}

        for future in as_completed(futures):
            code = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                logger.error(f"Worker for {code} crashed: {e}")

    duration = time.time() - start_time
    logger.info(f"ETL Completed in {duration:.2f} seconds.")

    # Summarize
    success_count = sum(1 for r in results if "Success" in r)
    no_data_count = sum(1 for r in results if "No Data" in r)
    fail_count = len(results) - success_count - no_data_count

    logger.info(f"Summary: Success={success_count}, NoData={no_data_count}, Failed={fail_count}")

if __name__ == "__main__":
    main()
