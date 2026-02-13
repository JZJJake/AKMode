import os
import csv
import logging
from config.settings import DATA_DIR, STOCK_FILTER_REGEX, is_valid_stock_code

logger = logging.getLogger(__name__)

def get_stock_list_file():
    return os.path.join(DATA_DIR, "stock_list.csv")

def load_stock_list():
    """
    Loads the stock list from data/stock_list.csv.
    Returns a list of stock codes.
    """
    file_path = get_stock_list_file()
    if not os.path.exists(file_path):
        logger.warning(f"Stock list file not found at {file_path}. Returning empty list.")
        return []

    stocks = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    code = row[0].strip()
                    if is_valid_stock_code(code):
                        stocks.append(code)
    except Exception as e:
        logger.error(f"Failed to read stock list: {e}")

    return stocks

def save_stock_list(codes):
    """
    Saves a list of codes to data/stock_list.csv
    """
    file_path = get_stock_list_file()
    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for code in codes:
                writer.writerow([code])
    except Exception as e:
        logger.error(f"Failed to save stock list: {e}")
