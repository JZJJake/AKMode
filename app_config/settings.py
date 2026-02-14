import os
import re

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_TICKS_DIR = os.path.join(DATA_DIR, "raw_ticks")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure data directories exist
os.makedirs(RAW_TICKS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# TDX API Configuration
TDX_API_BASE_URL = os.getenv("TDX_API_URL", "http://localhost:8080")
TDX_API_HISTORY_ENDPOINT = "/api/trade-history"

# Stock Filtering
STOCK_FILTER_REGEX = re.compile(r"^(000|002|003|300|600|601|603|605)\d{3}$")
EXCLUDED_REGEX = re.compile(r"^(688|8|4)\d{3}$") # Redundant but explicit for clarity
# Note: 000xxx includes Shenzhen Main Board. 002xxx is SME (now merged). 300xxx is ChiNext.
# 600xxx, 601xxx, 603xxx, 605xxx are Shanghai Main Board.

def is_valid_stock_code(code: str) -> bool:
    """
    Checks if the stock code is within the target scope:
    - Include: 00xxxx (Shenzhen), 300xxx (ChiNext), 60xxxx (Shanghai Main)
    - Exclude: 688xxx (STAR), 8xxxxx (Beijing), 4xxxxx (Other)
    """
    if not code:
        return False

    # Check exclusion first
    if EXCLUDED_REGEX.match(code):
        return False

    # Check inclusion
    return bool(STOCK_FILTER_REGEX.match(code))

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
