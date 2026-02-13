import requests
import pandas as pd
import logging
from urllib.parse import urljoin
from app_config.settings import TDX_API_BASE_URL, TDX_API_HISTORY_ENDPOINT

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data fetching from the TDX API.
    Ref: https://github.com/oficcejo/tdx-api
    """

    def __init__(self, base_url: str = TDX_API_BASE_URL, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_history_ticks(self, code: str, date: str) -> pd.DataFrame:
        """
        Fetches all trade ticks for a given stock on a specific date.

        Args:
            code (str): Stock code (e.g., "000001").
            date (str): Date string (e.g., "20241101").

        Returns:
            pd.DataFrame: DataFrame containing all ticks (Time, Price, Volume, Status).
                          Returns an empty DataFrame on failure or no data.
                          Columns: ['Time', 'Price', 'Volume', 'Status']
                          Price in Yuan (converted from 厘), Volume in Shares (converted from Hands).
        """
        all_ticks = []
        start = 0
        count = 2000 # Fetch limit per request

        while True:
            params = {
                "code": code,
                "date": date,
                "start": start,
                "count": count
            }

            try:
                url = urljoin(self.base_url, TDX_API_HISTORY_ENDPOINT)
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                # Check API success code (assumed 0 based on context, possibly 'code' field)
                if data.get("code") != 0:
                    logger.error(f"API Error for {code} on {date}: {data.get('msg', 'Unknown Error')}")
                    break

                # Extract data based on described structure
                tick_data = data.get("data", {})
                ticks_list = tick_data.get("List", [])

                if not ticks_list:
                    break

                all_ticks.extend(ticks_list)

                if len(ticks_list) < count:
                    break

                start += len(ticks_list)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {code} on {date}: {e}")
                return pd.DataFrame()

        if not all_ticks:
            return pd.DataFrame()

        df = pd.DataFrame(all_ticks)

        # Verify columns exist (PascalCase check)
        required_cols = ['Time', 'Price', 'Volume', 'Status']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in API response for {code}. Found: {df.columns}")
            # Attempt case-insensitive mapping if needed, or fail.
            # For now, strict check as requested.
            return pd.DataFrame()

        # Type Conversion
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        df['Status'] = pd.to_numeric(df['Status'], errors='coerce').fillna(2) # Default to neutral if error

        # Unit Conversion
        # Price: 厘 -> Yuan (/1000)
        df['Price'] = df['Price'] / 1000.0

        # Volume: Hands -> Shares (*100)
        df['Volume'] = df['Volume'] * 100

        return df[['Time', 'Price', 'Volume', 'Status']]

if __name__ == "__main__":
    # Simple test
    loader = DataLoader()
    try:
        df = loader.fetch_history_ticks("000001", "20231101")
        print(df.head())
    except Exception as e:
        print(f"Test run failed: {e}")
