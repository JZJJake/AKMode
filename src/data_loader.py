import requests
import pandas as pd
import logging
from urllib.parse import urljoin
from config.settings import TDX_API_BASE_URL, TDX_API_HISTORY_ENDPOINT

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data fetching from the TDX API.
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
        """
        all_ticks = []
        start = 0
        count = 2000 # Fetch limit per request (adjust based on API limits)

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

                if data.get("code") != 0:
                    logger.error(f"API Error for {code} on {date}: {data.get('msg', 'Unknown Error')}")
                    break

                tick_data = data.get("data", {})
                ticks_list = tick_data.get("List", [])
                total_count = tick_data.get("Count", 0)

                if not ticks_list:
                    break

                all_ticks.extend(ticks_list)

                # If received fewer records than requested, we've likely reached the end.
                if len(ticks_list) < count:
                    break

                start += len(ticks_list)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {code} on {date}: {e}")
                # Simple retry logic could be added here, but for now we break or return partial data
                # Depending on requirement, we might want to fail hard or continue.
                # Given strictness, let's log and return empty to avoid partial data corruption.
                return pd.DataFrame()

        if not all_ticks:
            return pd.DataFrame()

        df = pd.DataFrame(all_ticks)
        return df

if __name__ == "__main__":
    # Simple test
    loader = DataLoader()
    # Note: This will fail without a running server, but verifies syntax.
    try:
        df = loader.fetch_history_ticks("000001", "20231101")
        print(df.head())
    except Exception as e:
        print(f"Test run failed: {e}")
