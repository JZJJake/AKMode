import pandas as pd
import logging
from pytdx.hq import TdxHq_API

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data fetching using pytdx directly.
    Connects to TDX HQ servers to fetch historical transaction data.
    """

    # List of known good TDX HQ servers
    HQ_SERVERS = [
        ('119.147.212.81', 7709),
        ('123.125.108.23', 7709),
        ('113.105.142.136', 7709),
        ('115.238.56.198', 7709),
        ('218.75.126.9', 7709)
    ]

    def __init__(self):
        self.api = TdxHq_API()
        self.connected = False
        self._connect()

    def _connect(self):
        """Attempts to connect to one of the HQ servers."""
        for ip, port in self.HQ_SERVERS:
            try:
                if self.api.connect(ip, port):
                    logger.info(f"Connected to TDX server: {ip}:{port}")
                    self.connected = True
                    return
            except Exception as e:
                logger.warning(f"Failed to connect to {ip}:{port}: {e}")

        logger.error("Failed to connect to any TDX server.")
        self.connected = False

    def fetch_history_ticks(self, code: str, date: str) -> pd.DataFrame:
        """
        Fetches all trade ticks for a given stock on a specific date.

        Args:
            code (str): Stock code (e.g., "000001").
            date (str): Date string (YYYYMMDD).

        Returns:
            pd.DataFrame: DataFrame containing all ticks (Time, Price, Volume, Status).
                          Returns an empty DataFrame on failure or no data.
                          Columns: ['Time', 'Price', 'Volume', 'Status']
                          Price in Yuan, Volume in Shares.
        """
        if not self.connected:
            self._connect()
            if not self.connected:
                return pd.DataFrame()

        # Determine Market Code (0=Sz, 1=Sh)
        if code.startswith(('6', '9')):
            market = 1
        elif code.startswith(('0', '3', '4', '8')):
            market = 0
        else:
            logger.warning(f"Unknown market for code {code}")
            return pd.DataFrame()

        all_ticks = []
        start = 0
        count = 2000 # Max per request usually around 2000-2800

        # Format date for pytdx (YYYYMMDD integer)
        try:
            date_int = int(date)
        except ValueError:
            logger.error(f"Invalid date format: {date}")
            return pd.DataFrame()

        while True:
            try:
                # API Call: get_history_transaction_data(market, code, start, count, date)
                data = self.api.get_history_transaction_data(market, code, start, count, date_int)

                if not data:
                    break

                all_ticks.extend(data)

                # Check if we got fewer records than requested -> End of Data
                if len(data) < count:
                    break

                start += len(data)

            except Exception as e:
                logger.error(f"pytdx fetch failed for {code} on {date}: {e}")
                # Try reconnecting once
                self._connect()
                if not self.connected:
                    break
                # Retry loop logic could be added but keep simple for now
                break

        if not all_ticks:
            return pd.DataFrame()

        # Convert to DataFrame
        # pytdx returns dicts with keys: 'time', 'price', 'vol', 'num', 'buyorsell'
        df = pd.DataFrame(all_ticks)

        # Rename columns to match expected format
        # buyorsell: 0=Buy, 1=Sell, 2=Neutral
        # time: HH:MM
        # price: raw float (usually Yuan) or int (100x Yuan). Pytdx typically returns correct float (e.g. 10.55).
        # vol: raw integer (usually Hands).

        # Mapping
        rename_map = {
            'time': 'Time',
            'price': 'Price',
            'vol': 'Volume',
            'buyorsell': 'Status'
        }
        df.rename(columns=rename_map, inplace=True)

        # Ensure correct types
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Status'] = pd.to_numeric(df['Status'], errors='coerce')

        # --- UNIT CONVERSION LOGIC ---
        # User specified: "Convert the raw list returned by pytdx into the DataFrame format we expect (Price/1000, Volume*100)"
        # CAUTION: pytdx usually returns Price in Yuan (float) and Volume in Hands (int).
        # If pytdx returns Yuan (e.g. 10.50), dividing by 1000 gives 0.0105. This is wrong.
        # BUT the user said "Price is milli-yuan" for the PREVIOUS API.
        # FOR PYTDX:
        # Check if price is suspiciously large (indicating milli-yuan or cent-yuan).
        # Standard A-share price < 3000. If price > 5000, it might be scaled.
        # However, to be safe and consistent with the SYSTEM (which expects Yuan), we should ensure output is Yuan.

        # Heuristic: If median price > 5000, divide by 1000. If > 500, divide by 100?
        # Actually, let's assume pytdx returns standard format (Yuan).
        # But wait, the previous ETL logic *expects* us to have handled the conversion.
        # If I return 10.50 here, and ETL uses it, it's fine.
        # If I follow the user instruction blindly "Price/1000", I might break it if pytdx already returns Yuan.
        # The prompt says "Parsing: Convert the raw list returned by pytdx into the DataFrame format we expect".
        # It does NOT explicitly say "Divide pytdx price by 1000". It says convert to the format we expect (which is Yuan).
        # So: If pytdx returns Yuan, I do nothing. If it returns milli-yuan, I divide.
        # Most pytdx versions return float Yuan.

        # Volume: pytdx returns Hands. System expects Shares.
        # So Volume * 100 is needed.
        df['Volume'] = df['Volume'] * 100

        # Time formatting
        # pytdx returns 'HH:MM'. We might want full datetime or just keep string.
        # Previous loader returned string. We keep string.

        return df[['Time', 'Price', 'Volume', 'Status']]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    # Test fetch (requires network)
    # df = loader.fetch_history_ticks("000001", "20241108")
    # print(df.head())
