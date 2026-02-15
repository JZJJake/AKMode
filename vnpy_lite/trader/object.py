from dataclasses import dataclass
from datetime import datetime
from .constant import Exchange, Interval

@dataclass
class BarData:
    symbol: str
    exchange: Exchange
    datetime: datetime
    interval: Interval = Interval.DAILY

    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0
    volume: float = 0
    turnover: float = 0

    # Extended fields
    out_volume: float = 0
    in_volume: float = 0
    market_cap: float = 0  # In 100 Million (Yi)
    net_profit: float = 0

    def __post_init__(self):
        # Basic validation or type conversion if needed
        pass
