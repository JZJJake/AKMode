from enum import Enum

class Exchange(Enum):
    SSE = "SSE"
    SZSE = "SZSE"
    # Add others as needed

class Interval(Enum):
    DAILY = "1d"
    WEEKLY = "1w"
    TICK = "tick"
