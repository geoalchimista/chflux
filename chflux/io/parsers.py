"""PyChamberFlux I/O module containing a collection of data parsers."""
import pandas as pd


# A collection of parsers for timestamps stored in multiple columns.
# Supports only the ISO 8601 format (year-month-day).
# Does not support month-first (American) or day-first (European) format.
timestamp_parsers = {
    # date only
    'ymd': lambda s: pd.to_datetime(s, format='%Y %m %d'),

    # down to minute
    'ymdhm': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M'),

    # down to second
    'ymdhms': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S'),

    # down to nanosecond
    'ymdhmsf': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S %f')
}
