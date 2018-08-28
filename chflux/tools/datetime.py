import copy
import os
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

__all__ = ['timestamp_parsers', 'extract_date_str', 'parse_day_number',
           'parse_unix_time', 'timedelta_days']


# A dictionary of parser functions for timestamps stored in multiple columns.
# Supports only the ISO 8601 format (year-month-day).
# Does not support month-first (American) or day-first (European) format.
timestamp_parsers = {
    # parse date-only string
    'ymd': lambda s: pd.to_datetime(s, format='%Y %m %d'),
    # parse datetime string that is accurate to minutes
    'ymdhm': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M'),
    # parse datetime string that is accurate to seconds
    'ymdhms': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S'),
    # parse datetime string that is accurate to nanoseconds
    'ymdhmsf': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S %f'),
}


def extract_date_str(paths: List[str],
                     date_format: str = '%Y%m%d') -> Tuple[List[str],
                                                           pd.DatetimeIndex]:
    """
    Extract date substrings from a list of file paths and parse to timestamps.

    Parameters
    ----------
    paths : list
        A list of file paths.
    date_format : str, optional
        Date format for the substring in file names. Supported date formats
        include: ``'%Y%m%d'``, ``'%Y_%m_%d'``, ``'%Y-%m-%d'``, or similar
        formats with a two-digit year number (replacing ``'%Y'`` with
        ``'%y'``). However, it is not guaranteed that a format other than
        these would work.

    Returns
    -------
    date_strs : list
        A list of date substrings extracted from file names in ``paths``.
    ts_series : pandas.DatetimeIndex
        A series of pandas timestamps converted from ``date_strs``.
    """
    # replace date_format string to make a regex pattern
    re_replace_dict = {
        '-': '\\-',        # dash separators
        '%Y': '[0-9]{4}',  # 4-digit year
        '%y': '[0-9]{2}',  # 2-digit
        '%m': '[0-9]{2}',
        '%d': '[0-9]{2}',
    }
    re_date_pattern = copy.copy(date_format)
    for k in list(re_replace_dict.keys()):
        re_date_pattern = re_date_pattern.replace(k, re_replace_dict.pop(k))
    # match regex
    re_match_list = [re.search(re_date_pattern, os.path.basename(f))
                     for f in paths]
    # note: re.search(...) -> Optional[Match[str]]; must remove None
    # get substring list
    date_strs = [m.group() for m in re_match_list if m is not None]
    ts_series = pd.to_datetime(date_strs, format=date_format,
                               errors='coerce')
    return date_strs, ts_series


def parse_day_number(day, year):
    """
    Parse day-of-year numbers and convert to ``pandas.Timestamp``.

    Parameters
    ----------
    day : float or array_like
        Zero-based day-of-year number(s).
    year : int or str or array_like
        Four-digit year number(s) to specify the reference year(s).

    Returns
    -------
    ts : pandas.Timestamp or pandas.Series
        Converted timestamp or series of timestamps. If both ``day`` and
        ``year`` are scalars, the returned value is an instance of
        ``pandas.Timestamp``.

    Examples
    --------
    >>> parse_day_number(180., 2017)
    Timestamp('2017-06-30 00:00:00')

    >>> parse_day_number(180., [2017, 2018, 2019])
    0   2017-06-30
    1   2018-06-30
    2   2019-06-30
    dtype: datetime64[ns]

    >>> parse_day_number([30., 60., 90.], 2017)
    0   2017-01-31
    1   2017-03-02
    2   2017-04-01
    dtype: datetime64[ns]

    >>> parse_day_number([30., 60., 90.], ['2017', '2018', '2019'])
    0   2017-01-31
    1   2018-03-02
    2   2019-04-01
    dtype: datetime64[ns]
    """
    if np.isscalar(year):
        year_start = pd.Timestamp(str(year))
    else:
        year_start = pd.to_datetime({'year': year, 'month': 1, 'day': 1})
    ts = year_start + pd.to_timedelta(day, unit='D', errors='coerce')
    if isinstance(ts, pd.DatetimeIndex):
        ts = pd.Series(ts.values)  # for array input
    return ts


def parse_unix_time(seconds, epoch_year: Optional[Union[int, str]] = None):
    """
    Parse Unix time in seconds and convert to ``pandas.Timestamp``.

    Parameters
    ----------
    seconds : float or array_like
        Unix seconds.
    epoch_year : int or str
        The epoch year to reference to. Default is 1970 (POSIX standard).
        However, other applications or operating systems may use a different
        epoch year as the reference.

    Returns
    -------
    ts : pandas.Timestamp or pandas.Series
        Converted timestamp or series of timestamps. If both ``seconds`` and
        ``year`` are scalars, the returned value is an instance of
        ``pandas.Timestamp``.

    Examples
    --------
    >>> parse_unix_time(3474921600., 1904)  # 1904 for LabVIEW
    Timestamp('2014-02-11 00:00:00')

    >>> parse_unix_time(2464456901. + np.arange(5.))
    0   2048-02-04 19:21:41
    1   2048-02-04 19:21:42
    2   2048-02-04 19:21:43
    3   2048-02-04 19:21:44
    4   2048-02-04 19:21:45
    dtype: datetime64[ns]
    """
    if epoch_year is None:
        epoch_year = '1970'
    ts = pd.Timestamp(str(epoch_year)) + pd.to_timedelta(
        seconds, unit='s', errors='coerce')
    if isinstance(ts, pd.DatetimeIndex):
        ts = pd.Series(ts.values)  # for array input
    return ts


def timedelta_days(ts: Union[pd.Timestamp, pd.Series, pd.DatetimeIndex],
                   ref: pd.Timestamp):
    """Calculate timedelta [in days] with respect to a reference timestamp."""
    return (ts - ref) / pd.Timedelta(1, 'D')
