"""A collection of tools dealing with datetime data."""
import copy
import os
import re

import numpy as np
import pandas as pd


__all__ = ['timestamp_parsers', 'extract_date_substr', 'parse_day_number',
           'parse_unix_time']


# A dictionary of parser functions for timestamps stored in multiple columns.
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
    'ymdhmsf': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S %f'),
}
# NOTE: this module variable cannot be documented by sphinx (1.6.3); check
# later versions


def extract_date_substr(flist, date_format='%Y%m%d'):
    """
    Extract date substring from a list of file paths and parse to timestamps.

    Parameters
    ----------
    flist : list
        A list of file paths.
    date_format : str, optional
        Date format for the substring in file names. Supported date formats
        include: ``'%Y%m%d'``, ``'%Y_%m_%d'``, ``'%Y-%m-%d'``, or similar
        formats with a two-digit year number (replacing ``'%Y'`` with
        ``'%y'``). However, it is not guaranteed that a format other than
        these would work.

    Returns
    -------
    date_substr_list : list
        A list of date substrings extracted from file names in ``flist``.
    ts_series : pandas.DatetimeIndex
        A series of pandas timestamps converted from ``date_substr_list``.
    """
    # replace date_format string to get regex pattern
    re_replace_dict = {
        '-': '\\-',        # dash separators
        '%Y': '[0-9]{4}',  # 4-digit year
        '%y': '[0-9]{2}',  # 2-digit
        '%m': '[0-9]{2}',
        '%d': '[0-9]{2}',
    }
    # get regex date pattern
    re_date_pattern = copy.copy(date_format)
    for k in list(re_replace_dict.keys()):
        re_date_pattern = re_date_pattern.replace(k, re_replace_dict.pop(k))

    date_substr_list = [re.search(re_date_pattern, os.path.basename(f)).group()
                        for f in flist]
    ts_series = pd.to_datetime(date_substr_list, format=date_format,
                               errors='coerce')
    return date_substr_list, ts_series


def parse_day_number(doy, year):
    """
    Parse day numbers and convert to pandas.Timestamp.

    Parameters
    ----------
    doy : float or array_like
        Zero-based day of year numbers.
    year : int or str or array_like
        Four-digit year numbers.

    Returns
    -------
    ts : pandas.Timestamp or pandas.Series
        Converted timestamp or series of timestamps. If both ``doy`` and
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
    ts = year_start + pd.to_timedelta(doy, unit='D', errors='coerce')
    if isinstance(ts, pd.DatetimeIndex):
        # for array inputs, ensure that a series of Timestamp is returned
        ts = pd.Series(ts.values)
    return ts


def parse_unix_time(sec, epoch_year=None):
    """
    Parse Unix time in seconds (POSIX) and convert to pandas.Timestamp.

    Parameters
    ----------
    sec : float or array_like
        Unix seconds.
    epoch_year : int or str
        The epoch year that is referenced to. Default is 1970 for the POSIX
        standard. However, beware that many softwares may have used different
        epoch years in their time systems.

    Returns
    -------
    ts : pandas.Timestamp or pandas.Series
        Converted timestamp or series of timestamps. If ``sec`` is an array,
        the returned value is an instance of ``pandas.Series``.

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
    ts = pd.Timestamp(str(epoch_year)) + \
        pd.to_timedelta(sec, unit='s', errors='coerce')
    if isinstance(ts, pd.DatetimeIndex):
        # for array inputs, ensure that a series of Timestamp is returned
        ts = pd.Series(ts.values)
    return ts
