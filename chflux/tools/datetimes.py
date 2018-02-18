"""A collection of tools dealing with datetime data."""
import copy
import os
import re

import pandas as _pd


__all__ = ['timestamp_parsers', 'extract_date_substr']


# A dictionary of parser functions for timestamps stored in multiple columns.
# Supports only the ISO 8601 format (year-month-day).
# Does not support month-first (American) or day-first (European) format.
timestamp_parsers = {
    # date only
    'ymd': lambda s: _pd.to_datetime(s, format='%Y %m %d'),
    # down to minute
    'ymdhm': lambda s: _pd.to_datetime(s, format='%Y %m %d %H %M'),
    # down to second
    'ymdhms': lambda s: _pd.to_datetime(s, format='%Y %m %d %H %M %S'),
    # down to nanosecond
    'ymdhmsf': lambda s: _pd.to_datetime(s, format='%Y %m %d %H %M %S %f'),
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
    ts_series = _pd.to_datetime(date_substr_list, format=date_format,
                                errors='coerce')
    return date_substr_list, ts_series
