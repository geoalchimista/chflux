"""
A collection of tools dealing with date and time for PyChamberFlux

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
import re
import os
import copy
import numpy as np
import pandas as pd


def extract_date_substr(flist, date_format='%Y%m%d'):
    """
    Extract date substring from a list of file paths and parse as timestamps.

    Parameters
    ----------
    flist : list
        A list of file paths, typically generated from `glob.glob()`.
    date_format : str, optional
        Date format for the substring in file names. Supported data formats
        include: '%Y%m%d', '%Y_%m_%d', '%Y-%m-%d', or similar formats with
        two-digit year number (replacing '%Y' with '%y').
        It is not guaranteed that a format other than these would work.

    Returns
    -------
    date_substr_list : list
        A list of date substring extracted from file names, in the same order
        as the file paths in the input `flist`.
    ts_series : pandas.DatetimeIndex
        A series of timestamps converted from `date_substr_list`.

    """
    re_date_format = copy.copy(date_format)  # regex date format
    re_date_format = re_date_format.replace('-', '\-')  # dash separators
    re_date_format = re_date_format.replace('%Y', '[0-9]{4}')  # 4-digit year
    re_date_format = re_date_format.replace('%y', '[0-9]{2}')  # 2-digit
    re_date_format = re_date_format.replace('%m', '[0-9]{2}')
    re_date_format = re_date_format.replace('%d', '[0-9]{2}')
    date_substr_list = [re.search(re_date_format, os.path.basename(f)).group()
                        for f in flist]

    ts_series = pd.to_datetime(date_substr_list, format=date_format,
                               errors='coerce')
    return date_substr_list, ts_series


def parse_datetime(ts_input, year=None):
    """
    Parse datetime and convert to pandas.Timestamp.

    Parameters
    ----------
    ts_input : float, str, or datetime.datetime, or array_like
        Input timestamp, parsed differently according to its type. If
        `ts_input` is float, parse it as the day of year number (must also
        supply the year number). If `ts_input` is string or datetime.datetime,
        use the default mechanism of `pandas.Timestamp` to parse. Support
        array like input of the aforementioned types.
    year : int, optional
        If the `ts_input` is of float type, treated as day of year value, must
        also supply the year number.

    Returns
    -------
    ts : pandas.Timestamp or pandas.DatetimeIndex
        Converted timestamp or array of timestamps.

    """
    if (issubclass(type(ts_input), float) or
            issubclass(type(ts_input), np.floating)):
        # parse as day of year number
        if year is None:
            raise ValueError('Missing `year` when parsing day of year input.')
        ts = pd.Timestamp('%d-01-01' % year) + \
            pd.to_timedelta(ts_input, unit='D', errors='coerce')
    elif issubclass(type(ts_input), np.ndarray):
        if issubclass(ts_input.dtype.type, np.floating):
            # parse as day of year number
            if year is None:
                raise ValueError(
                    'Missing `year` when parsing day of year input.')
            ts = pd.Timestamp('%d-01-01' % year) + \
                pd.to_timedelta(ts_input, unit='D', errors='coerce')
        else:
            # use pandas.to_datetime
            ts = pd.to_datetime(ts_input, errors='coerce')
    else:
        # use pandas.to_datetime
        ts = pd.to_datetime(ts_input, errors='coerce')

    return ts
