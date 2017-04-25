"""A collection of tools dealing with date and time"""

import re
import os
import copy
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
