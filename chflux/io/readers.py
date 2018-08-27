import copy
import glob
import json
import warnings
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

from chflux.tools import timestamp_parsers

__all__ = ['read_yaml', 'read_json', 'read_tabulated']


def read_yaml(path: str) -> Optional[Dict]:
    """Read a YAML file into a Python dict. If fails, return ``None``."""
    with open(path, 'r') as fp:
        try:
            ydict = yaml.load(fp)
        except yaml.YAMLError as exception_yaml:
            print(exception_yaml)
            ydict = None
    return ydict


def read_json(path: str) -> Optional[Dict]:
    """Read a JSON file into a Python dict. If fails, return ``None``."""
    with open(path, 'r') as fp:
        try:
            jdict = json.load(fp)
        except json.JSONDecodeError as exception_json:
            print(exception_json)
            jdict = None
    return jdict


def read_tabulated(name: str, config: Dict,
                   query: Optional[Union[str, List[str]]] = None) -> Optional[
                       pd.DataFrame]:
    """
    Read tabulated data files.

    Parameters
    ---------
    name : str
        Data type name. Allowed values are

            - ``'biomet'``: biometeorological data
            - ``'concentration'``: concentration data
            - ``'flow'``: flow rate data
            - ``'leaf'``: leaf area data
            - ``'timelag'``: timelag data
    config : dict
        Configuration dictionary parsed from the config file.
    query : str or list
        A query string or list of query strings used to filter the data file
        list. If ``None`` (default), no filtering is performed.

    Return
    ------
    df: pandas.DataFrame
        The loaded tabulated data.

    Raises
    -----
    RuntimeError
        If ``name`` is wrong.
    """
    data_names = ['biomet', 'concentration', 'flow', 'leaf', 'timelag']
    if name not in data_names:
        raise RuntimeError(
            f'Wrong data type name! Allowed name is one of {data_names}.')
    # extract relevant config for the data reader
    reader_config = {k.replace(f'{name}.', ''): v for k, v in config.items()
                     if f'{name}.' in k}
    # get the data file list
    files = sorted(glob.glob(reader_config['files']))
    # filter the list of data files with query string(s)
    if isinstance(query, str):
        files_filtered = sorted([f for f in files if query in f])
    elif isinstance(query, list) and all(isinstance(q, str) for q in query):
        files_filtered = sorted([f for f in files
                                 if any(q in f for q in query)])
    else:
        files_filtered = files  # no filtering if query is None or illegal
    # check data file existence
    if len(files_filtered) == 0:
        warnings.warn(f'Cannot find a valid {name} data file!', RuntimeWarning)
        return None
    else:
        print(f'Found {len(files_filtered)} {name} data file(s). Loading...')
    # read csv data
    date_parser = timestamp_parsers.get(reader_config['timestamp.parser'])
    read_csv_options = {
        'sep': reader_config['csv.delimiter'],
        'header': reader_config['csv.header'],
        'names': reader_config['csv.names'],
        'usecols': reader_config['csv.usecols'],
        'dtype': reader_config['csv.dtype'],
        'na_values': reader_config['csv.na_values'],
        'parse_dates': reader_config['csv.parse_dates'],
        'date_parser': date_parser,
        'infer_datetime_format': True,
        'engine': 'c',
        'encoding': 'utf-8',
    }
    df_loaded = [pd.read_csv(f, **read_csv_options) for f in files_filtered]
    for f in files_filtered:
        print(f)
    # return the concatenated dataframe
    try:
        df = pd.concat(df_loaded, ignore_index=True)
    except ValueError as pderr:
        warnings.warn('Cannot concatenate data frames!', RuntimeWarning)
        print(pderr)
        return None

    print(f'{df.shape[0]} lines read from {name} data.')
    return df
