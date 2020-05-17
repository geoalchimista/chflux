import copy
import glob
import json
import warnings
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

from chflux.tools import timestamp_parsers, extract_date_str

__all__ = ['read_yaml', 'read_json', 'read_tabulated',
           'make_filedict', 'read_data_by_date']


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
                   path: Optional[Union[str, List[str]]] = None,
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
    path : str or list of str, optional
        File path(s) to override the path setting in ``config``. No wildcard
        pattern matching is allowed if ``path`` is a list of strings.
    query : str or list of str, optional
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
    if isinstance(path, str):
        files = sorted(glob.glob(path))
    elif isinstance(path, list) and all(isinstance(p, str) for p in path):
        files = path.copy()
    else:
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


def make_filedict(config: Dict) -> Tuple[Dict[str, Dict[str, List[str]]],
                                         List[str]]:
    """
    Generate a nested dictionary with the mapping
    {_data name_ => {_date_ => [_file-1_, _file-2_, ...]}}.
    """
    filedict = {}
    dates: List[str] = []
    for name in ['biomet', 'concentration', 'flow', 'leaf', 'timelag']:
        if isinstance(config[f'{name}.files'], str):
            files = sorted(glob.glob(config[f'{name}.files']))
            # extract date strings and timestamps from file names
            _, ts_series = extract_date_str(
                files, config[f'{name}.date_format'])
            date_to_files: Dict[str, List[str]] = {}
            for f, ts in zip(files, ts_series):
                if not pd.isnull(ts):
                    ts_str = ts.strftime('%Y%m%d')
                    if ts_str not in dates:
                        dates.append(ts_str)
                    if ts_str in date_to_files:
                        date_to_files[ts_str].append(f)
                    else:
                        date_to_files[ts_str] = [f]
            filedict[name] = date_to_files
        else:
            filedict[name] = {}

    dates.sort()
    return filedict, dates


def read_data_by_date(config: Dict, filedict: Dict[str, Dict[str, List[str]]],
                      date: str):
    df_dict = {}
    for name in ['biomet', 'concentration', 'flow', 'leaf', 'timelag']:
        paths = filedict[name].get(date)
        if (isinstance(paths, list) and len(paths) > 0):
            df_dict[name] = read_tabulated(name, config, paths)
        else:
            df_dict[name] = None

    return df_dict
