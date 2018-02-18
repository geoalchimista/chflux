"""PyChamberFlux I/O module for reading config and data files."""
import collections
import copy
import glob

import yaml
import pandas as pd

# from chflux.io.parsers import timestamp_parsers
from chflux.tools import timestamp_parsers


def read_yaml(filepath):
    """Read a YAML file as a dict.  Return an empty dict if fail to read."""
    with open(filepath, 'r') as f:
        try:
            ydict = yaml.load(f)
        except yaml.YAMLError as exc_yaml:
            print(exc_yaml)
            ydict = {}  # fall back to an empty dict if fail to read

    return ydict


def update_dict(dct, updater):
    """
    (dict, dict) -> dict

    Return a dict from updating the keys of `dct` with the `updater` dict.
    Both can be nested dicts of arbitrary depth.

    Note: The original dict, `dct`, is unchanged.
    """
    def _update_dict_altered(dct, updater):
        """
        This inner function performs the updating by recursion.  It will alter
        the input `dct`.
        """
        for k, v in updater.items():
            if (k in dct and isinstance(dct[k], dict) and
                    isinstance(updater[k], collections.Mapping)):
                _update_dict_altered(dct[k], updater[k])
            else:
                dct[k] = updater[k]

    dct_copy = copy.deepcopy(dct)
    _update_dict_altered(dct_copy, updater)
    return dct_copy


def read_tabulated_data(data_name, config, query=None):
    """
    A generalized function to read tabulated data specified in the config.

    Parameters
    ----------
    data_name : str
        Data name, allowed values are
        - 'biomet': biometeorological data
        - 'conc': concentration data
        - 'flow': flow rate data
        - 'leaf': leaf area data
        - 'timelag': timelag data
    config : dict
        Configuration dictionary parsed from the YAML config file.
    query : list
        A list of query strings used to search in all available data files.
        If `None` (default), read all data files.

    Return
    ------
    df : pandas.DataFrame
        The loaded tabulated data.
    """
    # check the validity of `data_name` parameter
    if data_name not in ['biomet', 'conc', 'flow', 'leaf', 'timelag']:
        raise RuntimeError('Wrong data name.  Allowed values are ' +
                           "'biomet', 'conc', 'flow', 'leaf', 'timelag'.")
    # get file list
    data_flist = glob.glob(config['data_dir'][data_name + '_data'])
    # get the data settings
    data_settings = config[data_name + '_data_settings']

    # ensure that `query` is a list
    if type(query) is str:
        query = [query]

    # filter the list of data files with query strings
    if query is not None:
        data_flist = [f for f in data_flist if any(q in f for q in query)]
        data_flist = sorted(data_flist)  # ensure the list is sorted by name

    # check data file existence
    if not len(data_flist):
        print('Cannot find the %s data file!' % data_name)
        return None
    else:
        print('%d %s data files are found. ' % (len(data_flist), data_name) +
              'Loading...')

    # check date parser: if legit, use it; if not, set it to `None`
    if data_settings['date_parser'] in timestamp_parsers:
        date_parser = timestamp_parsers[data_settings['date_parser']]
    else:
        date_parser = None

    read_csv_options = {
        'sep': data_settings['delimiter'],
        'header': data_settings['header'],
        'names': data_settings['names'],
        'usecols': data_settings['usecols'],
        'dtype': data_settings['dtype'],
        'na_values': data_settings['na_values'],
        'parse_dates': data_settings['parse_dates'],
        'date_parser': date_parser,
        'infer_datetime_format': True,
        'engine': 'c',
        'encoding': 'utf-8'}
    df_loaded = \
        [pd.read_csv(entry, **read_csv_options) for entry in data_flist]

    # echo the list of data files
    for entry in data_flist:
        print(entry)

    try:
        df = pd.concat(df_loaded, ignore_index=True)
    except ValueError:
        print('Cannot concatenate data tables!')
        # if the list to concatenate is empty
        return None

    del df_loaded

    # echo data status
    print('%d lines read from %s data.' % (df.shape[0], data_name))

    return df
