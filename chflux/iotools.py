"""
A collection of input/output tools for PyChamberFlux

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
import yaml
import glob
import warnings
import pandas as pd


# a collection of date parsers for timestamps stored in multiple columns
# This does not support month-first (American) or day-first (European) format,
# put them by the year-month-day order (ISO 8601) using index orders.
date_parsers_dict = {
    # date only
    'ymd': lambda s: pd.to_datetime(s, format='%Y %m %d'),

    # down to minute
    'ymdhm': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M'),

    # down to second
    'ymdhms': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S'),

    # down to nanosecond
    'ymdhmsf': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S %f')
}


def load_config(filepath):
    """Load YAML configuration file from a given filepath."""
    with open(filepath, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc_yaml:
            print(exc_yaml)
            config = {}  # return a blank dict if fail to load

    return config


def load_tabulated_data(data_name, config, query=None):
    """
    A general function to read tabulated data (biometeorological,
    concentration, flow rate, leaf area, and timelag data).

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
        List of the query strings used to search in all available data files.
        If `None` (default), read all data files.

    Return
    ------
    df : pandas.DataFrame
        The loaded tabulated data.
    """
    # check the validity of `data_name` parameter
    if data_name not in ['biomet', 'conc', 'flow', 'leaf', 'timelag']:
        raise RuntimeError('Wrong data name. Allowed values are ' +
                           "'biomet', 'conc', 'flow', 'leaf', 'timelag'.")
    # search file list
    data_flist = glob.glob(config['data_dir'][data_name + '_data'])
    # get the data settings
    data_settings = config[data_name + '_data_settings']

    if type(query) is str:
        # `query` must be a list
        query = [query]

    if query is not None:
        data_flist = [f for f in data_flist if any(q in f for q in query)]
        data_flist = sorted(data_flist)  # ensure the list is sorted by name

    # check data existence
    if not len(data_flist):
        print('Cannot find the %s data file!' % data_name)
        return None
    else:
        print('%d %s data files are found. ' % (len(data_flist), data_name) +
              'Loading...')

    # check date parser: if legit, extract it, and if not, set it to `None`
    if data_settings['date_parser'] in date_parsers_dict:
        date_parser = date_parsers_dict[data_settings['date_parser']]
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
        'encoding': 'utf-8', }
    df_loaded = \
        [pd.read_csv(entry, **read_csv_options) for entry in data_flist]

    for entry in data_flist:
        print(entry)

    try:
        df = pd.concat(df_loaded, ignore_index=True)
    except ValueError:
        # if the list to concatenate is empty
        return None

    del df_loaded

    # parse 'doy' as 'time_doy'
    if 'doy' in df.columns and 'time_doy' not in df.columns:
        df.rename(columns={'doy': 'time_doy'}, inplace=True)

    # parse 'datetime' as 'timestamp'
    if 'datetime' in df.columns and 'timestamp' not in df.columns:
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)

    # echo data status
    print('%d lines read from %s data.' % (df.shape[0], data_name))

    # the year number to which the day of year values are referenced
    year_ref = config['%s_data_settings' % data_name]['year_ref']

    # parse time variables if not already exist
    if 'timestamp' in df.columns.values:
        if type(df.loc[0, 'timestamp']) is not pd.Timestamp:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # note: no need to catch out-of-bound error if set 'coerce'
        if 'time_doy' not in df.columns.values:
            # add a time variable in days of year (float) if not already there
            year_start = year_ref if year_ref is not None else \
                df.loc[0, 'timestamp'].year
            df['time_doy'] = (df['timestamp'] -
                              pd.Timestamp('%d-01-01' % year_start)) / \
                pd.Timedelta(days=1)
    elif 'time_doy' in df.columns.values:
        # starting year must be specified for day of year
        year_start = year_ref
        df['timestamp'] = pd.Timestamp('%d-01-01' % year_start) + \
            df['time_doy'] * pd.Timedelta(days=1.)
    elif 'time_sec' in df.columns.values:
        time_sec_start = config[
            '%s_data_settings' % data_name]['time_sec_start']
        if time_sec_start is None:
            time_sec_start = 1904
        df['timestamp'] = pd.Timestamp('%d-01-01' % time_sec_start) + \
            df['time_sec'] * pd.Timedelta(seconds=1)
        # add a time variable in days of year (float)
        year_start = year_ref if year_ref is not None else \
            df.loc[0, 'timestamp'].year
        year_start_in_sec = (
            pd.Timestamp('%d-01-01' % year_start) -
            pd.Timestamp('%d-01-01' % time_sec_start)) / \
            pd.Timedelta(seconds=1)
        df['time_doy'] = (df['time_sec'] - year_start_in_sec) / 86400.
    else:
        warnings.warn('No time variable is found!', UserWarning)

    return df


def create_output_header(data_type, species_list, biomet_var_list=[]):
    """
    A helper function to create the header for output data frame.

    Parameters
    ----------
    data_type : str
        The type of output dataframe
        * 'flux' - flux data
        * 'diag' - curve fitting diagnostics
    species_list : list of str
        List of gas species
    biomet_var_list : list of str
        List of biometeorological variable names

    Returns
    -------
    header : list of str
        Table header for the output dataframe. If `data_type` is illegal,
        return a blank list.
    """
    if data_type == 'flux':
        header = ['doy_utc', 'doy_local', 'ch_no', 'ch_label', 'A_ch', 'V_ch']
        for conc_suffix in ['_atmb', '_chb', '_cha', '_atma']:
            header += [s + conc_suffix for s in species_list]
            header += ['sd_' + s + conc_suffix for s in species_list]

        header += [s + '_chc_iqr' for s in species_list]
        for flux_method in ['_lin', '_rlin', '_nonlin']:
            header += ['f' + s + flux_method for s in species_list]
            header += ['se_f' + s + flux_method for s in species_list]

        # add quality flags for fluxes
        header += ['qc_' + s for s in species_list]

        # add number of valid observations of concentrations
        header += ['n_obs_' + s for s in species_list]

        # biomet variables and other auxiliary variables
        header += ['flow_lpm', 't_turnover', 't_lag_nom', 't_lag_optmz',
                   'status_tlag', 'pres', 'T_log', 'T_inst'] + biomet_var_list
    elif data_type == 'diag':
        header = ['doy_utc', 'doy_local', 'ch_no']
        for s in species_list:
            header += ['k_lin_' + s, 'b_lin_' + s, 'r_lin_' + s,
                       'p_lin_' + s, 'rmse_lin_' + s, 'delta_lin_' + s]

        for s in species_list:
            header += ['k_rlin_' + s, 'b_rlin_' + s,
                       'k_lolim_rlin_' + s, 'k_uplim_rlin_' + s,
                       'rmse_rlin_' + s, 'delta_rlin_' + s]

        for s in species_list:
            header += ['p0_nonlin_' + s, 'p1_nonlin_' + s,
                       'se_p0_nonlin_' + s, 'se_p1_nonlin_' + s,
                       'rmse_nonlin_' + s, 'delta_nonlin_' + s]
    else:
        return []

    return header
