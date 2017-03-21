"""
Main program for flux calculation

(c) Wu Sun <wu.sun@ucla.edu> 2016-2017

"""
import os
import glob
import datetime
import argparse
import warnings
import yaml
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker

from common_func import *
from default_config import default_config


# Command-line argument parser
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyChamberFlux: Main program for flux calculation.')
parser.add_argument('-c', '--config', dest='config',
                    action='store', help='set the config file')

args = parser.parse_args()

# current_dir = os.path.dirname(os.path.abspath(__file__))
# default_config_filepath = current_dir + '/config.yaml'

# if args.config is None:
#     args.config = default_config_filepath


# Global settings (not from the config file)
# =============================================================================
if LooseVersion(mpl.__version__) < LooseVersion('2.0.0'):
    # enforce sans-serif math for matplotlib version before 2.0.0
    plt.rcParams.update({'mathtext.default': 'regular'})

# suppress the annoying numpy runtime warning of "mean of empty slice"
# @FIXME: warning suppression is lame; needs to be improved
warnings.simplefilter('ignore', category=RuntimeWarning)
# suppress the annoying matplotlib tight_layout user warning
# warnings.simplefilter('ignore', category=UserWarning)


# a collection of date parsers to use, when dates are stored in multiple
# columns, like YYYY MM DD etc.
# does not support month-first (American) or day-first (European) format,
# put them by the year-month-day order using index orders.
date_parsers_dict = {
    # date only
    'ymd': lambda s: pd.to_datetime(s, format='%Y %m %d'),
    # down to minute
    'ymdhm': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M'),
    # down second
    'ymdhms': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S'),
    # down to nanosecond
    'ymdhmsf': lambda s: pd.to_datetime(s, format='%Y %m %d %H %M %S %f')
}


def load_config(filepath):
    """Load configuration file from a given filepath."""
    with open(filepath, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc_yaml:
            print(exc_yaml)

    return config


def load_tabulated_data(data_name, config, query=None):
    """
    A general function to read tabulated data (biometeorological,
    concentration, flow rate, and leaf area data).

    Parameters
    ----------
    data_name : str
        Data name, allowed values are
        - 'biomet': biometeorological data
        - 'conc': concentration data
        - 'flow': flow rate data
        - 'leaf': leaf area data
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
    # check the legality of `data_name` parameter
    if data_name not in ['biomet', 'conc', 'flow', 'leaf']:
        raise RuntimeError('Wrong data name. Allowed values are ' +
                           "'biomet', 'conc', 'flow', 'leaf'.")
    # search file list
    data_flist = glob.glob(config['data_dir'][data_name + '_data'])
    # get the data settings
    data_settings = config[data_name + '_data_settings']

    # check data existence
    if not len(data_flist):
        print('Cannot find the %s data file!' % data_name)
        return None
    else:
        print('%d %s data files are found. ' % (len(data_flist), data_name) +
              'Loading...')

    if query is not None:
        data_flist = [f for f in data_flist if any(q in f for q in query)]
        data_flist = sorted(data_flist)  # ensure the list is sorted by name

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
        df = None  # if the list to concatenate is empty
        return df  # return None if not a valid DataFrame

    del df_loaded

    # @DEPRECATED
    # load all data
    # use pd.concat() + list comprehension to boost speed
    # df = None
    # for entry in data_flist:
    #     print(entry)
    #     df_loaded = pd.read_csv(
    #         entry, delimiter=data_settings['delimiter'],
    #         header=data_settings['header'],
    #         names=data_settings['names'],
    #         usecols=data_settings['usecols'],
    #         dtype=data_settings['dtype'],
    #         na_values=data_settings['na_values'],
    #         parse_dates=data_settings['parse_dates'],
    #         date_parser=date_parser,
    #         infer_datetime_format=True,
    #         engine='c', encoding='utf-8')
    #     # Note: sometimes it may need explicit definitions of data types to
    #     # avoid a numpy NaN-to-integer error
    #     if df is None:
    #         df = df_loaded
    #     else:
    #         df = pd.concat([df, df_loaded], ignore_index=True)
    #     del(df_loaded)

    # parse 'doy' as 'time_doy'
    if 'doy' in df.columns and 'time_doy' not in df.columns:
        df.rename(columns={'doy': 'time_doy'}, inplace=True)

    # parse 'datetime' as 'timestamp'
    if 'datetime' in df.columns and 'timestamp' not in df.columns:
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)

    # echo data status
    print('%d lines read from %s data.' % (df.shape[0], data_name))

    # parse time variables if not already exist
    if 'timestamp' in df.columns.values:
        if type(df.loc[0, 'timestamp']) is not pd.Timestamp:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # no need to catch out-of bound error if set 'coerce'
        # add a time variable in days of year (float) if not already there
        if 'time_doy' not in df.columns.values:
            year_start = df.loc[0, 'timestamp'].year
            df['time_doy'] = (df['timestamp'] -
                              pd.Timestamp('%d-01-01 00:00' % year_start)) / \
                pd.Timedelta(days=1)
    elif 'time_doy' in df.columns.values:
        # starting year must be specified for day of year
        year_start = config['%s_data_settings' % data_name]['year_ref']
        df['timestamp'] = pd.Timestamp('%d-01-01 00:00' % year_start) + \
            df['time_doy'] * pd.Timedelta(days=1.)
    elif 'time_sec' in df.columns.values:
        time_sec_start = config[
            '%s_data_settings' % data_name]['time_sec_start']
        if time_sec_start is None:
            time_sec_start = 1904
        df['timestamp'] = pd.Timestamp('%d-01-01 00:00' % time_sec_start) + \
            df['time_sec'] * pd.Timedelta(seconds=1)
        # add a time variable in days of year (float) if not already there
        if 'time_doy' not in df.columns.values:
            year_start = df.loc[0, 'timestamp'].year
            year_start_in_sec = (
                pd.Timestamp('%d-01-01' % year_start) -
                pd.Timestamp('%d-01-01' % time_sec_start)) / \
                pd.Timedelta(seconds=1)
            df['time_doy'] = (df['time_sec'] - year_start_in_sec) / 86400.
    else:
        warnings.warn('No time variable is found!', UserWarning)

    return df


def flux_calc(df_biomet, df_conc, df_flow, df_leaf,
              doy, year, config, chamber_config):
    """
    Calculate fluxes and generate plots.

    Parameters
    ----------
    df_biomet : pandas.DataFrame
        The biometeorological data.
    df_conc : pandas.DataFrame
        The concentration data.
    doy : float
        Day of year number of the current function call. Note that `doy`
        here is the fractional DOY, always smaller than the integer DOY
        (Julian day number).
    year : int
        Current year in four digits.
    config : dict
        Configuration dictionary parsed from the YAML config file.

    Returns
    -------
    None

    """
    # settings
    # =========================================================================
    # unpack config
    run_options = config['run_options']
    data_dir = config['data_dir']
    # biomet_data_settings = config['biomet_data_settings']
    # conc_data_settings = config['conc_data_settings']
    # flow_data_settings = config['flow_data_settings']
    # consts = config['constants']
    site_parameters = config['site_parameters']
    species_settings = config['species_settings']

    if run_options['plot_style'] is not None:
        plt.style.use(run_options['plot_style'])

    # extract species settings
    n_species = len(species_settings['species_list'])
    species_list = species_settings['species_list']
    conc_factor = [species_settings[s]['multiplier'] for s in species_list]
    species_unit_names = []
    for i, s in enumerate(species_settings['species_list']):
        if np.isclose(species_settings[s]['output_unit'], 1e-12):
            unit_name = 'pmol mol$^{-1}$'
        elif np.isclose(species_settings[s]['output_unit'], 1e-9):
            unit_name = 'nmol mol$^{-1}$'
        elif np.isclose(species_settings[s]['output_unit'], 1e-6):
            unit_name = '$\mu$mol mol$^{-1}$'
        elif np.isclose(species_settings[s]['output_unit'], 1e-3):
            unit_name = 'mmol mol$^{-1}$'
        elif np.isclose(species_settings[s]['output_unit'], 1e-2):
            unit_name = '%'
        elif np.isclose(species_settings[s]['output_unit'], 1.):
            unit_name = 'mol mol$^{-1}$'
        else:
            unit_name = 'undefined unit'
        species_unit_names.append(unit_name)

    # create or locate directories for output
    # for output data
    output_dir = data_dir['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for daily flux summary plots
    if run_options['save_daily_plots']:
        daily_plots_dir = data_dir['plot_dir'] + '/daily_plots/'
        if not os.path.exists(daily_plots_dir):
            os.makedirs(daily_plots_dir)

    # a date string for current run; used in echo and in output file names
    run_date_str = (datetime.datetime(year, 1, 1) +
                    datetime.timedelta(doy + 0.5)).strftime('%Y%m%d')

    # for fitting plots
    if run_options['save_fitting_plots']:
        fitting_plots_path = data_dir['plot_dir'] + \
            '/fitting/%s/' % run_date_str
        if not os.path.exists(fitting_plots_path):
            os.makedirs(fitting_plots_path)

    # unpack time variables
    # @TODO: switch from day of year based subsetting to timestamp subsetting
    if 'timestamp' not in df_biomet.columns.values:
        raise RuntimeError('No time variable found in the biomet data.')

    if 'time_doy' in df_conc.columns.values:
        doy_conc = df_conc['time_doy'].values
    else:
        raise RuntimeError(
            'No time variable found in the concentration data.')

    if 'time_doy' in df_flow.columns.values:
        doy_flow = df_flow['time_doy'].values
    else:
        raise RuntimeError(
            'No time variable found in the flow rate data.')

    if data_dir['separate_leaf_data']:
        if 'time_doy' in df_leaf.columns.values:
            doy_leaf = df_leaf['time_doy'].values
        else:
            raise RuntimeError(
                'No time variable found in the leaf area data.')

    # get today's chamber schedule: `ch_start` and `ch_no`
    # =========================================================================
    timer = 0.
    ch_no = np.array([], dtype='int')
    ch_start = np.array([])
    while timer < 1.:
        chlut = chamber_lookup_table_func(doy + timer, chamber_config)
        df_chlut = chlut.df
        # n_ch = chlut.n_ch
        smpl_cycle_len = chlut.smpl_cycle_len
        # n_cycle_per_day = chlut.n_cycle_per_day
        schedule_end = chlut.schedule_end

        ch_start = np.append(ch_start,
                             df_chlut['ch_start'].values + doy + timer)
        ch_no = np.append(ch_no, df_chlut['ch_no'].values)
        timer += smpl_cycle_len
        if doy + timer > schedule_end:
            # if the schedule is switched
            ch_no = ch_no[ch_start < schedule_end]
            ch_start = ch_start[ch_start < schedule_end]
            switch_index = ch_no.size
            # apply the switched schedule
            # chlut_now, n_ch, smpl_cycle_len, n_cycle_per_day, _, = \
            #     chamber_lookup_table_func_old(doy + timer, return_all=True)
            chlut = chamber_lookup_table_func(doy + timer, chamber_config)
            df_chlut = chlut.df
            # n_ch = chlut.n_ch
            smpl_cycle_len = chlut.smpl_cycle_len
            # n_cycle_per_day = chlut.n_cycle_per_day
            schedule_end = chlut.schedule_end

            timer = np.floor(timer / smpl_cycle_len - 1) * smpl_cycle_len
            ch_start = np.append(ch_start,
                                 df_chlut['ch_start'].values + doy + timer)
            ch_no = np.append(ch_no, df_chlut['ch_no'].values)
            # remove duplicate segment
            ch_start[switch_index:][
                ch_start[switch_index:] < schedule_end] = np.nan
            ch_no = ch_no[np.isfinite(ch_start)]
            ch_start = ch_start[np.isfinite(ch_start)]
            timer += smpl_cycle_len

    ch_no = ch_no[ch_start < doy + 1.]
    # note: `ch_no` defined above are the nominal chamber numbers
    # it needs to be updated with the actual chamber numbers, if such variable
    # is recorded in the biomet data table

    ch_start = ch_start[ch_start < doy + 1.]

    # total number of possible samples of the current day
    n_smpl_per_day = ch_no.size

    # variable initialization
    # =========================================================================
    # times for chamber control actions (e.g., opening and closing)
    # with respect to 'ch_start', in fraction of a day
    # - 'ch_o_b': chamber open before closure
    # - 'ch_cls': chamber closing
    # - 'ch_o_a': chamber re-open after closure
    # - 'ch_atm_a': atmospheric line after closure, if exists
    # - 'ch_end': end of chamber sampling
    ch_o_b = np.zeros(n_smpl_per_day) * np.nan
    ch_cls = np.zeros(n_smpl_per_day) * np.nan
    ch_o_a = np.zeros(n_smpl_per_day) * np.nan
    ch_atm_a = np.zeros(n_smpl_per_day) * np.nan
    ch_end = np.zeros(n_smpl_per_day) * np.nan

    # time, chamber labels, and chamber parameters
    ch_time = np.zeros(n_smpl_per_day) * np.nan  # in day of year (fractional)
    # timestamps for chamber measurements defined as
    # the middle point of the closure period
    ch_label = []
    A_ch = np.zeros(n_smpl_per_day) * np.nan
    V_ch = np.zeros(n_smpl_per_day) * np.nan

    # conc and flux variables
    # - '_ch': the 15-min sampling period proper (not used)
    # - '_atmb': atmospheric line before chamber closure
    # - '_chb': chamber, before closure
    # - '_chc': chamber closure period
    # - '_cha': chamber, after closure
    # - '_atma': atmospheric line after chamber closure
    conc_atmb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    conc_chb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    conc_cha = np.zeros((n_smpl_per_day, n_species)) * np.nan
    conc_atma = np.zeros((n_smpl_per_day, n_species)) * np.nan

    sd_conc_atmb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_conc_chb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_conc_cha = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_conc_atma = np.zeros((n_smpl_per_day, n_species)) * np.nan

    conc_chc_iqr = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # fluxes calculated from: linear fit, robust linear fit, and nonlinear fit
    # - '_lin': flux estimate and standard error from linear fit
    # - '_rlin': flux estimate and standard error from robust linear fit
    # - '_nonlin': flux estimate and standard error from nonlinear fit
    flux_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    se_flux_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    flux_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    se_flux_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    flux_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    se_flux_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # save fitting diagnostics to a separate file
    # linear fitting diagnostics
    # --------------------------
    # - `k_lin`: slopes from linear fit
    # - `b_lin`: intercepts from linear fit
    # - `r_lin`: r values from linear fit
    # - `p_lin`: p values from linear fit
    # - `rmse_lin`: root mean square error of fitted concentrations
    # - `delta_lin`: fitted C_end - C_init, i.e.,
    #   fitted changes of concentration during the closure period
    k_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    b_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    r_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    p_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    rmse_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    delta_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # robust linear fitting diagnostics
    # ---------------------------------
    # - `k_rlin`: (median) slopes from robust linear fit
    # - `b_rlin`: intercepts from robust linear fit
    # - `k_lolim_rlin`: lower bounds of the confidence interval of the slope
    # - `k_uplim_rlin`: upper bounds of the confidence interval of the slope
    # - `rmse_rlin`: root mean square error of fitted concentrations
    # - `delta_rlin`: fitted C_end - C_init, i.e.,
    #   fitted changes of concentration during the closure period
    k_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    b_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    k_lolim_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    k_uplim_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    rmse_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    delta_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # nonlinear fit diagnostics
    # -------------------------
    # - `p0_nonlin`: parameter 0, the pre-exponential factor
    # - `p1_nonlin`: parameter 1, the small time lag assigned for better fit
    # - `se_p0_nonlin`: standard error of parameter 0
    # - `se_p1_nonlin`: standard error of parameter 1
    # root mean square error of fitted concentrations
    # - `delta_nonlin`: fitted C_end - C_init, i.e.,
    #   fitted changes of concentration during the closure period
    p0_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    p1_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    se_p0_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    se_p1_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    rmse_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    delta_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # search for biomet variable names
    # --------------------------------
    # but only those with possibly multiple names are searched
    # - `T_atm_names`: T_atm variable names
    # - `RH_atm_names`: RH_atm variable names
    # - `T_ch_names`: T_ch variable names
    # - `PAR_names`: PAR variable names (not associated with chambers)
    # - `PAR_ch_names`: PAR variable names [associated with chambers]
    # - `T_leaf_names`: T_leaf variable names
    # - `T_soil_names`: T_soil variable names
    # - `w_soil_names`: w_soil variable names
    # - `flow_ch_names`: flow_ch variable names
    T_atm_names = [s for s in df_biomet.columns.values
                   if 'T_atm' in s]
    RH_atm_names = [s for s in df_biomet.columns.values
                    if 'RH_atm' in s]
    T_ch_names = [s for s in df_biomet.columns.values
                  if 'T_ch' in s]
    PAR_names = [s for s in df_biomet.columns.values
                 if 'PAR' in s and 'PAR_ch' not in s]
    PAR_ch_names = [s for s in df_biomet.columns.values
                    if 'PAR_ch' in s]
    T_leaf_names = [s for s in df_biomet.columns.values
                    if 'T_leaf' in s]
    T_soil_names = [s for s in df_biomet.columns.values
                    if 'T_soil' in s]
    w_soil_names = [s for s in df_biomet.columns.values
                    if 'w_soil' in s]
    flow_ch_names = [s for s in df_flow.columns.values
                     if 'flow_ch' in s]

    # initialize biomet variables
    # ---------------------------
    # - `T_atm`: atmospheric temperature in Celsius degree
    # - `RH_atm`: atmospheric relative humidity in percentage
    # - `T_ch`: chamber temperatures in Celsius degree
    # - `T_dew_ch`: dew point temp in chambers, in Celsius degree
    # - `PAR`: PAR (not associated with chambers) in mumol m^-2 s^-1
    # - `PAR_ch`: PAR associated with chambers in mumol m^-2 s^-1
    # - `T_leaf`: leaf temperatures in Celsius degree
    # - `T_soil`: soil temperatures in Celsius degree
    # - `w_soil`: soil moisture in volumetric fraction (m^3 water m^-3 soil)
    # - `flow`: flow rate of the chamber currently being measured (mol s^-1)
    # - `flow_lpm`: chamber flowrate in liter per min
    #   (convert standard liter per minute to liter per minute if needed)
    # - `V_ch_mol`: chamber volume converted to moles of air
    # - `t_turnover`: chamber air turnover time in sec,
    #   equals to (V_ch_mol / flow), or (V_ch / flow_lpm)
    pres = np.zeros(n_smpl_per_day) * np.nan  # ambient pressure in Pascal
    T_log = np.zeros(n_smpl_per_day) * np.nan   # datalogger panel temp
    T_inst = np.zeros(n_smpl_per_day) * np.nan  # gas analyzer temp

    if len(T_atm_names) > 0:
        T_atm = np.zeros((n_smpl_per_day, len(T_atm_names))) * np.nan

    if len(RH_atm_names) > 0:
        RH_atm = np.zeros((n_smpl_per_day, len(RH_atm_names))) * np.nan

    if len(T_ch_names) > 0:
        T_ch = np.zeros((n_smpl_per_day, len(T_ch_names))) * np.nan

    if len(PAR_names) > 0:
        PAR = np.zeros((n_smpl_per_day, len(PAR_names))) * np.nan

    if len(PAR_ch_names) > 0:
        PAR_ch = np.zeros((n_smpl_per_day, len(PAR_ch_names))) * np.nan

    if len(T_leaf_names) > 0:
        T_leaf = np.zeros((n_smpl_per_day, len(T_leaf_names))) * np.nan

    if len(T_soil_names) > 0:
        T_soil = np.zeros((n_smpl_per_day, len(T_soil_names))) * np.nan

    if len(w_soil_names) > 0:
        w_soil = np.zeros((n_smpl_per_day, len(w_soil_names))) * np.nan

    T_dew_ch = np.zeros(n_smpl_per_day) * np.nan
    flow = np.zeros(n_smpl_per_day) * np.nan
    flow_lpm = np.zeros(n_smpl_per_day) * np.nan
    V_ch_mol = np.zeros(n_smpl_per_day) * np.nan
    t_turnover = np.zeros(n_smpl_per_day) * np.nan

    # timelag diagnostics
    # -------------------
    # - `time_lag_nominal`: nominal time lag in seconds
    # - `time_lag_optmz`: optimized time lag in seconds
    # - `status_time_lag_optmz`: status code for time lag optimization
    #   initial value is -1
    time_lag_nominal = np.zeros(n_smpl_per_day) * np.nan
    time_lag_optmz = np.zeros(n_smpl_per_day) * np.nan
    status_time_lag_optmz = np.zeros(n_smpl_per_day, dtype='int') - 1

    # loops for averaging biomet variables and calculating fluxes
    # =========================================================================
    for loop_num in range(n_smpl_per_day):
        # get the current chamber's meta info
        df_chlut_current = chamber_lookup_table_func(
            ch_start[loop_num], chamber_config).df
        df_chlut_current = df_chlut_current[
            df_chlut_current['ch_no'] == ch_no[loop_num]]
        A_ch[loop_num] = df_chlut_current['A_ch'].values[0]
        V_ch[loop_num] = df_chlut_current['V_ch'].values[0]
        ch_label.append(df_chlut_current['ch_label'].values[0])
        # Note: 'ch_label' is a list! not an array

        ch_o_b[loop_num] = ch_start[loop_num] + \
            df_chlut_current['ch_o_b'].values[0]
        ch_cls[loop_num] = ch_start[loop_num] + \
            df_chlut_current['ch_cls'].values[0]
        ch_o_a[loop_num] = ch_start[loop_num] + \
            df_chlut_current['ch_o_a'].values[0]
        ch_atm_a[loop_num] = ch_start[loop_num] + \
            df_chlut_current['ch_atm_a'].values[0]
        ch_end[loop_num] = ch_start[loop_num] + \
            df_chlut_current['ch_end'].values[0]

        ch_time[loop_num] = 0.5 * (ch_cls[loop_num] + ch_o_a[loop_num])

        # correct leaf area if supplied by external data
        if (data_dir['separate_leaf_data'] and df_leaf is not None and
                df_chlut_current['is_leaf_chamber'].values[0]):
            A_ch[loop_num] = np.interp(
                ch_time[loop_num], doy_leaf, df_leaf[ch_label[-1]].values)

        # # extract indices for averaging biomet variables, no time lag
        # ind_ch_biomet = np.where((doy_biomet >= ch_start[loop_num]) &
        #                          (doy_biomet < ch_end[loop_num]))[0]

        # extract indices for averaging biomet variables, no time lag needed
        ts_ch_start = pd.Timestamp('%d-01-01 00:00' % year) + \
            pd.Timedelta(days=ch_start[loop_num])
        ts_ch_end = pd.Timestamp('%d-01-01 00:00' % year) + \
            pd.Timedelta(days=ch_end[loop_num])
        ind_ch_biomet = np.where((df_biomet['timestamp'] >= ts_ch_start) &
                                 (df_biomet['timestamp'] < ts_ch_end))[0]

        # include the full chamber period
        n_ind_ch_biomet = ind_ch_biomet.size

        # variables averaged in the whole chamber period
        if n_ind_ch_biomet > 0:
            # ambient pressure in Pascal
            if 'pres' in df_biomet.columns.values:
                pres[loop_num] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, 'pres'].values)
            else:
                if site_parameters['site_pressure'] is None:
                    pres[loop_num] = phys_const['p_std']
                    # use standard atm pressure if no site pressure is defined
                else:
                    pres[loop_num] = site_parameters['site_pressure']
                    # use defined site pressure

            # datalogger panel temp
            if 'T_log' in df_biomet.columns.values:
                T_log[loop_num] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, 'T_log'].values)

            # instrument temperature
            if 'T_inst' in df_biomet.columns.values:
                T_inst[loop_num] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, 'T_inst'].values)

            # atmospheric temperature
            if len(T_atm_names) > 0:
                for i in range(len(T_atm_names)):
                    T_atm[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, T_atm_names[i]].values)

            # atmospheric RH
            if len(RH_atm_names) > 0:
                for i in range(len(RH_atm_names)):
                    RH_atm[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, RH_atm_names[i]].values)

            # chamber temperatures
            if len(T_ch_names) > 0:
                for i in range(len(T_ch_names)):
                    T_ch[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, T_ch_names[i]].values)

            # PAR (not associated with chambers)
            if len(PAR_names) > 0:
                for i in range(len(PAR_names)):
                    PAR[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, PAR_names[i]].values)

            # PAR associated with chambers
            if len(PAR_ch_names) > 0:
                for i in range(len(PAR_ch_names)):
                    PAR_ch[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, PAR_ch_names[i]].values)

            # leaf temperatures
            if len(T_leaf_names) > 0:
                for i in range(len(T_leaf_names)):
                    T_leaf[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, T_leaf_names[i]].values)

            # soil temperatures
            if len(T_soil_names) > 0:
                for i in range(len(T_soil_names)):
                    T_soil[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, T_soil_names[i]].values)

            # soil water content in volumetric fraction (m^3 water m^-3 soil)
            if len(w_soil_names) > 0:
                for i in range(len(w_soil_names)):
                    w_soil[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, w_soil_names[i]].values)

        # extract indices for averaging flow rates, no time lag
        ind_ch_flow = np.where((doy_flow >= ch_start[loop_num]) &
                               (doy_flow < ch_end[loop_num]))[0]
        # include the full chamber period
        n_ind_ch_flow = ind_ch_flow.size

        # flow rate is only needed for the chamber currently being measured
        if len(flow_ch_names) > 0:
            # find the column location to extract the flow rate of the current
            # chamber
            flow_loc = [k for k, s in enumerate(flow_ch_names)
                        if 'ch_' + str(ch_no[loop_num]) in s]
            if len(flow_loc) > 0:
                flow_lpm[loop_num] = \
                    np.nanmean(df_flow.loc[ind_ch_flow,
                                           flow_ch_names[flow_loc[0]]].values)
                # convert standard liter per minute to liter per minute, if
                # applicable
                if config['flow_data_settings']['flow_rate_in_STP']:
                    flow_lpm[loop_num] *= \
                        (T_ch[loop_num, ch_no[loop_num] - 1] +
                            phys_const['T_0']) / \
                        phys_const['T_0'] * \
                        phys_const['p_std'] / pres[loop_num]

        # convert volumetric flow to mass flow (mol s^-1)
        flow[loop_num] = flow_lpm[loop_num] * 1e-3 / 60. * \
            pres[loop_num] / phys_const['R_gas'] / \
            (T_ch[loop_num, ch_no[loop_num] - 1] + phys_const['T_0'])

        # convert chamber volume to mol
        V_ch_mol[loop_num] = V_ch[loop_num] * pres[loop_num] / \
            phys_const['R_gas'] / \
            (T_ch[loop_num, ch_no[loop_num] - 1] + phys_const['T_0'])

        t_turnover[loop_num] = V_ch_mol[loop_num] / flow[loop_num]
        # turnover time in seconds, useful in flux calculation

        # indices for flux calculation
        # - '_chb' - before closure
        # - '_chc' - during closure
        # - '_chs' - starting 1 min of the closure  # disabled
        # - '_cha' - after closure

        # these need to be assigned properly later
        dt_lmargin = 0.
        dt_rmargin = 0.
        time_lag_in_day = 0.

        # 'ind_ch_full' index is only used for plotting
        ind_ch_full = np.where((doy_conc > ch_o_b[loop_num]) &
                               (doy_conc < ch_end[loop_num] +
                                time_lag_in_day))[0]
        ind_atmb = np.where(
            (doy_conc > ch_start[loop_num] + time_lag_in_day + dt_lmargin) &
            (doy_conc < ch_o_b[loop_num] + time_lag_in_day - dt_rmargin))[0]
        ind_chb = np.where(
            (doy_conc > ch_o_b[loop_num] + time_lag_in_day + dt_lmargin) &
            (doy_conc < ch_cls[loop_num] + time_lag_in_day - dt_rmargin))[0]
        ind_chc = np.where(
            (doy_conc > ch_cls[loop_num] + time_lag_in_day + dt_lmargin) &
            (doy_conc < ch_o_a[loop_num] + time_lag_in_day - dt_rmargin))[0]
        # Note: after the line is switched, regardless of the time lag,
        # the analyzer will sample the next line.
        # This is the reason that a time lag is not added to the terminal time.
        ind_cha = np.where(
            (doy_conc > ch_o_a[loop_num] + time_lag_in_day + dt_lmargin) &
            (doy_conc < ch_atm_a[loop_num]))[0]
        ind_atma = np.where(
            (doy_conc > ch_atm_a[loop_num] + time_lag_in_day + dt_lmargin) &
            (doy_conc < ch_end[loop_num]))[0]

        n_ind_atmb = ind_atmb.size
        n_ind_chb = ind_chb.size
        n_ind_chc = ind_chc.size
        n_ind_cha = ind_cha.size
        n_ind_atma = ind_atma.size

        # check if there are enough data points for calculating fluxes
        # note that concentration data might not be sampled every second.
        # if this is the case, the criterion needs to be modified
        # `flag_calc_flux`: 0 - don't calculate fluxes; 1 - calculate fluxes
        if n_ind_chc >= 2. * 60. and flow[loop_num] > 0.:
            # needs at least 2 min good data in the closure period to proceed
            # flow rate value needs to be positive, otherwise the chamber
            # cannot be flushed by the inlet air
            flag_calc_flux = 1
        else:
            flag_calc_flux = 0

        # average the concentration data for output
        for spc_id in range(n_species):
            conc_atmb[loop_num, spc_id] = np.nanmean(
                df_conc.loc[ind_atmb, species_list[spc_id]].values) * \
                conc_factor[spc_id]
            sd_conc_atmb[loop_num, spc_id] = np.nanstd(
                df_conc.loc[ind_atmb, species_list[spc_id]].values, ddof=1) * \
                conc_factor[spc_id]

            conc_chb[loop_num, spc_id] = np.nanmean(
                df_conc.loc[ind_chb, species_list[spc_id]].values) * \
                conc_factor[spc_id]
            sd_conc_chb[loop_num, spc_id] = np.nanstd(
                df_conc.loc[ind_chb, species_list[spc_id]].values, ddof=1) * \
                conc_factor[spc_id]

            conc_cha[loop_num, spc_id] = np.nanmean(
                df_conc.loc[ind_cha, species_list[spc_id]].values) * \
                conc_factor[spc_id]
            sd_conc_cha[loop_num, spc_id] = np.nanstd(
                df_conc.loc[ind_cha, species_list[spc_id]].values, ddof=1) * \
                conc_factor[spc_id]

            conc_atma[loop_num, spc_id] = np.nanmean(
                df_conc.loc[ind_atma, species_list[spc_id]].values) * \
                conc_factor[spc_id]
            sd_conc_atma[loop_num, spc_id] = np.nanstd(
                df_conc.loc[ind_atma, species_list[spc_id]].values, ddof=1) * \
                conc_factor[spc_id]

            conc_chc_iqr[loop_num, spc_id] = IQR_func(
                df_conc.loc[ind_chc, species_list[spc_id]].values) * \
                conc_factor[spc_id]

        # if the species 'h2o' exist, calculate chamber dew temperature
        spc_id_h2o = np.where(np.array(species_list) == 'h2o')[0]
        if (conc_chb[loop_num, spc_id_h2o] > 0 and
            conc_chb[loop_num, spc_id_h2o] *
                species_settings['h2o']['output_unit'] <= 1.):
            T_dew_ch[loop_num] = dew_temp(
                conc_chb[loop_num, spc_id_h2o] *
                species_settings['h2o']['output_unit'] * pres[loop_num])
        else:
            T_dew_ch[loop_num] = np.nan

        # fitted conc and baselines; save them for plotting purposes
        # only need two points to draw a line for each species
        # - `conc_bl_pts`: before and after closure points that mark the
        #    zero-flux baseline
        # - `t_bl_pts`: times for the two points that mark the baseline
        # - `conc_bl`: fitted baselines
        # - `conc_fitted_lin`: fitted concentrations during closure,
        #    from the simple linear method
        # - `conc_fitted_rlin`: fitted concentrations during closure,
        #    from the robust linear method
        # - `conc_fitted_nonlin`: fitted concentrations during closure,
        #    from the nonlinear method
        conc_bl_pts = np.zeros((n_species, 2))
        t_bl_pts = np.zeros(2)

        conc_bl = np.zeros((n_species, n_ind_chc))
        conc_fitted_lin = np.zeros((n_species, n_ind_chc))
        conc_fitted_rlin = np.zeros((n_species, n_ind_chc))
        conc_fitted_nonlin = np.zeros((n_species, n_ind_chc))

        if flag_calc_flux:
            # loop through each species
            for spc_id in range(n_species):
                # extract closure segments and convert the DOY to seconds
                # after 'ch_start' time for fitting plots
                # - `ch_full_time`: the whole sampling interval
                # - `atmb_time`: atmospheric line, before closure
                # - `chb_time`: chamber open, before closure
                # - `chc_time`: chamber closure
                # - `cha_time`: chamber open, after closure
                # - `atma_time`: atmospheric line, after closure
                ch_full_time = (doy_conc[ind_ch_full] -
                                ch_start[loop_num]) * 86400.
                chb_time = (doy_conc[ind_chb] - ch_start[loop_num]) * 86400.
                atmb_time = (doy_conc[ind_atmb] - ch_start[loop_num]) * 86400.
                chc_time = (doy_conc[ind_chc] - ch_start[loop_num]) * 86400.
                cha_time = (doy_conc[ind_cha] - ch_start[loop_num]) * 86400.
                atma_time = (doy_conc[ind_atma] - ch_start[loop_num]) * 86400.

                # conc of current species defined with 'spc_id'
                chc_conc = df_conc.loc[
                    ind_chc, species_list[spc_id]].values * conc_factor[spc_id]

                # calculate slopes and intercepts of the zero-flux baselines
                # baseline end points changed from mean to medians (05/05/2016)
                # - `median_chb_time`: median time for chamber open period
                #   (before closure), in seconds
                # - `median_cha_time`: median time for chamber open period
                #   (after closure), in seconds
                median_chb_time = np.nanmedian(
                    doy_conc[ind_chb] - ch_start[loop_num]) * 86400.
                median_cha_time = np.nanmedian(
                    doy_conc[ind_cha] - ch_start[loop_num]) * 86400.
                median_chb_conc = np.nanmedian(
                    df_conc.loc[ind_chb, species_list[spc_id]].values) * \
                    conc_factor[spc_id]
                median_cha_conc = np.nanmedian(
                    df_conc.loc[ind_cha, species_list[spc_id]].values) * \
                    conc_factor[spc_id]
                # if `median_cha_conc` is not a finite value, set it equal to
                # `median_chb_conc`. Thus `k_bl` will be zero.
                if np.isnan(median_cha_conc):
                    median_cha_conc = median_chb_conc
                    median_cha_time = chc_time[-1]

                k_bl = (median_cha_conc - median_chb_conc) / \
                    (median_cha_time - median_chb_time)
                b_bl = median_chb_conc - k_bl * median_chb_time

                # subtract the baseline to correct for instrument drift
                # (assuming linear drift)
                conc_bl = k_bl * chc_time + b_bl

                # linear fit
                # -------------------------------------------------------------
                # see the supp. info of Sun et al. (2016) JGR-Biogeosci.
                y_fit = (chc_conc - conc_bl) * flow[loop_num] / A_ch[loop_num]
                x_fit = np.exp(- (chc_time - chc_time[0] +
                                  (time_lag_in_day + dt_lmargin) * 8.64e4) /
                               t_turnover[loop_num])

                # boolean index array for finite concentration values
                ind_conc_fit = np.isfinite(y_fit)

                # if no finite concentration values, skip the current step
                if np.sum(ind_conc_fit) == 0:
                    continue

                slope, intercept, r_value, p_value, se_slope = \
                    stats.linregress(x_fit[ind_conc_fit], y_fit[ind_conc_fit])

                # save the fitted conc values
                conc_fitted_lin[spc_id, :] = (slope * x_fit + intercept) * \
                    A_ch[loop_num] / flow[loop_num] + conc_bl

                # save the linear fit results and diagnostics
                flux_lin[loop_num, spc_id] = - slope
                se_flux_lin[loop_num, spc_id] = np.abs(se_slope)
                k_lin[loop_num, spc_id] = slope
                b_lin[loop_num, spc_id] = intercept
                r_lin[loop_num, spc_id] = r_value
                p_lin[loop_num, spc_id] = p_value
                rmse_lin[loop_num, spc_id] = np.sqrt(
                    np.nanmean((conc_fitted_lin[spc_id, :] - chc_conc) ** 2))
                delta_lin[loop_num, spc_id] = \
                    (conc_fitted_lin[spc_id, -1] - conc_bl[-1]) - \
                    (conc_fitted_lin[spc_id, 0] - conc_bl[0])

                # clear temporary fitting parameters
                del slope, intercept, r_value, p_value, se_slope

                # robust linear fit
                # -------------------------------------------------------------
                medslope, medintercept, lo_slope, up_slope = \
                    stats.theilslopes(y_fit, x_fit, alpha=0.95)

                # save the fitted conc values
                conc_fitted_rlin[spc_id, :] = \
                    (medslope * x_fit + medintercept) * A_ch[loop_num] / \
                    flow[loop_num] + conc_bl

                # save the robust linear fit results and diagnostics
                flux_rlin[loop_num, spc_id] = - medslope
                se_flux_rlin[loop_num, spc_id] = \
                    np.abs(up_slope - lo_slope) / 3.92
                # Note: 0.95 C.I. is equivalent to +/- 1.96 sigma
                k_rlin[loop_num, spc_id] = medslope
                b_rlin[loop_num, spc_id] = medintercept
                k_lolim_rlin[loop_num, spc_id] = lo_slope
                k_uplim_rlin[loop_num, spc_id] = up_slope
                rmse_rlin[loop_num, spc_id] = np.sqrt(
                    np.nanmean((conc_fitted_rlin[spc_id, :] - chc_conc) ** 2))
                delta_rlin[loop_num, spc_id] = \
                    (conc_fitted_rlin[spc_id, -1] - conc_bl[-1]) - \
                    (conc_fitted_rlin[spc_id, 0] - conc_bl[0])

                # clear temporary fitted parameters
                del medslope, medintercept, lo_slope, up_slope

                # nonlinear fit
                # -------------------------------------------------------------
                t_fit = (chc_time - chc_time[0] +
                         (time_lag_in_day + dt_lmargin) * 8.64e4) / \
                    t_turnover[loop_num]
                params_nonlin_guess = [- flux_lin[loop_num, spc_id], 0.]
                params_nonlin = optimize.least_squares(
                    resid_conc_func, params_nonlin_guess,
                    bounds=([-np.inf, -10. / t_turnover[loop_num]],
                            [np.inf, 10. / t_turnover[loop_num]]),
                    loss='soft_l1', f_scale=0.5,
                    args=(t_fit[ind_conc_fit], y_fit[ind_conc_fit]))

                # save the fitted conc values
                conc_fitted_nonlin[spc_id, :] = \
                    conc_func(params_nonlin.x, t_fit) * A_ch[loop_num] / \
                    flow[loop_num] + conc_bl

                # standard errors of estimated parameters
                # `J^T J` is a Gauss-Newton approximation of the negative of
                # the Hessian of the cost function.
                # The variance-covariance matrix of the parameter estimates is
                # the inverse of the negative of Hessian matrix evaluated
                # at the parameter estimates.
                neg_hess = np.dot(params_nonlin.jac.T, params_nonlin.jac)
                # debug print: check if the hessian is positive definite
                # print(np.all(np.linalg.eigvals(neg_hess) > 0))
                try:
                    inv_neg_hess = np.linalg.inv(neg_hess)
                except np.linalg.LinAlgError:
                    try:
                        inv_neg_hess = np.linalg.pinv(neg_hess)
                    except np.linalg.LinAlgError:
                        inv_neg_hess = neg_hess * np.nan
                # variance-covariance matrix of parameter estimates
                MSE = np.nansum(params_nonlin.fun ** 2) / (t_fit.size - 2)
                pcov = inv_neg_hess * MSE
                # save the robust linear fit results and diagnostics
                flux_nonlin[loop_num, spc_id] = params_nonlin.x[0]
                se_flux_nonlin[loop_num, spc_id] = np.sqrt(pcov[0, 0])
                p0_nonlin[loop_num, spc_id] = params_nonlin.x[0]
                p1_nonlin[loop_num, spc_id] = params_nonlin.x[1]
                se_p0_nonlin[loop_num, spc_id] = np.sqrt(pcov[0, 0])
                se_p1_nonlin[loop_num, spc_id] = np.sqrt(pcov[1, 1])
                rmse_nonlin[loop_num, spc_id] = \
                    np.sqrt(np.nanmean((conc_fitted_nonlin[spc_id, :] -
                                        chc_conc) ** 2))
                delta_nonlin[loop_num, spc_id] = \
                    (conc_fitted_nonlin[spc_id, -1] - conc_bl[-1]) - \
                    (conc_fitted_nonlin[spc_id, 0] - conc_bl[0])

                # clear temporary fitted parameters
                del params_nonlin_guess, params_nonlin, neg_hess, \
                    inv_neg_hess, MSE, pcov

                # save the baseline conc's
                # baseline end points changed from mean to medians
                conc_bl_pts[spc_id, :] = median_chb_conc, median_cha_conc

            # used for plotting the baseline
            t_bl_pts[:] = median_chb_time, median_cha_time

            # generate fitting plots
            # -----------------------------------------------------------------
            if flag_calc_flux and run_options['save_fitting_plots']:
                fig, axes = plt.subplots(nrows=n_species, sharex=True,
                                         figsize=(8, 3 * n_species))

                for i, s in enumerate(species_list):
                    # color different time segments
                    axes[i].plot(
                        ch_full_time,
                        df_conc[s].values[ind_ch_full] * conc_factor[i], 'k.')
                    axes[i].plot(
                        atmb_time,
                        df_conc[s].values[ind_atmb] * conc_factor[i], '.-',
                        color='#0571b0')
                    axes[i].plot(
                        chb_time,
                        df_conc[s].values[ind_chb] * conc_factor[i], '.-',
                        color='#ca0020')
                    axes[i].plot(
                        chc_time,
                        df_conc[s].values[ind_chc] * conc_factor[i], '.-',
                        color='#33a02c')
                    axes[i].plot(
                        cha_time,
                        df_conc[s].values[ind_cha] * conc_factor[i], '.-',
                        color='#ca0020')
                    axes[i].plot(
                        atma_time,
                        df_conc[s].values[ind_atma] * conc_factor[i], '.-',
                        color='#0571b0')
                    # draw baselines
                    axes[i].plot(t_bl_pts, conc_bl_pts[i, :],
                                 'x--', c='gray', linewidth=1.5,
                                 markeredgewidth=1.25)
                    # draw timelag lines
                    axes[i].axvline(x=time_lag_in_day * 86400.,
                                    linestyle='dashed', c='k')
                    # draw fitted lines
                    axes[i].plot(chc_time, conc_fitted_lin[i, :], '-',
                                 c='k', lw=1.5, label='linear')
                    axes[i].plot(chc_time, conc_fitted_rlin[i, :], '--',
                                 c='firebrick', lw=2, label='robust linear')
                    axes[i].plot(chc_time, conc_fitted_nonlin[i, :], '-.',
                                 c='darkblue', lw=2, label='nonlinear')
                    # axis settings
                    axes[i].set_ylabel(species_settings['species_names'][i] +
                                       ' (%s)' % species_unit_names[i])
                    # title setting
                    # for the top panel, add an additional linebreak before it
                    axes[i].set_title(
                        (i == 0) * '\n' +
                        'flux: %.3f (linear), ' % flux_lin[loop_num, i] +
                        '%.3f (robust linear), ' % flux_rlin[loop_num, i] +
                        '%.3f (nonlinear)' % flux_nonlin[loop_num, i])

                # set the common x axis
                t_min = np.floor(
                    np.nanmin(np.append(atmb_time, ch_full_time)) /
                    60. - 0.5) * 60.
                t_max = np.ceil(np.nanmax(ch_full_time) / 60. + 0.5) * 60.
                axes[-1].set_xlim([t_min, t_max])
                axes[-1].set_xticks(np.arange(t_min, t_max + 60., 60.))
                axes[-1].set_xlabel('Time (s)')

                # figure legend
                fig.legend(handles=axes[0].lines[-3:],
                           labels=['linear', 'robust linear', 'nonlinear'],
                           loc='upper right', ncol=3, fontsize=12,
                           handlelength=3,
                           frameon=False, framealpha=0.5)

                # figure annotation
                plt.annotate(ch_label[loop_num],
                             xy=(0.025, 0.985), xycoords='figure fraction',
                             ha='left', va='top', fontsize=12)

                fig.tight_layout()

                run_datetime_str = datetime.datetime.strftime(
                    datetime.timedelta(ch_start[loop_num]) +
                    datetime.datetime(year, 1, 1), '%Y%m%d_%H%M')

                plt.savefig(fitting_plots_path +
                            'chfit_%s.png' % run_datetime_str)

                # important! release the memory after figure is saved
                fig.clf()
                plt.close()
        else:
            pass

    # End of loops. Save data and plots.

    # output data file
    # =========================================================================
    if data_dir['output_filename_prefix'] != '':
        output_fname = output_dir + data_dir['output_filename_prefix'] + \
            '_flux_%s.csv' % run_date_str
    else:
        output_fname = output_dir + 'flux_' + run_date_str + '.csv'

    header = ['doy_utc', 'doy_local', 'ch_no', 'ch_label', 'A_ch', 'V_ch', ]
    for conc_suffix in ['_atmb', '_chb', '_cha', '_atma']:
        header += [s + conc_suffix for s in species_settings['species_list']]
        header += ['sd_' + s + conc_suffix
                   for s in species_settings['species_list']]

    header += [s + '_chc_iqr' for s in species_settings['species_list']]
    for flux_method in ['_lin', '_rlin', '_nonlin']:
        header += ['f' + s + flux_method
                   for s in species_settings['species_list']]
        header += ['se_f' + s + flux_method
                   for s in species_settings['species_list']]

    # biomet variable names
    header = header + ['flow_lpm', 't_turnover', 't_lag_nom',
                       't_lag_optmz', 'status_tlag', 'pres',
                       'T_log', 'T_inst'] + \
        T_atm_names + RH_atm_names + T_ch_names + ['T_dew_ch'] + \
        T_leaf_names + T_soil_names + w_soil_names + PAR_names + PAR_ch_names

    # create output dataframe for concentrations, fluxes and biomet variables
    # no need to define `dtype` since it self-adapts to the assigned columns
    df_flux = pd.DataFrame(index=range(n_smpl_per_day), columns=header)

    # assign columns
    if config['biomet_data_settings']['time_in_UTC']:
        df_flux['doy_utc'] = ch_time
        df_flux['doy_local'] = ch_time + site_parameters['time_zone'] / 24.
    else:
        df_flux['doy_local'] = ch_time
        df_flux['doy_utc'] = ch_time - site_parameters['time_zone'] / 24.

    df_flux['ch_no'] = ch_no
    df_flux['ch_label'] = np.array(ch_label)
    df_flux['A_ch'] = A_ch
    df_flux['V_ch'] = V_ch

    df_flux['pres'] = pres
    df_flux['T_log'] = T_log
    df_flux['T_inst'] = T_inst
    df_flux['flow_lpm'] = flow_lpm
    df_flux['t_turnover'] = t_turnover
    df_flux['T_dew_ch'] = T_dew_ch

    for i in range(len(T_atm_names)):
        df_flux[T_atm_names[i]] = T_atm[:, i]

    for i in range(len(RH_atm_names)):
        df_flux[RH_atm_names[i]] = RH_atm[:, i]

    for i in range(len(T_ch_names)):
        df_flux[T_ch_names[i]] = T_ch[:, i]

    for i in range(len(PAR_names)):
        df_flux[PAR_names[i]] = PAR[:, i]

    for i in range(len(PAR_ch_names)):
        df_flux[PAR_ch_names[i]] = PAR_ch[:, i]

    for i in range(len(T_leaf_names)):
        df_flux[T_leaf_names[i]] = T_leaf[:, i]

    for i in range(len(T_soil_names)):
        df_flux[T_soil_names[i]] = T_soil[:, i]

    for i in range(len(w_soil_names)):
        df_flux[w_soil_names[i]] = w_soil[:, i]

    for spc_id in range(n_species):
        df_flux[species_list[spc_id] + '_atmb'] = conc_atmb[:, spc_id]
        df_flux['sd_%s_atmb' % species_list[spc_id]] = sd_conc_atmb[:, spc_id]
        df_flux[species_list[spc_id] + '_chb'] = conc_chb[:, spc_id]
        df_flux['sd_%s_chb' % species_list[spc_id]] = sd_conc_chb[:, spc_id]
        df_flux[species_list[spc_id] + '_cha'] = conc_cha[:, spc_id]
        df_flux['sd_%s_cha' % species_list[spc_id]] = sd_conc_cha[:, spc_id]
        df_flux[species_list[spc_id] + '_atma'] = conc_atma[:, spc_id]
        df_flux['sd_%s_atma' % species_list[spc_id]] = sd_conc_atma[:, spc_id]
        df_flux[species_list[spc_id] + '_chc_iqr'] = conc_chc_iqr[:, spc_id]

        df_flux['f%s_lin' % species_list[spc_id]] = flux_lin[:, spc_id]
        df_flux['se_f%s_lin' % species_list[spc_id]] = se_flux_lin[:, spc_id]
        df_flux['f%s_rlin' % species_list[spc_id]] = flux_rlin[:, spc_id]
        df_flux['se_f%s_rlin' % species_list[spc_id]] = se_flux_rlin[:, spc_id]
        df_flux['f%s_nonlin' % species_list[spc_id]] = flux_nonlin[:, spc_id]
        df_flux['se_f%s_nonlin' % species_list[spc_id]] = \
            se_flux_nonlin[:, spc_id]

    # rounding off to reduce output file size
    # '%.6f' is the accuracy of single-precision floating numbers
    # do not round off day of year variables or chamber descriptors
    df_flux = df_flux.round(
        {key: 6 for key in df_flux.columns.values
         if key not in ['doy_utc', 'doy_local', 'ch_no',
                        'ch_label', 'A_ch', 'V_ch']})

    df_flux.to_csv(output_fname, sep=',', na_rep='NaN', index=False)
    # no need to have 'row index', therefore, set `index=False`

    print('\nRaw data on the day %s processed.' % run_date_str)

    # output curve fitting diagnostics
    # =========================================================================
    if data_dir['output_filename_prefix'] != '':
        diag_fname = output_dir + data_dir['output_filename_prefix'] + \
            '_diag_%s.csv' % run_date_str
    else:
        diag_fname = output_dir + 'diag_' + run_date_str + '.csv'

    header_diag = ['doy_utc', 'doy_local', 'ch_no', ]
    for s in species_settings['species_list']:
        header_diag += ['k_lin_' + s, 'b_lin_' + s, 'r_lin_' + s,
                        'p_lin_' + s, 'rmse_lin_' + s, 'delta_lin_' + s]

    for s in species_settings['species_list']:
        header_diag += ['k_rlin_' + s, 'b_rlin_' + s,
                        'k_lolim_rlin_' + s, 'k_uplim_rlin_' + s,
                        'rmse_rlin_' + s, 'delta_rlin_' + s]

    for s in species_settings['species_list']:
        header_diag += ['p0_nonlin_' + s, 'p1_nonlin_' + s,
                        'se_p0_nonlin_' + s, 'se_p1_nonlin_' + s,
                        'rmse_nonlin_' + s, 'delta_nonlin_' + s]

    # create output dataframe for fitting diagnostics
    # no need to define `dtype` since it self-adapts to the assigned columns
    df_diag = pd.DataFrame(index=range(n_smpl_per_day), columns=header_diag)

    # assign columns
    if config['biomet_data_settings']['time_in_UTC']:
        df_diag['doy_utc'] = ch_time
        df_diag['doy_local'] = ch_time + site_parameters['time_zone'] / 24.
    else:
        df_diag['doy_local'] = ch_time
        df_diag['doy_utc'] = ch_time - site_parameters['time_zone'] / 24.

    df_diag['ch_no'] = ch_no

    for spc_id in range(n_species):
        df_diag['k_lin_' + species_list[spc_id]] = k_lin[:, spc_id]
        df_diag['b_lin_' + species_list[spc_id]] = b_lin[:, spc_id]
        df_diag['r_lin_' + species_list[spc_id]] = r_lin[:, spc_id]
        df_diag['p_lin_' + species_list[spc_id]] = p_lin[:, spc_id]
        df_diag['rmse_lin_' + species_list[spc_id]] = rmse_lin[:, spc_id]
        df_diag['delta_lin_' + species_list[spc_id]] = delta_lin[:, spc_id]

        df_diag['k_rlin_' + species_list[spc_id]] = k_rlin[:, spc_id]
        df_diag['b_rlin_' + species_list[spc_id]] = b_rlin[:, spc_id]
        df_diag['k_lolim_rlin_' + species_list[spc_id]] = \
            k_lolim_rlin[:, spc_id]
        df_diag['k_uplim_rlin_' + species_list[spc_id]] = \
            k_uplim_rlin[:, spc_id]
        df_diag['rmse_rlin_' + species_list[spc_id]] = rmse_rlin[:, spc_id]
        df_diag['delta_rlin_' + species_list[spc_id]] = delta_rlin[:, spc_id]

        df_diag['p0_nonlin_' + species_list[spc_id]] = p0_nonlin[:, spc_id]
        df_diag['p1_nonlin_' + species_list[spc_id]] = p1_nonlin[:, spc_id]
        df_diag['se_p0_nonlin_' + species_list[spc_id]] = \
            se_p0_nonlin[:, spc_id]
        df_diag['se_p1_nonlin_' + species_list[spc_id]] = \
            se_p1_nonlin[:, spc_id]
        df_diag['rmse_nonlin_' + species_list[spc_id]] = \
            rmse_nonlin[:, spc_id]
        df_diag['delta_nonlin_' + species_list[spc_id]] = \
            delta_nonlin[:, spc_id]

    # rounding off to reduce output file size
    # '%.6f' is the accuracy of single-precision floating numbers
    # do not round off day of year variables or chamber descriptors
    # also, do not round off p-value
    df_diag = df_diag.round(
        {key: 6 for key in df_diag.columns.values
         if key not in ['doy_utc', 'doy_local', 'ch_no'] and 'p_' not in key})

    df_diag.to_csv(diag_fname, sep=',', na_rep='NaN', index=False)
    # no need to have 'row index', therefore, set `index=False`

    print('Data and curve fitting diagnostics written to files.')

    # generate daily summary plots
    if run_options['save_daily_plots']:
        hr_local = (ch_time - np.round(ch_time[0])) * 24.
        if config['biomet_data_settings']['time_in_UTC']:
            tz_str = 'UTC'
        else:
            tz_str = 'UTC+%d' % site_parameters['time_zone']

        for i_ch in np.unique(ch_no):
            fig_daily, axes_daily = plt.subplots(nrows=n_species, sharex=True,
                                                 figsize=(8, 3 * n_species))
            for j in range(n_species):
                axes_daily[j].errorbar(
                    hr_local[ch_no == i_ch], flux_lin[ch_no == i_ch, j],
                    yerr=se_flux_lin[ch_no == i, spc_id] * 2.,
                    c='k', fmt='o', markeredgecolor='None',
                    linestyle='-', lw=1.5, capsize=0, label='linear')
                axes_daily[j].errorbar(
                    hr_local[ch_no == i_ch], flux_rlin[ch_no == i_ch, j],
                    yerr=se_flux_rlin[ch_no == i, spc_id] * 2.,
                    c='firebrick', fmt='o', markeredgecolor='None',
                    linestyle='--', lw=1.5, capsize=0, label='robust linear')
                axes_daily[j].errorbar(
                    hr_local[ch_no == i_ch], flux_nonlin[ch_no == i_ch, j],
                    yerr=se_flux_nonlin[ch_no == i, spc_id] * 2.,
                    c='darkblue', fmt='o', markeredgecolor='None',
                    linestyle='-.', lw=1.5, capsize=0, label='nonlinear')
                # set y labels
                axes_daily[j].set_ylabel(species_settings['species_names'][j] +
                                         ' (%s)' % species_unit_names[j])

            # set common x axis
            axes_daily[-1].set_xlim((0, 24))
            axes_daily[-1].xaxis.set_major_locator(ticker.MultipleLocator(2))
            axes_daily[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
            axes_daily[-1].set_xlabel('Hour (%s)' % tz_str)

            # figure legend
            axes_daily[0].set_title(ch_label[np.where(ch_no == i_ch)[0][0]],
                                    loc='left', fontsize=12)
            fig_daily.legend(handles=axes_daily[0].lines[-3:],
                             labels=['linear', 'robust linear', 'nonlinear'],
                             loc='upper right', ncol=3, fontsize=12,
                             handlelength=3,
                             frameon=False, framealpha=0.5)

            # # figure annotation
            # plt.annotate(ch_label[np.where(ch_no == i_ch)[0][0]],
            #              xy=(0.025, 0.985), xycoords='figure fraction',
            #              ha='left', va='top', fontsize=12)

            fig_daily.tight_layout()
            fig_daily.savefig(
                daily_plots_dir + 'daily_flux_%s_ch%d.png' %
                (run_date_str, i_ch))

            # important! release the memory after figure is saved
            fig_daily.clf()
            plt.close()

        print('Daily flux summary plots generated.')

    return None


def main():
    # Echo program starting
    # =========================================================================
    print('Starting data processing...')
    dt_start = datetime.datetime.now()
    print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
    print('numpy version = %s\n' % np.__version__ +
          'pandas version = %s\n' % pd.__version__ +
          'matplotlib version = %s\n' % mpl.__version__ +
          'Config file is set as `%s`' % args.config)

    # Load config file and data files; extract time as day of year
    # =========================================================================
    if args.config is None:
        config = default_config
    else:
        user_config = load_config(args.config)
        config = default_config
        for key in config:
            if key in user_config:
                config[key].update(user_config[key])

    chamber_config = load_config(
        config['run_options']['chamber_config_filepath'])
    print('Chamber config file is set as `%s`\n' %
          config['run_options']['chamber_config_filepath'])

    # sanity check for config file
    if len(config['species_settings']['species_list']) < 1:
        raise RuntimeError('No gas species is specified in the config.')

    # generate query strings if process only part of all available data
    if config['run_options']['process_recent_period']:
        tz = config['site_parameters']['time_zone']
        # end timestamp
        ts_end = pd.to_datetime(datetime.datetime.utcnow() +
                                datetime.timedelta(seconds=tz * 3600.))
        # start timestamp
        ts_start = ts_end - \
            pd.Timedelta(days=config['run_options']['traceback_in_days'])
        ts_range = pd.date_range(start=ts_start.date(), end=ts_end.date())
        # a list of timestamp query strings
        ts_query = ts_range.strftime('%y%m%d').tolist()
        # use 6-digit yymmdd for now
        # @FIXME: add corresponding options in the `default_config` module,
        # and parse those options here
    else:
        ts_query = None  # assign None to fall back to the default option

    # read biomet data
    df_biomet = load_tabulated_data('biomet', config, query=ts_query)
    # check data size; if no data entry in it, terminate the program
    if df_biomet is None:
        raise RuntimeError('No biomet data file is found.')
    elif df_biomet.shape[0] == 0:
        raise RuntimeError('No entry in the biomet data.')
    # check timestamp existence
    if 'timestamp' not in df_biomet.columns.values:
        raise RuntimeError('No time variable found in the biomet data.')

    # read concentration data
    if config['data_dir']['separate_conc_data']:
        # if concentration data are in their own files, read from files
        df_conc = load_tabulated_data('conc', config, query=ts_query)
        # check data size; if no data entry in it, terminate the program
        if df_conc is None:
            raise RuntimeError('No concentration data file is found.')
        elif df_conc.shape[0] == 0:
            raise RuntimeError('No entry in the concentration data.')
        # check timestamp existence
        if 'timestamp' not in df_conc.columns.values:
            raise RuntimeError(
                'No time variable found in the concentration data.')
    else:
        # if concentration data are not in their own files, create aliases
        # for biomet data and the parsed time variable
        df_conc = df_biomet
        print('Notice: Concentration data are extracted from biomet data, ' +
              'because they are not stored in their own files.')

    # read flow data
    if config['data_dir']['separate_flow_data']:
        # if flow data are in their own files, read from files
        # df_flow = load_flow_data(config)
        df_flow = load_tabulated_data('flow', config, query=ts_query)
        # check data size; if no data entry in it, terminate the program
        if df_flow is None:
            raise RuntimeError('No flow data file is found.')
        elif df_flow.shape[0] == 0:
            raise RuntimeError('No entry in the flow data.')
        # check timestamp existence
        if 'timestamp' not in df_flow.columns.values:
            raise RuntimeError('No time variable found in the flow rate data.')
    else:
        # if flow data are not in their own files, create aliases for
        # biomet data and the parsed time variable
        df_flow = df_biomet
        print('Notice: Flow rate data are extracted from biomet data, ' +
              'because they are not stored in their own files.')

    # read leaf data
    if config['data_dir']['separate_leaf_data']:
        print('Notice: Leaf area data are stored separately ' +
              'from chamber configuration.')
        # if leaf data are in their own files, read from files
        # df_leaf = load_leaf_data(config)
        df_leaf = load_tabulated_data('leaf', config)
        # check data size; if no data entry in it, terminate the program
        if df_leaf is None:
            raise RuntimeError('No leaf area data file is found.')
        elif df_leaf.shape[0] == 0:
            raise RuntimeError('No entry in the leaf area data.')
        # check timestamp existence
        if 'timestamp' not in df_leaf.columns.values:
            raise RuntimeError('No time variable found in the leaf area data.')
    else:
        # if leaf data are not in their own files, set them to None
        df_leaf = None

    year_biomet = df_biomet.loc[0, 'timestamp'].year
    if year_biomet is None:
        year_biomet = config['biomet_data_settings']['year_ref']

    year_conc = df_conc.loc[0, 'timestamp'].year
    if year_conc is None:
        year_conc = config['conc_data_settings']['year_ref']

    if year_biomet != year_conc and config['data_dir']['separate_conc_data']:
        raise RuntimeError('Year numbers do not match between ' +
                           'biomet data and concentration data.')

    # Calculate fluxes, and output plots and the processed data
    # =========================================================================
    print('Calculating fluxes...')

    doy_start = np.floor(df_biomet['time_doy'].min())  # NaN skipped by default
    doy_end = np.ceil(df_biomet['time_doy'].max())  # NaN skipped by default
    year = year_biomet

    # calculate fluxes day by day
    for doy in np.arange(doy_start, doy_end):
        flux_calc(df_biomet, df_conc, df_flow, df_leaf,
                  doy, year, config, chamber_config)

    # Echo program ending
    # =========================================================================
    dt_end = datetime.datetime.now()
    print('\n%s' % datetime.datetime.strftime(dt_end, '%Y-%m-%d %X'))
    print('Done. Finished in %.2f seconds.' %
          (dt_end - dt_start).total_seconds())

    # return 0 if executed properly (a standard in C)
    return 0


if __name__ == '__main__':
    main()
