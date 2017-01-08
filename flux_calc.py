#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program for flux calculation

Revision history
----------------
- Created by Wu Sun @ UCLA <wu.sun "at" ucla.edu>. (18 July 2016)
- Restructured. (W.S., 6 Jan 2016)
- Reformatted to comply with PEP8 standard. (W.S., 7 Jan 2016)
"""
import os
import glob
import datetime
import argparse
import warnings
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, optimize

from general_func import *

# deprecated imports
# -----------------------------------------------------------------------------
# import sys
# import copy
# import linecache
# import scipy.constants.constants as sci_consts
# os.chdir('/Users/wusun/Dropbox/Projects/models/chflux/')  # for temporary use
# from general_config import *  # deprecated
# warnings.simplefilter('ignore', category=UserWarning)
# suppress the annoying matplotlib tight_layout user warning


# Command-line argument parser
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyChamberFlux: Main program for flux calculation.')
parser.add_argument('-c', '--config', dest='config',
                    action='store', help='set the config file')

args = parser.parse_args()

if args.config is None:
    args.config = 'config.yaml'


# Global settings (not from the config file)
# =============================================================================
plt.rcParams.update({'mathtext.default': 'regular'})  # sans-serif math

# suppress the annoying numpy runtime warning of "mean of empty slice"
warnings.simplefilter('ignore', category=RuntimeWarning)


def load_config(filepath):
    """Load configuration file from a given filepath."""
    with open(filepath, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc_yaml:
            print(exc_yaml)

    return config


def load_biomet_data(config, query=None):
    """
    Read biometeorological data.

    Parameters
    ----------
    config : dict
        Configuration dictionary parsed from the YAML config file.
    query : str
        The query string used to search in all available data files.
        If `None` (default), read all data files.

    Returns
    -------
    df_biomet : pandas.DataFrame
        The loaded biometeorological data.
    """
    # unpack config
    data_dir = config['data_dir']
    biomet_data_settings = config['biomet_data_settings']

    # search for data files
    biomet_flist = glob.glob(data_dir['biomet_data'])

    # check biomet data existence
    if not len(biomet_flist):
        print('Cannot find the biomet data file!')
        return None
    else:
        print('%d biomet data files are found. ' % len(biomet_flist) +
              'Loading...')

    if query is not None:
        biomet_flist = [f for f in biomet_flist if query in f]

    # load all biomet data
    df_biomet = None
    for entry in biomet_flist:
        print(entry)
        df_biomet_loaded = pd.read_csv(
            entry, delimiter=biomet_data_settings['delimiter'],
            header=biomet_data_settings['header'],
            names=biomet_data_settings['names'],
            usecols=biomet_data_settings['usecols'],
            dtype=biomet_data_settings['dtype'],
            na_values=biomet_data_settings['na_values'],
            engine='c', encoding='utf-8')
        # Note: sometimes it may need explicit definitions of data types to
        # avoid a numpy NaN-to-integer error

        if df_biomet is None:
            df_biomet = df_biomet_loaded
        else:
            df_biomet = pd.concat([df_biomet, df_biomet_loaded],
                                  ignore_index=True)

        del(df_biomet_loaded)

    # echo biomet data status
    print('%d lines read from biomet data.' % df_biomet.shape[0])

    return df_biomet


def load_conc_data(config, query=None):
    """
    Read concentration data.

    Parameters
    ----------
    config : dict
        Configuration dictionary parsed from the YAML config file.
    query : str
        The query string used to search in all available data files.
        If `None` (default), read all data files.

    Returns
    -------
    df_conc : pandas.DataFrame
        The loaded concentration data.
    """
    # unpack config
    data_dir = config['data_dir']
    conc_data_settings = config['conc_data_settings']

    # search for data files
    conc_flist = glob.glob(data_dir['conc_data'])

    # check conc data existence
    if not len(conc_flist):
        print('Cannot find the concentration data file!')
        return None
    else:
        print('%d concentration data files are found. ' % len(conc_flist) +
              'Loading...')

    if query is not None:
        conc_flist = [f for f in conc_flist if query in f]

    # load all conc data
    df_conc = None
    for entry in conc_flist:
        print(entry)
        df_conc_loaded = pd.read_csv(
            entry, delimiter=conc_data_settings['delimiter'],
            header=conc_data_settings['header'],
            names=conc_data_settings['names'],
            usecols=conc_data_settings['usecols'],
            dtype=conc_data_settings['dtype'],
            na_values=conc_data_settings['na_values'],
            engine='c', encoding='utf-8')
        # Note: sometimes it may need explicit definitions of data types to
        # avoid a numpy NaN-to-integer error

        if df_conc is None:
            df_conc = df_conc_loaded
        else:
            df_conc = pd.concat([df_conc, df_conc_loaded],
                                ignore_index=True)

        del(df_conc_loaded)

    # echo conc data status
    print('%d lines read from concentration data.' % df_conc.shape[0])

    return df_conc


def timestamp_to_doy(df, timestamp_format=None, time_sec_start=None):
    """
    Convert timestamp in a dataframe to day of year.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe from which timestamp is extracted.
    timestamp_format, optional : str
        Datetime format string. Default is '%Y-%m-%d %X'.
    time_sec_start, optional : int
        The reference year of the time in seconds. Default is 1904.

    Returns
    -------
    doy: array_like
        Day of year number (float); minimum value is 0 (Jan 1 midnight), not 1.
    """
    # set default timestamp format and `time_sec` starting year, if not given
    if timestamp_format is None:
        timestamp_format = '%Y-%m-%d %X'
    if time_sec_start is None:
        time_sec_start = 1904

    # reference datetime for the `time_sec` variable
    time_sec_ref_dt = datetime.datetime(time_sec_start, 1, 1)

    doy = np.zeros(df.shape[0]) * np.nan  # initialize

    if 'time_doy' in df.columns.values:
        # if 'time_doy' is already in the dataframe, define an alias for it
        doy = df['time_doy'].values
    elif 'timestamp' in df.columns.values:
        year_start = datetime.datetime.strptime(
            df.loc[0, 'timestamp'], timestamp_format).year
        for i in range(df.shape[0]):
            doy[i] = (
                datetime.datetime.strptime(
                    df.loc[i, 'timestamp'], timestamp_format) -
                datetime.datetime(year_start, 1, 1)).total_seconds() / 86400.
    elif 'time_sec' in df.columns.values:
        year_start = (time_sec_ref_dt +
                      datetime.timedelta(seconds=df.loc[0, 'time_sec'])).year
        for i in range(df.shape[0]):
            doy[i] = (
                time_sec_ref_dt +
                datetime.timedelta(seconds=df.loc[i, 'time_sec']) -
                datetime.datetime(year_start, 1, 1)).total_seconds() / 86400.
    elif ('yr' in df.columns.values and 'mon' in df.columns.values and
          'day' in df.columns.values):
        year_start = df.loc[0, 'yr']
        for i in range(df.shape[0]):
            doy[i] = (
                datetime.datetime(df.loc[i, 'yr'], df.loc[i, 'mon'],
                                  df.loc[i, 'day']) -
                datetime.datetime(year_start, 1, 1)).total_seconds() / 86400.
            if 'hr' in df.columns.values:
                doy[i] += df.loc[i, 'hr'] / 24.
            if 'min' in df.columns.values:
                doy[i] += df.loc[i, 'min'] / 1440.
            if 'sec' in df.columns.values:
                doy[i] += df.loc[i, 'sec'] / 86400.
    else:
        print('No time variable is found!')
        return None

    return doy


def check_starting_year(df, timestamp_format=None, time_sec_start=None):
    # set default timestamp format and `time_sec` starting year, if not given
    if timestamp_format is None:
        timestamp_format = '%Y-%m-%d %X'
    if time_sec_start is None:
        time_sec_start = 1904

    # get the starting year number
    if 'yr' in df.columns.values:
        year_start = df.loc[0, 'yr']
    elif 'timestamp' in df.columns.values:
        year_start = datetime.datetime.strptime(
            df.loc[0, 'timestamp'], timestamp_format).year
    elif 'time_sec' in df.columns.values:
        time_sec_ref_dt = datetime.datetime(time_sec_start, 1, 1)
        year_start = (time_sec_ref_dt +
                      datetime.timedelta(seconds=df.loc[0, 'time_sec'])).year
    else:
        year_start = None

    return year_start


def flux_calc(df_biomet, doy_biomet, df_conc, doy_conc, doy, year, config):
    """
    Calculate fluxes and generate plots.

    Parameters
    ----------
    df_biomet : pandas.DataFrame
        The biometeorological data.
    doy_biomet : 1D numpy.ndarray
        Day of year variable for the biometeorological data.
    df_conc : pandas.DataFrame
        The concentration data.
    doy_conc : 1D numpy.ndarray
        Day of year variable for the concentration data.
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
    # unpack config
    run_options = config['run_options']
    data_dir = config['data_dir']
    biomet_data_settings = config['biomet_data_settings']
    conc_data_settings = config['conc_data_settings']
    consts = config['constants']
    site_parameters = config['site_parameters']
    species_settings = config['species_settings']

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

    # get today's chamber schedule: `ch_start` and `ch_no`
    timer = 0.
    ch_no = np.array([], dtype='int')
    ch_start = np.array([])
    while timer < 1.:
        chlut_now, n_ch, smpl_cycle_len, n_cycle_per_day, next_schedule_switch = \
            chamber_lookup_table_func(doy + timer, return_all=True)
        ch_start = np.append(ch_start,
                             chlut_now['ch_start'].values + doy + timer)
        ch_no = np.append(ch_no, chlut_now['ch_no'].values)
        timer += smpl_cycle_len
        if doy + timer > next_schedule_switch:
            # if the schedule is switched
            ch_no = ch_no[ch_start < next_schedule_switch]
            ch_start = ch_start[ch_start < next_schedule_switch]
            switch_index = ch_no.size
            # apply the switched schedule
            chlut_now, n_ch, smpl_cycle_len, n_cycle_per_day, _, = \
                chamber_lookup_table_func(doy + timer, return_all=True)
            timer = np.floor(timer / smpl_cycle_len - 1) * smpl_cycle_len
            ch_start = np.append(ch_start,
                                 chlut_now['ch_start'].values + doy + timer)
            ch_no = np.append(ch_no, chlut_now['ch_no'].values)
            # remove duplicate segment
            ch_start[switch_index:][ch_start[switch_index:] < next_schedule_switch] = np.nan
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

    # times for chamber control actions (e.g., opening and closing)
    ch_o_b = np.zeros(n_smpl_per_day) * np.nan
    ch_cls = np.zeros(n_smpl_per_day) * np.nan
    ch_o_a = np.zeros(n_smpl_per_day) * np.nan
    ch_atm_a = np.zeros(n_smpl_per_day) * np.nan
    ch_end = np.zeros(n_smpl_per_day) * np.nan

    # time, chamber labels, and chamber parameters
    ch_time = np.zeros(n_smpl_per_day) * np.nan  # in day of year (fractional)
    # timestamps for chamber measurements defined as
    # the middle point of the closure period
    ch_label = list()
    A_ch = np.zeros(n_smpl_per_day) * np.nan
    V_ch = np.zeros(n_smpl_per_day) * np.nan

    # conc and flux variables
    n_species = len(species_settings['species_list'])
    species_list = species_settings['species_list']
    conc_factor = [species_settings[s]['multiplier'] for s in species_list]

    conc_atmb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_conc_atmb = np.zeros((n_smpl_per_day, n_species)) * np.nan

    conc_chb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_conc_chb = np.zeros((n_smpl_per_day, n_species)) * np.nan

    conc_cha = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_conc_cha = np.zeros((n_smpl_per_day, n_species)) * np.nan

    conc_atma = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_conc_atma = np.zeros((n_smpl_per_day, n_species)) * np.nan

    conc_chc_iqr = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # fluxes calculated from: linear fit, robust linear fit, and nonlinear fit
    flux_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_flux_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    flux_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_flux_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    flux_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_flux_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

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
    # - `sd_p0_nonlin`: standard error of parameter 0
    # - `sd_p1_nonlin`: standard error of parameter 1
    # root mean square error of fitted concentrations
    # - `delta_nonlin`: fitted C_end - C_init, i.e.,
    #   fitted changes of concentration during the closure period
    p0_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    p1_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_p0_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    sd_p1_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    rmse_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    delta_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # search for biomet variable names
    # --------------------------------
    # but only those with possibly multiple names are searched
    # - `T_atm_names`: T_atm variable names
    # - `RH_atm_names`: RH_atm variable names
    # - `T_ch_names`: T_ch variable names
    # - `T_dew_ch_names` (*deprecated*): T_dew_ch variable names
    #   dew temperatures probably wouldn't be measured directly in the chamber,
    #   they are more likely to be calculated from other variables
    # - `PAR_names`: PAR variable names (not associated with chambers)
    # - `PAR_ch_names`: PAR variable names [associated with chambers]
    # - `T_leaf_names`: T_leaf variable names
    # - `T_soil_names`: T_soil variable names
    # - `w_soil_names`: w_soil variable names
    # - `flow_ch_names`: flow_ch variable names
    T_atm_names = [s for s in biomet_data_settings['names']
                   if 'T_atm_' in s or s == 'T_atm']
    RH_atm_names = [s for s in biomet_data_settings['names']
                    if 'RH_atm_' in s or s == 'RH_atm']
    T_ch_names = [s for s in biomet_data_settings['names']
                  if 'T_ch_' in s or s == 'T_ch']
    # T_dew_ch_names = [s for s in biomet_data_settings['names']
    #                   if 'T_dew_ch_' in s or s == 'T_dew_ch']
    PAR_names = [s for s in biomet_data_settings['names']
                 if ('PAR_' in s or s == 'PAR') and 'PAR_ch' not in s]
    PAR_ch_names = [s for s in biomet_data_settings['names']
                    if 'PAR_ch_' in s or s == 'PAR_ch']
    T_leaf_names = [s for s in biomet_data_settings['names']
                    if 'T_leaf_' in s or s == 'T_leaf']
    T_soil_names = [s for s in biomet_data_settings['names']
                    if 'T_soil_' in s or s == 'T_soil']
    w_soil_names = [s for s in biomet_data_settings['names']
                    if 'w_soil_' in s or s == 'w_soil']
    flow_ch_names = [s for s in biomet_data_settings['names']
                     if 'flow_ch_' in s or s == 'flow_ch']

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

    T_dew_ch = np.zeros((n_smpl_per_day)) * np.nan
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

    # create the directory for fitting plots, if not already exists
    if run_options['save_fitting_plots']:
        fitting_plots_path = data_dir['plot_dir'] + \
            '/fitting/%s/' % run_date_str
        if not os.path.exists(fitting_plots_path):
            os.makedirs(fitting_plots_path)

    # loops for averaging biomet variables and calculating fluxes
    for loop_num in range(n_smpl_per_day):
        # get the current chamber's meta info
        chlut_current = chamber_lookup_table_func(ch_start[loop_num])
        chlut_current = chlut_current[chlut_current['ch_no'] == ch_no[loop_num]]
        A_ch[loop_num] = chlut_current['A_ch'].values[0]
        V_ch[loop_num] = chlut_current['V_ch'].values[0]
        ch_label.append(chlut_current['ch_label'].values[0])
        # Note: 'ch_label' is a list! not an array

        ch_o_b[loop_num] = ch_start[loop_num] + chlut_current['ch_o_b'].values[0]
        ch_cls[loop_num] = ch_start[loop_num] + chlut_current['ch_cls'].values[0]
        ch_o_a[loop_num] = ch_start[loop_num] + chlut_current['ch_o_a'].values[0]
        ch_atm_a[loop_num] = ch_start[loop_num] + chlut_current['ch_atm_a'].values[0]
        ch_end[loop_num] = ch_start[loop_num] + chlut_current['ch_end'].values[0]

        ch_time[loop_num] = 0.5 * (ch_cls[loop_num] + ch_o_a[loop_num])

        # extract indices for averaging biomet variables, no time lag
        ind_ch_biomet = np.where((doy_biomet >= ch_start[loop_num]) &
                                 (doy_biomet < ch_end[loop_num]))[0]
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
                    pres[loop_num] = site_parameters['p_std']
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

            '''
            # chamber flow rates in standard liter per minute, measured
            if len(flow_ch_names) > 0:
                for i in range(len(flow_ch_names)):
                    flow_ch[loop_num, i] = np.nanmean(
                        df_biomet.loc[ind_ch_biomet, flow_ch_names[i]].values)
            '''

        # flow rate is only needed for the chamber currently being measured
        if len(flow_ch_names) > 0:
            # find the column location to extract the flow rate of the current chamber
            flow_loc = [k for k, s in enumerate(flow_ch_names)
                        if 'ch_' + str(ch_no[loop_num]) in s]
            if len(flow_loc) > 0:
                # flow_lpm[loop_num] = flow_ch[loop_num, flow_loc[0]]
                flow_lpm[loop_num] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, flow_ch_names[flow_loc[0]]].values)
                # convert standard liter per minute to liter per minute, if applicable
                if biomet_data_settings['flow_rate_in_STP']:
                    flow_lpm[loop_num] = flow_lpm[loop_num] * \
                        (T_ch[loop_num, ch_no[loop_num] - 1] + consts['T_0']) / \
                        consts['T_0'] * consts['p_std'] / pres[loop_num]
        else:
            # call the user-defined flow rate function if there is no flow rate data in the biomet data table
            # `chamber_flow_rates` function is imported from `general_func.py`
            flow_lpm[loop_num], is_flow_STP = chamber_flow_rates(ch_time[loop_num], ch_no[loop_num])
            if is_flow_STP:
                flow_lpm[loop_num] = flow_lpm[loop_num] * \
                    (T_ch[loop_num, ch_no[loop_num] - 1] + consts['T_0']) / \
                    consts['T_0'] * consts['p_std'] / pres[loop_num]

        # convert volumetric flow to mass flow (mol s^-1)
        flow[loop_num] = flow_lpm[loop_num] * 1e-3 / 60. * \
            pres[loop_num] / consts['R_gas'] / \
            (T_ch[loop_num, ch_no[loop_num] - 1] + consts['T_0'])
        # # print(flow[loop_num])  # for test only

        # convert chamber volume to mol
        V_ch_mol[loop_num] = V_ch[loop_num] * pres[loop_num] / \
            consts['R_gas'] / \
            (T_ch[loop_num, ch_no[loop_num] - 1] + consts['T_0'])
        # # print(V_ch_mol[loop_num])  # for test only

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
                               (doy_conc < ch_end[loop_num] + time_lag_in_day))[0]

        ind_atmb = np.where((doy_conc > ch_start[loop_num] + time_lag_in_day + dt_lmargin) &
                            (doy_conc < ch_o_b[loop_num] + time_lag_in_day - dt_rmargin))[0]
        ind_chb = np.where((doy_conc > ch_o_b[loop_num] + time_lag_in_day + dt_lmargin) &
                           (doy_conc < ch_cls[loop_num] + time_lag_in_day - dt_rmargin))[0]
        ind_chc = np.where((doy_conc > ch_cls[loop_num] + time_lag_in_day + dt_lmargin) &
                           (doy_conc < ch_o_a[loop_num] + time_lag_in_day - dt_rmargin))[0]
        # Note: after the line is switched, regardless of the time lag, the analyzer will sample the next line.
        # This is the reason why a time lag is not added to the terminal time.
        ind_cha = np.where((doy_conc > ch_o_a[loop_num] + time_lag_in_day + dt_lmargin) &
                           (doy_conc < ch_atm_a[loop_num]))[0]
        ind_atma = np.where((doy_conc > ch_atm_a[loop_num] + time_lag_in_day + dt_lmargin) &
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
        if n_ind_chc >= 2. * 60.:
            # needs at least 2 min good data in the closure period to proceed
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
        if conc_chb[loop_num, spc_id_h2o] > 0:
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
                ch_full_time = (doy_conc[ind_ch_full] - ch_start[loop_num]) * 86400.
                chb_time = (doy_conc[ind_chb] - ch_start[loop_num]) * 86400.
                cha_time = (doy_conc[ind_cha] - ch_start[loop_num]) * 86400.
                chc_time = (doy_conc[ind_chc] - ch_start[loop_num]) * 86400.

                # conc of current species defined with 'spc_id'
                chc_conc = df_conc.loc[ind_chc, species_list[spc_id]].values * \
                    conc_factor[spc_id]

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
                x_fit = np.exp(
                    - (chc_time - chc_time[0] + (time_lag_in_day + dt_lmargin) * 8.64e4) / t_turnover[loop_num])

                slope, intercept, r_value, p_value, se_slope = \
                    stats.linregress(x_fit, y_fit)

                # save the fitted conc values
                conc_fitted_lin[spc_id, :] = (slope * x_fit + intercept) * \
                    A_ch[loop_num] / flow[loop_num] + conc_bl

                # save the linear fit results and diagnostics
                flux_lin[loop_num, spc_id] = - slope
                sd_flux_lin[loop_num, spc_id] = np.abs(se_slope)
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
                del(slope, intercept, r_value, p_value, se_slope)

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
                sd_flux_rlin[loop_num, spc_id] = \
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
                del(medslope, medintercept, lo_slope, up_slope)

                # nonlinear fit
                # -------------------------------------------------------------
                t_fit = (chc_time - chc_time[0] + (time_lag_in_day + dt_lmargin) * 8.64e4) / t_turnover[loop_num]
                params_nonlin_guess = [- flux_lin[loop_num, spc_id], 0.]
                params_nonlin = optimize.least_squares(
                    resid_conc_func, params_nonlin_guess,
                    bounds=([-np.inf, -10. / t_turnover[loop_num]],
                            [np.inf, 10. / t_turnover[loop_num]]),
                    loss='soft_l1', f_scale=0.5, args=(t_fit, y_fit))

                # save the fitted conc values
                conc_fitted_nonlin[spc_id, :] = \
                    conc_func(params_nonlin.x, t_fit) * A_ch[loop_num] / \
                    flow[loop_num] + conc_bl

                # save the robust linear fit results and diagnostics
                flux_nonlin[loop_num, spc_id] = params_nonlin.x[0]
                sd_flux_nonlin[loop_num, spc_id] = np.nan
                # use NaN as placeholder for now
                p0_nonlin[loop_num, spc_id] = params_nonlin.x[0]
                p1_nonlin[loop_num, spc_id] = params_nonlin.x[1]
                sd_p0_nonlin[loop_num, spc_id] = np.nan
                # use NaN as placeholder for now
                sd_p1_nonlin[loop_num, spc_id] = np.nan
                # use NaN as placeholder for now
                rmse_nonlin[loop_num, spc_id] = np.sqrt(
                    np.nanmean((conc_fitted_nonlin[spc_id, :] - chc_conc) ** 2))
                delta_nonlin[loop_num, spc_id] = \
                    (conc_fitted_nonlin[spc_id, -1] - conc_bl[-1]) - \
                    (conc_fitted_nonlin[spc_id, 0] - conc_bl[0])

                # clear temporary fitted parameters
                del(params_nonlin_guess, params_nonlin)

                # save the baseline conc's
                # baseline end points changed from mean to medians
                conc_bl_pts[spc_id, :] = median_chb_conc, median_cha_conc

            # used for plotting the baseline
            t_bl_pts[:] = median_chb_time, median_cha_time
        else:
            flux_lin[loop_num, :] = np.nan
            sd_flux_lin[loop_num, :] = np.nan

    # End of loops. Save data and plots.

    # output data file
    # =========================================================================
    if data_dir['output_filename_str'] != '':
        output_fname = output_dir + data_dir['output_filename_str'] + \
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
        header += ['sd_f' + s + flux_method
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
    if biomet_data_settings['time_in_UTC']:
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
        df_flux['sd_f%s_lin' % species_list[spc_id]] = sd_flux_lin[:, spc_id]
        df_flux['f%s_rlin' % species_list[spc_id]] = flux_rlin[:, spc_id]
        df_flux['sd_f%s_rlin' % species_list[spc_id]] = sd_flux_rlin[:, spc_id]
        df_flux['f%s_nonlin' % species_list[spc_id]] = flux_nonlin[:, spc_id]
        df_flux['sd_f%s_nonlin' % species_list[spc_id]] = \
            sd_flux_nonlin[:, spc_id]

    df_flux.to_csv(output_fname, sep=',', na_rep='NaN', index=False)
    # no need to have 'row index', therefore, set `index=False`

    print('Raw data on the day %s processed.' % run_date_str)

    # output curve fitting diagnostics
    # =========================================================================
    if data_dir['output_filename_str'] != '':
        diag_fname = output_dir + data_dir['output_filename_str'] + \
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
                        'sd_p0_nonlin_' + s, 'sd_p1_nonlin_' + s,
                        'rmse_nonlin_' + s, 'delta_nonlin_' + s]

    # create output dataframe for fitting diagnostics
    # no need to define `dtype` since it self-adapts to the assigned columns
    df_diag = pd.DataFrame(index=range(n_smpl_per_day), columns=header_diag)

    # assign columns
    if biomet_data_settings['time_in_UTC']:
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
        df_diag['sd_p0_nonlin_' + species_list[spc_id]] = \
            sd_p0_nonlin[:, spc_id]
        df_diag['sd_p1_nonlin_' + species_list[spc_id]] = \
            sd_p1_nonlin[:, spc_id]
        df_diag['rmse_nonlin_' + species_list[spc_id]] = \
            rmse_nonlin[:, spc_id]
        df_diag['delta_nonlin_' + species_list[spc_id]] = \
            delta_nonlin[:, spc_id]

    df_diag.to_csv(diag_fname, sep=',', na_rep='NaN', index=False)
    # no need to have 'row index', therefore, set `index=False`

    print('Processed data and curve fitting diagnostics written to files.')

    return None


def main():
    # Echo program starting
    # =========================================================================
    print('Starting data processing...')
    dt_start = datetime.datetime.now()
    print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
    print('Config file is set as', args.config)
    print('numpy version = ' + np.__version__)
    print('pandas version = ' + pd.__version__)

    # Load config file and data files; extract time as day of year
    # =========================================================================
    config = load_config(args.config)
    # sanity check for config file
    if len(config['species_settings']['species_list']) < 1:
        print('Program is aborted: no gas species is specified in the config.')
        exit(1)

    # read biomet data
    df_biomet = load_biomet_data(config)
    # check data size; if no data entry in it, terminate the program
    if df_biomet is None:
        print('Program is aborted: no biomet data file is found.')
        exit(1)
    elif df_biomet.shape[0] == 0:
        print('Program is aborted: no entry in the biomet data.')
        exit(1)

    # parse time variable
    print('Parsing time variable in the biomet data...')
    doy_biomet = timestamp_to_doy(
        df_biomet,
        timestamp_format=config['biomet_data_settings']['timestamp_format'],
        time_sec_start=config['biomet_data_settings']['time_sec_start'])
    # check if the conversion to day of year is successful or not
    if doy_biomet is None:
        print('Program is aborted: no time variable found in the biomet data.')
        exit(1)
    else:
        print('Time variable parsed successfully.')

    # read concentration data
    if config['data_dir']['separate_conc_data']:
        # if concentration data are in their own files, read from files
        df_conc = load_conc_data(config)
        # check data size; if no data entry in it, terminate the program
        if df_conc is None:
            print('Program is aborted: no concentration data file is found.')
            exit(1)
        elif df_conc.shape[0] == 0:
            print('Program is aborted: no entry in the concentration data.')
            exit(1)

        # parse time variable
        print('Parsing time variable in the concentration data...')
        doy_conc = timestamp_to_doy(
            df_conc,
            timestamp_format=config['conc_data_settings']['timestamp_format'],
            time_sec_start=config['conc_data_settings']['time_sec_start'])
        # check if the conversion to day of year is successful or not
        if doy_conc is None:
            print('Program is aborted: ' +
                  'No time variable found in the concentration data.')
            exit(1)
        else:
            print('Time variable parsed successfully.')
    else:
        # if concentration data are not in their own files, create aliases
        # for biomet data and the parsed time variable
        df_conc = df_biomet
        doy_conc = doy_biomet
        print('Notice: Concentration data are extracted from biomet data, ' +
              'because they are not stored in their own files.')

    # check starting years of biomet data and conc data
    # this is to make sure the converted day of year variables are referenced
    # to the same staring year
    year_biomet = check_starting_year(
        df_biomet,
        timestamp_format=config['biomet_data_settings']['timestamp_format'],
        time_sec_start=config['biomet_data_settings']['time_sec_start'])
    year_conc = check_starting_year(
        df_conc,
        timestamp_format=config['conc_data_settings']['timestamp_format'],
        time_sec_start=config['conc_data_settings']['time_sec_start'])
    if year_biomet != year_conc and config['data_dir']['separate_conc_data']:
        print('Program is aborted: Year numbers do not match between ' +
              'biomet data and concentration data.')
        exit(1)

    # Calculate fluxes, and output plots and the processed data
    # =========================================================================
    print('Calculating fluxes...')

    doy_start = np.nanmin(np.floor(doy_biomet))
    doy_end = np.nanmax(np.ceil(doy_biomet))
    year = year_biomet

    # calculate fluxes day by day
    for doy in np.arange(doy_start, doy_end):
        flux_calc(df_biomet, doy_biomet, df_conc, doy_conc, doy, year, config)

    # Echo program ending
    # =========================================================================
    dt_end = datetime.datetime.now()
    print(datetime.datetime.strftime(dt_end, '%Y-%m-%d %X'))
    print('Done. Finished in %.2f seconds.' %
          (dt_end - dt_start).total_seconds())

    return None


if __name__ == '__main__':
    main()
