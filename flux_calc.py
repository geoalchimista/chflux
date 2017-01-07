# -*- coding: utf-8 -*-
# use python 3
"""
Main program for flux calculation
---------------------------------
Created by Wu Sun @ UCLA (18 July 2016).

email: wu.sun@ucla.edu
"""

import numpy as np
import scipy.constants.constants as sci_consts
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import scipy.optimize as optmz
import os, glob, datetime, copy, linecache
import yaml

# os.chdir('/Users/wusun/Dropbox/Projects/models/chflux/')  # for temporary use
# from general_config import *  # deprecated
from general_func import *

import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
# suppress the annoying numpy runtime warning of "mean of empty slice"
warnings.simplefilter('ignore', category=UserWarning)
# suppress the annoying matplotlib tight_layout user warning

plt.rcParams.update({'mathtext.default': 'regular'})  # san-serif math
print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))

# a time constant for converting Mac/Unix seconds
mac_sec_start = datetime.datetime(1904,1,1,0,0)

'''
Load configuration file
-----------------------
'''

with open('./user_config.yaml', 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

run_options = config['run_options']
data_dir = config['data_dir']
biomet_data_settings = config['biomet_data_settings']
conc_data_settings = config['conc_data_settings']
consts = config['constants']
species_settings = config['species_settings']


"""
Reading raw data files
----------------------
"""
biomet_flist = glob.glob(data_dir['biomet_data'])

# check biomet data existence
if not len(biomet_flist):
    print('Cannot find the biometeorological data file! Program is aborted. ')
    exit()
else:
    print(str(len(biomet_flist)) + ' biomet data files were found. Reading in...')

# load all biomet data
biomet_data = np.array([])
for entry in biomet_flist:
    # read biomet data file
    biomet_data_loaded = np.genfromtxt(entry, delimiter=biomet_data_settings['delimiter'],
        skip_header=biomet_data_settings['skip_header'],
        names=biomet_data_settings['names'], usecols=biomet_data_settings['usecols'],
        dtype=biomet_data_settings['dtype'],
        missing_values=biomet_data_settings['missing_values'], filling_values=np.nan)
    # Note: sometimes it may need explicit definitions of data types to avoid a numpy NaN-to-integer error
    if biomet_data.size > 0:
        biomet_data = np.concatenate((biomet_data, biomet_data_loaded))
    else:
        biomet_data = biomet_data_loaded
    del(biomet_data_loaded)

# convert biomet data timestamp to day of year (fractional), if there is not a 'day of year' variable in it
if 'time_doy' in biomet_data.dtype.names:
    # if 'time_doy' is already in the data table, define an alias for it
    biomet_doy = biomet_data['time_doy']
elif 'timestamp' in biomet_data.dtype.names:
    year_start = datetime.datetime.strptime(biomet_data[0]['timestamp'].decode('UTF-8'), '"%Y-%m-%d %X"').year
    biomet_doy = np.zeros(biomet_data[:]['timestamp'].size) * np.nan    # initialize
    for loop_num in range(biomet_data[:]['timestamp'].size):
        biomet_doy[loop_num] = (datetime.datetime.strptime(biomet_data[loop_num]['timestamp'].decode('UTF-8'), '"%Y-%m-%d %X"') - \
        datetime.datetime(year_start, 1, 1, 0, 0)).total_seconds() / 86400.
elif 'time_sec' in biomet_data.dtype.names:
    year_start = (mac_sec_start + datetime.timedelta(seconds=biomet_data[0]['time_sec'])).year
    biomet_doy = np.zeros(biomet_data[:]['time_sec'].size) * np.nan    # initialize
    for loop_num in range(biomet_data[:]['time_sec'].size):
        biomet_doy[loop_num] = (mac_sec_start + datetime.timedelta(seconds=biomet_data[loop_num]['time_sec']) - \
        datetime.datetime(year_start, 1, 1, 0, 0)).total_seconds() / 86400.
elif 'yr' in biomet_data.dtype.names and 'mon' in biomet_data.dtype.names and 'day' in biomet_data.dtype.names:
    year_start = biomet_data[0]['yr']
    biomet_doy = np.zeros(biomet_data[:]['timestamp'].size) * np.nan    # initialize
    for loop_num in range(biomet_data[:]['time_sec'].size):
        biomet_doy[loop_num] = (datetime.datetime(biomet_data[loop_num]['yr'], biomet_data[loop_num]['mon'], biomet_data[loop_num]['day'], 0, 0) -
            datetime.datetime(year_start, 1, 1, 0, 0)).total_seconds() / 86400.
        if 'hr' in biomet_data.dtype.names:
            biomet_doy[loop_num] += biomet_data[loop_num]['hr'] / 24.
        if 'min' in biomet_data.dtype.names:
            biomet_doy[loop_num] += biomet_data[loop_num]['min'] / 1440.
        if 'sec' in biomet_data.dtype.names:
            biomet_doy[loop_num] += biomet_data[loop_num]['sec'] / 86400.
else:
    print('No time variable is found in biomet data table! Program is aborted. ')
    exit()

# echo biomet data status
print(str(biomet_data.size) + ' lines read from biomet data file(s) starting on day ' + str(np.int(biomet_doy[0]) + 1))

# check conc data existence
if data_dir['separate_conc_data']:
    # read concentration data, if they are not in biomet data files
    conc_flist = glob.glob(data_dir['conc_data'])
    if not len(conc_flist):
        print('Cannot find the concentration data file! Program is aborted. ')
        exit()
    else:
        print(str(len(conc_flist)) + ' concentration data files were found. Reading in...')

    # load all conc data
    conc_data = np.array([])
    for entry in conc_flist:
        # read conc data file
        conc_data_loaded = np.genfromtxt(entry, delimiter=conc_data_settings['delimiter'],
            skip_header=conc_data_settings['skip_header'],
            names=conc_data_settings['names'], usecols=conc_data_settings['usecols'],
            dtype=conc_data_settings['dtype'],
            missing_values=conc_data_settings['missing_values'], filling_values=np.nan)
        # Note: sometimes it may need explicit definitions of data types to avoid a numpy NaN-to-integer error
        if conc_data.size > 0:
            conc_data = np.concatenate((conc_data, conc_data_loaded))
        else:
            conc_data = conc_data_loaded
        del(conc_data_loaded)

    # convert conc data timestamp to day of year (fractional), if there is not a 'day of year' variable in it
    if 'time_doy' in conc_data.dtype.names:
        # if 'time_doy' is already in the data table, define an alias for it
        conc_doy = conc_data['time_doy']
    elif 'timestamp' in conc_data.dtype.names:
        year_start = datetime.datetime.strptime(conc_data[0]['timestamp'].decode('UTF-8'), '"%Y-%m-%d %X"').year
        conc_doy = np.zeros(conc_data[:]['timestamp'].size) * np.nan    # initialize
        for loop_num in range(conc_data[:]['timestamp'].size):
            conc_doy[loop_num] = (datetime.datetime.strptime(conc_data[loop_num]['timestamp'].decode('UTF-8'), '"%Y-%m-%d %X"') - \
            datetime.datetime(year_start, 1, 1, 0, 0)).total_seconds() / 86400.
    elif 'time_sec' in conc_data.dtype.names:
        year_start = (mac_sec_start + datetime.timedelta(seconds=conc_data[0]['time_sec'])).year
        conc_doy = np.zeros(conc_data[:]['time_sec'].size) * np.nan    # initialize
        for loop_num in range(conc_data[:]['time_sec'].size):
            conc_doy[loop_num] = (mac_sec_start + datetime.timedelta(seconds=conc_data[loop_num]['time_sec']) - \
            datetime.datetime(year_start, 1, 1, 0, 0)).total_seconds() / 86400.
    elif 'yr' in conc_data.dtype.names and 'mon' in conc_data.dtype.names and 'day' in conc_data.dtype.names:
        year_start = conc_data[0]['yr']
        conc_doy = np.zeros(conc_data[:]['timestamp'].size) * np.nan    # initialize
        for loop_num in range(conc_data[:]['time_sec'].size):
            conc_doy[loop_num] = (datetime.datetime(conc_data[loop_num]['yr'], conc_data[loop_num]['mon'], conc_data[loop_num]['day'], 0, 0) -
                datetime.datetime(year_start, 1, 1, 0, 0)).total_seconds() / 86400.
            if 'hr' in conc_data.dtype.names:
                conc_doy[loop_num] += conc_data[loop_num]['hr'] / 24.
            if 'min' in conc_data.dtype.names:
                conc_doy[loop_num] += conc_data[loop_num]['min'] / 1440.
            if 'sec' in conc_data.dtype.names:
                conc_doy[loop_num] += conc_data[loop_num]['sec'] / 86400.
    else:
        print('No time variable is found in conc data table! Program is aborted. ')
        exit()

    # echo conc data status
    print(str(conc_data.size) + ' lines read from concentration data file(s) starting on day ' + str(np.int(conc_doy[0]) + 1))
else:
    # if concentration data are not in their own files, create aliases for biomet data and parsed time variable
    conc_data = biomet_data
    conc_doy = biomet_doy
    print('No separate concentration data file(s). Therefore, concentration measurements will be parsed from the biomet data file(s). ')


# --- configure the run ---
doy_start = np.nanmin(np.floor(biomet_doy))
doy_end = np.nanmax(np.ceil(biomet_doy))

# create or locate directories for saving plots and processed data
# for processed data tables
output_dir = data_dir['output_dir']
if not os.path.exists(output_dir): os.makedirs(output_dir)
# for daily flux summary plots
if run_options['save_daily_plots']:
    daily_plots_dir = data_dir['plot_dir'] + '/daily_plots/'
    if not os.path.exists(daily_plots_dir): os.makedirs(daily_plots_dir)
# --- end of the configuring ---

"""
Main processing procedure
-------------------------
"""
# loop through days (or only the current day)
for doy in np.arange(doy_start, doy_end):
    # note, 'doy' here is the fractional DOY, which is no larger than integer DOY (Julian day number)
    run_date_str = (datetime.datetime(year_start,1,1) + datetime.timedelta(doy+0.5)).strftime("%Y%m%d")

    # get today's chamber schedule: `ch_start` and `ch_no`
    timer = 0.
    ch_no = np.array([], dtype='int')
    ch_start = np.array([])
    while timer < 1.:
        chlut_now, n_ch, smpl_cycle_len, n_cycle_per_day, next_schedule_switch = chamber_lookup_table_func(doy + timer, return_all=True)
        ch_start = np.append(ch_start, chlut_now['ch_start'].values + doy + timer)
        ch_no = np.append(ch_no, chlut_now['ch_no'].values)
        timer += smpl_cycle_len
        if doy + timer > next_schedule_switch:
            # if the schedule is switched
            ch_no = ch_no[ch_start < next_schedule_switch]
            ch_start = ch_start[ch_start < next_schedule_switch]
            switch_index = ch_no.size
            # apply the switched schedule
            chlut_now, n_ch, smpl_cycle_len, n_cycle_per_day, _, = chamber_lookup_table_func(doy + timer, return_all=True)
            timer = np.floor(timer / smpl_cycle_len - 1) * smpl_cycle_len
            ch_start = np.append(ch_start, chlut_now['ch_start'].values + doy + timer)
            ch_no = np.append(ch_no, chlut_now['ch_no'].values)
            # remove duplicate segment
            ch_start[switch_index:][ch_start[switch_index:] < next_schedule_switch] = np.nan
            ch_no = ch_no[np.isfinite(ch_start)]
            ch_start = ch_start[np.isfinite(ch_start)]
            timer += smpl_cycle_len

    ch_no = ch_no[ch_start < doy + 1.]
    # note: `ch_no` defined above are the nominal chamber numbers
    # it needs to be updated by the actual chamber numbers if such variable is recorded in the biomet data table
    ch_start = ch_start[ch_start < doy + 1.]
    n_smpl_per_day = ch_no.size  # total number of possible samples of the current day
    # times for chamber control actions (e.g. opening and closing) will be calculated within the loop
    ch_o_b = np.zeros(n_smpl_per_day) * np.nan
    ch_cls = np.zeros(n_smpl_per_day) * np.nan
    ch_o_a = np.zeros(n_smpl_per_day) * np.nan
    ch_atm_a = np.zeros(n_smpl_per_day) * np.nan
    ch_end = np.zeros(n_smpl_per_day) * np.nan

    # time, chamber labels, and chamber parameters
    ch_time = np.zeros(n_smpl_per_day) * np.nan  # in day of year (fractional)
    # timestamps for chamber measurements defined as the middle point of the closure period
    ch_label = list()
    A_ch = np.zeros(n_smpl_per_day) * np.nan
    V_ch = np.zeros(n_smpl_per_day) * np.nan

    # conc and flux variables
    n_species = len(species_settings['species_list'])
    if n_species < 1:
        print('Error in gas species settings: no gas species is specified! Program is aborted. ')
        exit()
    species_list = species_settings['species_list']
    conc_factor = [ species_settings[s]['multiplier'] for s in species_list]

    conc_atmb = np.zeros((n_smpl_per_day, n_species)) * np.nan; sd_conc_atmb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    conc_chb = np.zeros((n_smpl_per_day, n_species)) * np.nan; sd_conc_chb = np.zeros((n_smpl_per_day, n_species)) * np.nan
    conc_cha = np.zeros((n_smpl_per_day, n_species)) * np.nan; sd_conc_cha = np.zeros((n_smpl_per_day, n_species)) * np.nan
    conc_atma = np.zeros((n_smpl_per_day, n_species)) * np.nan; sd_conc_atma = np.zeros((n_smpl_per_day, n_species)) * np.nan
    conc_chc_iqr = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # fluxes calculated from: linear fit, robust linear fit, and nonlinear fit
    flux_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan; sd_flux_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    flux_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan; sd_flux_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan
    flux_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan; sd_flux_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan

    # save fitting diagnostics to a separate file
    # linear fitting diagnostics
    k_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan # slopes from linear fit
    b_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # intercepts from linear fit
    r_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # r values from linear fit
    p_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # p values from linear fit
    rmse_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # root mean square error of fitted concentrations
    delta_lin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # fitted changes of concentration during the closure period
    # robust linear fitting diagnostics
    k_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan # (median) slopes from robust linear fit
    b_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # intercepts from robust linear fit
    k_lolim_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # lower bounds of the confidence interval on median slope
    k_uplim_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # upper bounds of the confidence interval on median slope
    rmse_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # root mean square error of fitted concentrations
    delta_rlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # fitted changes of concentration during the closure period
    # nonlinear fit diagnostics
    p0_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # parameter 0: the pre-exponential factor
    p1_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # parameter 1: the small time lag assigned for better fit
    sd_p0_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # std of param 0
    sd_p1_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # std of param 1
    rmse_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # root mean square error of fitted concentrations
    delta_nonlin = np.zeros((n_smpl_per_day, n_species)) * np.nan  # fitted changes of concentration during the closure period

    # biomet variables
    # search for variable names first; but only those with possibly multiple names are searched
    T_atm_names = [s for s in biomet_data_settings['names'] if 'T_atm_' in s or s == 'T_atm']  # T_atm variable names
    RH_atm_names = [s for s in biomet_data_settings['names'] if 'RH_atm_' in s or s == 'RH_atm']  # RH_atm variable names
    T_ch_names = [s for s in biomet_data_settings['names'] if 'T_ch_' in s or s == 'T_ch']  # T_ch variable names
    # T_dew_ch_names = [s for s in biomet_data_settings['names'] if 'T_dew_ch_' in s or s == 'T_dew_ch']  # T_dew_ch variable names
    # dew temperatures probably wouldn't be measured directly in the chamber; they are more likely to be calculated from other variables
    PAR_names = [s for s in biomet_data_settings['names'] if ('PAR_' in s or s == 'PAR') and 'PAR_ch' not in s]  # PAR variable names [not associated with chambers]
    PAR_ch_names = [s for s in biomet_data_settings['names'] if 'PAR_ch_' in s or s == 'PAR_ch']  # PAR variable names [associated with chambers]
    T_leaf_names = [s for s in biomet_data_settings['names'] if 'T_leaf_' in s or s == 'T_leaf']  # T_leaf variable names
    T_soil_names = [s for s in biomet_data_settings['names'] if 'T_soil_' in s or s == 'T_soil']  # T_soil variable names
    w_soil_names = [s for s in biomet_data_settings['names'] if 'w_soil_' in s or s == 'w_soil']  # w_soil variable names
    flow_ch_names = [s for s in biomet_data_settings['names'] if 'flow_ch_' in s or s == 'flow_ch']  # flow_ch variable names

    pres = np.zeros(n_smpl_per_day) * np.nan  # ambient pressure in Pascal
    T_log = np.zeros(n_smpl_per_day) * np.nan   # Datalogger panel temperature (e.g., Campbell Sci. CR1000)
    T_inst = np.zeros(n_smpl_per_day) * np.nan  # Infrared gas analyzer temperature (e.g., LI-COR LI840)

    if len(T_atm_names) > 0:
        T_atm = np.zeros((n_smpl_per_day, len(T_atm_names))) * np.nan  # atmospheric temperature in Celsius degree

    if len(RH_atm_names) > 0:
        RH_atm = np.zeros((n_smpl_per_day, len(RH_atm_names))) * np.nan  # atmospheric relative humidity in percentage

    if len(T_ch_names) > 0:
        T_ch = np.zeros((n_smpl_per_day, len(T_ch_names))) * np.nan  # chamber temperatures in Celsius degree

    if len(PAR_names) > 0:
        PAR = np.zeros((n_smpl_per_day, len(PAR_names))) * np.nan  # PAR [not associated with chambers] in mumol m^-2 s^-1

    if len(PAR_ch_names) > 0:
        PAR_ch = np.zeros((n_smpl_per_day, len(PAR_ch_names))) * np.nan  # PAR [associated with chambers] in mumol m^-2 s^-1

    if len(T_leaf_names) > 0:
        T_leaf = np.zeros((n_smpl_per_day, len(T_leaf_names))) * np.nan  # leaf temperatures in Celsius degree

    if len(T_soil_names) > 0:
        T_soil = np.zeros((n_smpl_per_day, len(T_soil_names))) * np.nan  # soil temperatures in Celsius degree

    if len(w_soil_names) > 0:
        w_soil = np.zeros((n_smpl_per_day, len(w_soil_names))) * np.nan  # soil water content in volumetric fraction (m^3 water m^-3 soil)

    '''
    if len(flow_ch_names) > 0:
        flow_ch = np.zeros((n_smpl_per_day, len(flow_ch_names))) * np.nan  # chamber flow rates, measured
    '''

    flow = np.zeros(n_smpl_per_day) * np.nan   # chamber flowrate, mol s-1
    flow_lpm = np.zeros(n_smpl_per_day) * np.nan  # chamber flowrate, liter per min (convert standard liter per minute to liter per minute if needed)
    V_ch_mol = np.zeros(n_smpl_per_day) * np.nan   # chamber 'volume' converted to moles of air
    t_turnover = np.zeros(n_smpl_per_day) * np.nan   # chamber air turnover time in sec, equals to (V_ch_mol / flow), or (V_ch / flow_lpm)

    T_dew_ch = np.zeros((n_smpl_per_day)) * np.nan  # dew point temp in chambers

    time_lag_nominal = np.zeros(n_smpl_per_day) * np.nan  # save nominal time lag (sec) as a diagnostic
    time_lag_optmz = np.zeros(n_smpl_per_day) * np.nan  # save optimized time lag (sec) as a diagnostic
    status_time_lag_optmz = np.zeros(n_smpl_per_day, dtype='int') - 1  # status code for time lag optimization; initial value -1

    # for fitting plots
    if run_options['save_fitting_plots']:
        fitting_plots_path = data_dir['plot_dir'] + '/fitting/' + run_date_str + '/'
        if not os.path.exists(fitting_plots_path): os.makedirs(fitting_plots_path)

    """ Loops for averaging biomet variables and calculating fluxes. """
    for loop_num in range(n_smpl_per_day):
        # get the current chamber's meta info
        chlut_current = chamber_lookup_table_func(ch_start[loop_num])
        chlut_current = chlut_current[ chlut_current['ch_no'] == ch_no[loop_num] ]
        A_ch[loop_num] = chlut_current['A_ch'].values[0]
        V_ch[loop_num] = chlut_current['V_ch'].values[0]
        ch_label.append( chlut_current['ch_label'].values[0] ) # 'ch_label' is a list! not an array

        ch_o_b[loop_num] = ch_start[loop_num] + chlut_current['ch_o_b'].values[0]
        ch_cls[loop_num] = ch_start[loop_num] + chlut_current['ch_cls'].values[0]
        ch_o_a[loop_num] = ch_start[loop_num] + chlut_current['ch_o_a'].values[0]
        ch_atm_a[loop_num] = ch_start[loop_num] + chlut_current['ch_atm_a'].values[0]
        ch_end[loop_num] = ch_start[loop_num] + chlut_current['ch_end'].values[0]

        ch_time[loop_num] = 0.5 * (ch_cls[loop_num] + ch_o_a[loop_num])

        # extract indices for averaging biomet variables, no time lag
        ind_ch_biomet = np.where( (biomet_doy >= ch_start[loop_num]) & (biomet_doy < ch_end[loop_num]) )  # include the full chamber period
        n_ind_ch_biomet= ind_ch_biomet[0].size

        # variables averaged in the whole chamber period
        if n_ind_ch_biomet > 0:
            if 'pres' in biomet_data.dtype.names:
                pres[loop_num] = np.nanmean(biomet_data[ind_ch_biomet]['pres'])    # ambient pressure in Pascal
            else:
                if consts['site_pressure'] is None:
                    pres[loop_num] = consts['p_std']  # use standard atmospheric pressure if no site pressure is defined
                else:
                    pres[loop_num] = consts['site_pressure']  # use defined site pressure

            if 'T_log' in biomet_data.dtype.names:
                T_log[loop_num] = np.nanmean(biomet_data[ind_ch_biomet]['T_log'])    # datalogger panel temp

            if 'T_inst' in biomet_data.dtype.names:
                T_inst[loop_num] = np.nanmean(biomet_data[ind_ch_biomet]['T_inst'])    # instrument temperature

            if len(T_atm_names) > 0:
                for i in range( len(T_atm_names) ):
                    T_atm[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ T_atm_names[i] ])  # atmospheric temperature(s)

            if len(RH_atm_names) > 0:
                for i in range( len(RH_atm_names) ):
                    RH_atm[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ RH_atm_names[i] ])  # atmospheric RH(s)

            if len(T_ch_names) > 0:
                for i in range( len(T_ch_names) ):
                    T_ch[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ T_ch_names[i] ])  # chamber temperature(s)

            if len(PAR_names) > 0:
                for i in range( len(PAR_names) ):
                    PAR[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ PAR_names[i] ])  # PAR (not associated with chambers)

            if len(PAR_ch_names) > 0:
                for i in range( len(PAR_ch_names) ):
                    PAR_ch[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ PAR_ch_names[i] ])  # PAR (not associated with chambers)

            if len(T_leaf_names) > 0:
                for i in range( len(T_leaf_names) ):
                    T_leaf[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ T_leaf_names[i] ])  # leaf temperature(s)

            if len(T_soil_names) > 0:
                for i in range( len(T_soil_names) ):
                    T_soil[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ T_soil_names[i] ])  # soil temperatures in Celsius degree

            if len(w_soil_names) > 0:
                for i in range( len(w_soil_names) ):
                    w_soil[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ w_soil_names[i] ])  # soil water content in volumetric fraction (m^3 water m^-3 soil)

            '''
            if len(flow_ch_names) > 0:
                for i in range( len(flow_ch_names) ):
                    flow_ch[loop_num, i] = np.nanmean(biomet_data[ind_ch_biomet][ flow_ch_names[i] ])
                    # chamber flow rates in standard liter per minute, measured
            '''

        # flow rate is only needed for the chamber currently being measured
        if len(flow_ch_names) > 0:
            # find the column location to extract the flow rate of the current chamber
            flow_loc = [k for k,s in enumerate(flow_ch_names) if 'ch_' + str(ch_no[loop_num]) in s]
            if len(flow_loc) > 0:
                # flow_lpm[loop_num] = flow_ch[loop_num, flow_loc[0]]
                flow_lpm[loop_num] = np.nanmean(biomet_data[ind_ch_biomet][ flow_ch_names[flow_loc[0]] ])
                # convert standard liter per minute to liter per minute, if applicable
                if biomet_data_settings['flow_rate_in_STP']:
                    flow_lpm[loop_num] = flow_lpm[loop_num] * ( T_ch[ loop_num, ch_no[loop_num]-1 ] + consts['T_0'] ) / consts['T_0'] * consts['p_std'] / pres[loop_num]
        else:
            # call the user-defined flow rate function if there is no flow rate data in the biomet data table
            # `chamber_flow_rates` function is imported from `general_func.py`
            flow_lpm[loop_num], is_flow_STP = chamber_flow_rates(ch_time[loop_num], ch_no[loop_num])
            if is_flow_STP:
                flow_lpm[loop_num] = flow_lpm[loop_num] * ( T_ch[ loop_num, ch_no[loop_num]-1 ] + consts['T_0'] ) / consts['T_0'] * consts['p_std'] / pres[loop_num]

        # convert volumetric flow to mass flow (mol s^-1)
        flow[loop_num] = flow_lpm[loop_num] * 1e-3 / 60. * pres[loop_num] / consts['R_gas'] / ( T_ch[ loop_num, ch_no[loop_num]-1 ] + consts['T_0'] )  # mol s^-1
        ## print(flow[loop_num]) # for test only

        # convert chamber volume to mol
        V_ch_mol[loop_num] = V_ch[loop_num] * pres[loop_num] / consts['R_gas'] / ( T_ch[ loop_num, ch_no[loop_num]-1 ] + consts['T_0'] )
        ## print(V_ch_mol[loop_num]) # for test only
        t_turnover[loop_num] = V_ch_mol[loop_num] / flow[loop_num]  # turnover time in seconds, will be useful in flux calculation

        '''
        Indices for flux calculation
        * '_chb' - before closure
        * '_chc' - during closure
        * '_chs' - starting 1 min of the closure  # disabled
        * '_cha' - after closure
        '''
        # these need to be assigned properly later
        dt_lmargin = 0.
        dt_rmargin = 0.
        time_lag_in_day = 0.

        ind_ch_full = np.where((conc_doy > ch_o_b[loop_num]) & (conc_doy < ch_end[loop_num] + time_lag_in_day))  # 'ind_ch_full' index is only used for plotting
        ind_atmb = np.where((conc_doy > ch_start[loop_num] + time_lag_in_day + dt_lmargin) & (conc_doy < ch_o_b[loop_num] + time_lag_in_day - dt_rmargin))
        ind_chb = np.where((conc_doy > ch_o_b[loop_num] + time_lag_in_day + dt_lmargin) & (conc_doy < ch_cls[loop_num] + time_lag_in_day - dt_rmargin))
        ind_chc = np.where((conc_doy > ch_cls[loop_num] + time_lag_in_day + dt_lmargin) & (conc_doy < ch_o_a[loop_num] + time_lag_in_day - dt_rmargin))
        # Note: after the line is switched, regardless of the time lag, the analyzer will sample the next line.
        # This is the reason why a time lag is not added to the terminal time.
        ind_cha = np.where((conc_doy > ch_o_a[loop_num] + time_lag_in_day + dt_lmargin) & (conc_doy < ch_atm_a[loop_num]))
        ind_atma = np.where((conc_doy > ch_atm_a[loop_num] + time_lag_in_day + dt_lmargin) & (conc_doy < ch_end[loop_num]))

        n_ind_atmb = ind_atmb[0].size
        n_ind_chb = ind_chb[0].size
        n_ind_chc = ind_chc[0].size
        n_ind_cha = ind_cha[0].size
        n_ind_atma = ind_atma[0].size

        # check if there are enough data points for calculating fluxes
        # note the conc data might not be sampled every second, so the judgment statement needs to be modified
        if n_ind_chc >= 2. * 60.:    # need at least 2 min data in the closure period to do flux calculation
            flag_calc_flux = 1    # 0 - dont calc fluxes; 1 - calc fluxes
        else:
            flag_calc_flux = 0

        # avg conc's for output
        for spc_id in range(n_species):
            conc_atmb[loop_num, spc_id] = np.nanmean(conc_data[species_list[spc_id]][ind_atmb] * conc_factor[spc_id])
            sd_conc_atmb[loop_num, spc_id] = np.nanstd(conc_data[species_list[spc_id]][ind_atmb] * conc_factor[spc_id], ddof=1)

            conc_chb[loop_num, spc_id] = np.nanmean(conc_data[species_list[spc_id]][ind_chb] * conc_factor[spc_id])
            sd_conc_chb[loop_num, spc_id] = np.nanstd(conc_data[species_list[spc_id]][ind_chb] * conc_factor[spc_id], ddof=1)

            conc_cha[loop_num, spc_id] = np.nanmean(conc_data[species_list[spc_id]][ind_cha] * conc_factor[spc_id])
            sd_conc_cha[loop_num, spc_id] = np.nanstd(conc_data[species_list[spc_id]][ind_cha] * conc_factor[spc_id], ddof=1)

            conc_atma[loop_num, spc_id] = np.nanmean(conc_data[species_list[spc_id]][ind_atma] * conc_factor[spc_id])
            sd_conc_atma[loop_num, spc_id] = np.nanstd(conc_data[species_list[spc_id]][ind_atma] * conc_factor[spc_id], ddof=1)

            conc_chc_iqr[loop_num, spc_id] = IQR_func(conc_data[species_list[spc_id]][ind_chc] * conc_factor[spc_id])


        # if the species 'h2o' exist, calculate chamber dew temperature
        spc_id_h2o = np.where( np.array(species_list) == 'h2o' )[0]
        if conc_chb[loop_num, spc_id_h2o] > 0:
        	T_dew_ch[loop_num] = dew_temp( conc_chb[loop_num, spc_id_h2o] * species_settings['h2o']['output_unit'] * pres[loop_num] )
        else:
        	T_dew_ch[loop_num] = np.nan

        # fitted conc and baselines, saved for plotting purposes
        # only need two points to draw a line for each species
        conc_bl_pts = np.zeros((n_species, 2))
        t_bl_pts = np.zeros(2)

        conc_bl = np.zeros((n_species, n_ind_chc))  # fitted baselines
        conc_fitted_lin = np.zeros((n_species, n_ind_chc))  # fitted concentrations during closure
        conc_fitted_rlin = np.zeros((n_species, n_ind_chc))  # fitted concentrations during closure
        conc_fitted_nonlin = np.zeros((n_species, n_ind_chc))  # fitted concentrations during closure

        if flag_calc_flux:
            # loop through each species
            for spc_id in range(n_species):
                # extract closure segments and convert the DOY to seconds after 'ch_start' time for fitting
                ch_full_time = (conc_doy[ind_ch_full] - ch_start[loop_num]) * 86400.  # for plotting only
                chb_time = (conc_doy[ind_chb] - ch_start[loop_num]) * 86400.
                cha_time = (conc_doy[ind_cha] - ch_start[loop_num]) * 86400.
                chc_time = (conc_doy[ind_chc] - ch_start[loop_num]) * 86400.

                # conc of current species defined with 'spc_id'
                chc_conc = conc_data[ind_chc][ species_list[spc_id] ] * conc_factor[spc_id]

                # calculate baselines' slopes and intercepts
                # baseline end points changed from mean to medians (05/05/2016)
                median_chb_time = np.nanmedian(conc_doy[ind_chb] - ch_start[loop_num]) * 86400. # median time for LC open (before), sec
                median_cha_time = np.nanmedian(conc_doy[ind_cha] - ch_start[loop_num]) * 86400. # median time for LC open (after)
                median_chb_conc = np.nanmedian(conc_data[ind_chb][ species_list[spc_id] ]) * conc_factor[spc_id]
                median_cha_conc = np.nanmedian(conc_data[ind_cha][ species_list[spc_id] ]) * conc_factor[spc_id]
                # if `median_cha_conc` is not a finite value, set it equal to `median_chb_conc`. Thus `k_bl` will be zero.
                if np.isnan(median_cha_conc):
                    median_cha_conc = median_chb_conc
                    median_cha_time = chc_time[-1]
                k_bl = (median_cha_conc - median_chb_conc) / (median_cha_time - median_chb_time)
                b_bl = median_chb_conc - k_bl * median_chb_time

                # subtract the baseline to correct for instrument drift (assumed linear)
                conc_bl = k_bl * chc_time + b_bl

                # linear fit, see the supplementary info of Sun et al. (2016) JGR-Biogeosci
                y_fit = (chc_conc - conc_bl) * flow[loop_num] / A_ch[loop_num]
                x_fit = np.exp(- (chc_time - chc_time[0] + (time_lag_in_day+dt_lmargin)*8.64e4) / t_turnover[loop_num] )

                slope, intercept, r_value, p_value, sd_slope = stats.linregress(x_fit, y_fit)
                # save the fitted conc values
                conc_fitted_lin[spc_id, :] = (slope * x_fit + intercept) * A_ch[loop_num] / flow[loop_num] + conc_bl
                # save the fitting diagnostics
                k_lin[loop_num, spc_id] = slope
                b_lin[loop_num, spc_id] = intercept
                r_lin[loop_num, spc_id] = r_value
                p_lin[loop_num, spc_id] = p_value
                rmse_lin[loop_num, spc_id] = np.sqrt(np.nanmean((conc_fitted_lin[spc_id, :] - chc_conc) ** 2))
                delta_lin[loop_num, spc_id] = (conc_fitted_lin[spc_id, -1] - conc_bl[-1]) - (conc_fitted_lin[spc_id, 0] - conc_bl[0])
                # saving fluxes calculated from linear fit
                flux_lin[loop_num, spc_id] = - slope
                sd_flux_lin[loop_num, spc_id] = np.abs(sd_slope)
                # clear temporary fitted parameters
                del(slope, intercept, r_value, p_value, sd_slope)

                # robust linear fit
                medslope, medintercept, lo_slope, up_slope = stats.theilslopes(y_fit, x_fit, alpha=0.95)
                # save the fitted conc values
                conc_fitted_rlin[spc_id, :] = (medslope * x_fit + medintercept) * A_ch[loop_num] / flow[loop_num] + conc_bl
                # save the fitting diagnostics
                k_rlin[loop_num, spc_id] = medslope
                b_rlin[loop_num, spc_id] = medintercept
                k_lolim_rlin[loop_num, spc_id] = lo_slope
                k_uplim_rlin[loop_num, spc_id] = up_slope
                rmse_rlin[loop_num, spc_id] = np.sqrt(np.nanmean((conc_fitted_rlin[spc_id, :] - chc_conc) ** 2))
                delta_rlin[loop_num, spc_id] = (conc_fitted_rlin[spc_id, -1] - conc_bl[-1]) - (conc_fitted_rlin[spc_id, 0] - conc_bl[0])
                # saving fluxes calculated from robust linear fit
                flux_rlin[loop_num, spc_id] = - medslope
                sd_flux_rlin[loop_num, spc_id] = np.abs(up_slope - lo_slope) / 3.92  # 0.95 CI is equivalent to +/- 1.96 sigma
                # clear temporary fitted parameters
                del(medslope, medintercept, lo_slope, up_slope)

                # nonlinear fit
                t_fit = (chc_time - chc_time[0] + (time_lag_in_day+dt_lmargin)*8.64e4) / t_turnover[loop_num]
                params_nonlin_guess = [ -flux_lin[loop_num, spc_id], 0. ]
                params_nonlin = optmz.least_squares(resid_conc_func, params_nonlin_guess,
                    bounds=( [-np.inf, -10. / t_turnover[loop_num] ], [np.inf, 10. / t_turnover[loop_num] ] ),
                    loss='soft_l1', f_scale=0.5, args=(t_fit, y_fit))
                # save the fitted conc values
                conc_fitted_nonlin[spc_id, :] = conc_func(params_nonlin.x, t_fit) * A_ch[loop_num] / flow[loop_num] + conc_bl

                # save the fitting diagnostics
                p0_nonlin[loop_num, spc_id] = params_nonlin.x[0]
                p1_nonlin[loop_num, spc_id] = params_nonlin.x[1]
                sd_p0_nonlin[loop_num, spc_id] = np.nan  # use NaN as placeholder for now
                sd_p1_nonlin[loop_num, spc_id] = np.nan  # use NaN as placeholder for now
                rmse_nonlin[loop_num, spc_id] = np.sqrt(np.nanmean((conc_fitted_nonlin[spc_id, :] - chc_conc) ** 2))
                delta_nonlin[loop_num, spc_id] = (conc_fitted_nonlin[spc_id, -1] - conc_bl[-1]) - (conc_fitted_nonlin[spc_id, 0] - conc_bl[0])
                # saving fluxes calculated from nonlinear fit
                flux_nonlin[loop_num, spc_id] = params_nonlin.x[0]
                sd_flux_nonlin[loop_num, spc_id] = np.nan  # use NaN as placeholder for now
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

    """End of loops. Save data and plots. """
    # output data file
    if data_dir['output_filename_str'] != '':
        output_fname = output_dir + data_dir['output_filename_str'] + '_flux_' + run_date_str + '.csv'
    else:
        output_fname = output_dir + 'flux_' + run_date_str + '.csv'

    header = ['doy_utc', 'doy_local', 'ch_no', 'ch_label', 'A_ch', 'V_ch', ]
    for conc_suffix in ['_atmb', '_chb', '_cha', '_atma']:
        header += [s + conc_suffix for s in species_settings['species_list']]
        header += ['sd_' + s + conc_suffix for s in species_settings['species_list']]

    header += [s + '_chc_iqr' for s in species_settings['species_list']]
    for flux_method in ['_lin', '_rlin', '_nonlin']:
        header += ['f' + s + flux_method for s in species_settings['species_list']]
        header += ['sd_f' + s + flux_method for s in species_settings['species_list']]

    # biomet variable names
    header = header + ['flow_lpm', 't_turnover', 't_lag_nom', 't_lag_optmz', 'status_tlag', 'pres', 'T_log', 'T_inst'] + \
        T_atm_names + RH_atm_names + T_ch_names + ['T_dew_ch'] + T_leaf_names + T_soil_names + w_soil_names + PAR_names + PAR_ch_names

    # create output dataframe for concentrations, fluxes and biomet variables
    df_flux = pd.DataFrame(index=range(n_smpl_per_day), columns=header)  # do not define `dtype` yet; it will self-adapt based on the assigned columns
    # assign columns
    if biomet_data_settings['time_in_UTC']:
        df_flux['doy_utc'] = ch_time
        df_flux['doy_local'] = ch_time + consts['time_zone'] / 24.
    else:
        df_flux['doy_local'] = ch_time
        df_flux['doy_utc'] = ch_time - consts['time_zone'] / 24.
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

    for i in range( len(T_atm_names) ):
        df_flux[ T_atm_names[i] ] = T_atm[:,i]

    for i in range( len(RH_atm_names) ):
        df_flux[ RH_atm_names[i] ] = RH_atm[:,i]

    for i in range( len(T_ch_names) ):
        df_flux[ T_ch_names[i] ] = T_ch[:,i]

    for i in range( len(PAR_names) ):
        df_flux[ PAR_names[i] ] = PAR[:,i]

    for i in range( len(PAR_ch_names) ):
        df_flux[ PAR_ch_names[i] ] = PAR_ch[:,i]

    for i in range( len(T_leaf_names) ):
        df_flux[ T_leaf_names[i] ] = T_leaf[:,i]

    for i in range( len(T_soil_names) ):
        df_flux[ T_soil_names[i] ] = T_soil[:,i]

    for i in range( len(w_soil_names) ):
        df_flux[ w_soil_names[i] ] = w_soil[:,i]

    for spc_id in range(n_species):
        df_flux[ species_list[spc_id] + '_atmb' ] = conc_atmb[:, spc_id]
        df_flux[ 'sd_' + species_list[spc_id] + '_atmb' ] = sd_conc_atmb[:, spc_id]
        df_flux[ species_list[spc_id] + '_chb' ] = conc_chb[:, spc_id]
        df_flux[ 'sd_' + species_list[spc_id] + '_chb' ] = sd_conc_chb[:, spc_id]
        df_flux[ species_list[spc_id] + '_cha' ] = conc_cha[:, spc_id]
        df_flux[ 'sd_' + species_list[spc_id] + '_cha' ] = sd_conc_cha[:, spc_id]
        df_flux[ species_list[spc_id] + '_atma' ] = conc_atma[:, spc_id]
        df_flux[ 'sd_' + species_list[spc_id] + '_atma' ] = sd_conc_atma[:, spc_id]
        df_flux[ species_list[spc_id] + '_chc_iqr' ] = conc_chc_iqr[:, spc_id]

        df_flux[ 'f'+ species_list[spc_id] + '_lin' ] = flux_lin[:, spc_id]
        df_flux[ 'sd_f'+ species_list[spc_id] + '_lin' ] = sd_flux_lin[:, spc_id]
        df_flux[ 'f'+ species_list[spc_id] + '_rlin' ] = flux_rlin[:, spc_id]
        df_flux[ 'sd_f'+ species_list[spc_id] + '_rlin' ] = sd_flux_rlin[:, spc_id]
        df_flux[ 'f'+ species_list[spc_id] + '_nonlin' ] = flux_nonlin[:, spc_id]
        df_flux[ 'sd_f'+ species_list[spc_id] + '_nonlin' ] = sd_flux_nonlin[:, spc_id]

    df_flux.to_csv(output_fname, sep=',', na_rep='NaN', index=False)  # do not output 'row name'
    print('Raw data on the day ' + run_date_str + ' processed. ')

    # output diagnostics
    if data_dir['output_filename_str'] != '':
        diag_fname = output_dir + data_dir['output_filename_str'] + '_diag_' + run_date_str + '.csv'
    else:
        diag_fname = output_dir + 'diag_' + run_date_str + '.csv'

    header_diag = ['doy_utc', 'doy_local', 'ch_no', ]
    for s in species_settings['species_list']:
        header_diag += ['k_lin_' + s, 'b_lin_' + s, 'r_lin_' + s, 'p_lin_' + s, 'rmse_lin_' + s, 'delta_lin_' + s]

    for s in species_settings['species_list']:
        header_diag += ['k_rlin_' + s, 'b_rlin_' + s, 'k_lolim_rlin_' + s, 'k_uplim_rlin_' + s, 'rmse_rlin_' + s, 'delta_rlin_' + s]

    for s in species_settings['species_list']:
        header_diag += ['p0_nonlin_' + s, 'p1_nonlin_' + s, 'sd_p0_nonlin_' + s, 'sd_p1_nonlin_' + s, 'rmse_nonlin_' + s, 'delta_nonlin_' + s]

    # create output dataframe for diagnostics
    df_diag = pd.DataFrame(index=range(n_smpl_per_day), columns=header_diag)  # do not define `dtype` yet; it will self-adapt based on the assigned columns
    # assign columns
    if biomet_data_settings['time_in_UTC']:
        df_diag['doy_utc'] = ch_time
        df_diag['doy_local'] = ch_time + consts['time_zone'] / 24.
    else:
        df_diag['doy_local'] = ch_time
        df_diag['doy_utc'] = ch_time - consts['time_zone'] / 24.
    df_diag['ch_no'] = ch_no

    for spc_id in range(n_species):
        df_diag[ 'k_lin_'+ species_list[spc_id] ] = k_lin[:, spc_id]
        df_diag[ 'b_lin_'+ species_list[spc_id] ] = b_lin[:, spc_id]
        df_diag[ 'r_lin_'+ species_list[spc_id] ] = r_lin[:, spc_id]
        df_diag[ 'p_lin_'+ species_list[spc_id] ] = p_lin[:, spc_id]
        df_diag[ 'rmse_lin_'+ species_list[spc_id] ] = rmse_lin[:, spc_id]
        df_diag[ 'delta_lin_'+ species_list[spc_id] ] = delta_lin[:, spc_id]

        df_diag[ 'k_rlin_'+ species_list[spc_id] ] = k_rlin[:, spc_id]
        df_diag[ 'b_rlin_'+ species_list[spc_id] ] = b_rlin[:, spc_id]
        df_diag[ 'k_lolim_rlin_'+ species_list[spc_id] ] = k_lolim_rlin[:, spc_id]
        df_diag[ 'k_uplim_rlin_'+ species_list[spc_id] ] = k_uplim_rlin[:, spc_id]
        df_diag[ 'rmse_rlin_'+ species_list[spc_id] ] = rmse_rlin[:, spc_id]
        df_diag[ 'delta_rlin_'+ species_list[spc_id] ] = delta_rlin[:, spc_id]

        df_diag[ 'p0_nonlin_'+ species_list[spc_id] ] = p0_nonlin[:, spc_id]
        df_diag[ 'p1_nonlin_'+ species_list[spc_id] ] = p1_nonlin[:, spc_id]
        df_diag[ 'sd_p0_nonlin_'+ species_list[spc_id] ] = sd_p0_nonlin[:, spc_id]
        df_diag[ 'sd_p1_nonlin_'+ species_list[spc_id] ] = sd_p1_nonlin[:, spc_id]
        df_diag[ 'rmse_nonlin_'+ species_list[spc_id] ] = rmse_nonlin[:, spc_id]
        df_diag[ 'delta_nonlin_'+ species_list[spc_id] ] = delta_nonlin[:, spc_id]

    df_diag.to_csv(diag_fname, sep=',', na_rep='NaN', index=False)  # do not output 'row name'
    print('Curve fitting diagnostics for flux calculation on the day ' + run_date_str + ' written to file. ')

# --- end of the main procedure ---

print('Done.')
print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
