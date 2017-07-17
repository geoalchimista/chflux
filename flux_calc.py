"""
Main program for flux calculation

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

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

from chflux.common import *
from chflux.default_config import default_config
from chflux.datetools import extract_date_substr
from chflux.iotools import *
from chflux.helpers import *


# Command-line argument parser
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyChamberFlux: Main program for flux calculation.')
parser.add_argument('-c', '--config', dest='config',
                    action='store', help='set the config file')

args = parser.parse_args()


# Global settings (not from the config file)
# =============================================================================
if LooseVersion(mpl.__version__) < LooseVersion('2.0.0'):
    # enforce sans-serif math for matplotlib version before 2.0.0
    plt.rcParams.update({'mathtext.default': 'regular'})

# suppress numpy runtime warning when dealing with NaN containing arrays
# # warnings.simplefilter('ignore', category=RuntimeWarning)  # suppresses all
warnings_to_ignore = [
    'Mean of empty slice',
    'Degrees of freedom <= 0 for slice.',
    'divide by zero encountered in true_divide',
    'invalid value encountered in true_divide']
for msg in warnings_to_ignore:
    warnings.filterwarnings('ignore', msg)


def flux_calc(df_biomet, df_conc, df_flow, df_leaf, df_timelag,
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
    site_parameters = config['site_parameters']
    species_settings = config['species_settings']

    # extract species settings
    # @TODO: this may be moved to `main()` to reduce unnecessary steps
    n_species = len(species_settings['species_list'])
    species_list = species_settings['species_list']
    conc_factor = [species_settings[s]['multiplier'] for s in species_list]

    output_unit_list = [species_settings[s]['output_unit']
                        for s in species_list]
    conc_unit_names, flux_unit_names = convert_unit_names(output_unit_list)

    species_for_timelag_optmz = \
        config['run_options']['timelag_optimization_species']
    if species_for_timelag_optmz not in species_list:
        spc_optmz_id = 0
    else:
        spc_optmz_id = species_list.index(species_for_timelag_optmz)

    # create or locate directories for output
    # for output data
    # keep this in the `flux_calc()` function for safety
    output_dir = data_dir['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if (run_options['save_fitting_diagnostics'] and
            not os.path.exists(output_dir + '/diag')):
        os.makedirs(output_dir + '/diag')
    # for daily flux summary plots
    if run_options['save_daily_plots']:
        daily_plots_dir = data_dir['plot_dir'] + '/daily_plots/'
        if not os.path.exists(daily_plots_dir):
            os.makedirs(daily_plots_dir)
    # save config if enabled
    if run_options['save_config']:
        if not os.path.exists(output_dir + '/config'):
            os.makedirs(output_dir + '/config')
        usercfg_filename = output_dir + '/config/user_config.yaml'
        chcfg_filename = output_dir + '/config/chamber.yaml'
        with open(usercfg_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False,
                      allow_unicode=True, indent=4)
            print('Configuration file saved to %s' % usercfg_filename)
        with open(run_options['chamber_config_filepath'], 'r') as fsrc, \
                open(chcfg_filename, 'w') as fdest:
            fdest.write(fsrc.read())
            print('Chamber setting file saved to %s' % chcfg_filename)

    # a date string for current run; used in echo and in output file names
    run_date_str = (datetime.datetime(year, 1, 1) +
                    datetime.timedelta(doy + 0.5)).strftime('%Y%m%d')

    # for curve-fitting plots of chamber headspace concentrations
    if run_options['save_fitting_plots']:
        fitting_plots_path = data_dir['plot_dir'] + \
            '/fitting/%s/' % run_date_str
        if not os.path.exists(fitting_plots_path):
            os.makedirs(fitting_plots_path)

    # # get timelag optimization settings
    # if run_options['timelag_method'] == 'optimized':
    #     flag_optimize_timelag = True
    # else:
    #     flag_optimize_timelag = False

    # unpack time variables
    # @TODO: switch from day of year based subsetting to timestamp subsetting?
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
            chlut = chamber_lookup_table_func(doy + timer, chamber_config)
            df_chlut = chlut.df
            # n_ch = chlut.n_ch
            smpl_cycle_len = chlut.smpl_cycle_len
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

    # quality flags
    qc_flags = np.zeros((n_smpl_per_day, n_species), dtype='int') - 1

    # number of valid observations of concentrations in the closure period
    n_obs = np.zeros((n_smpl_per_day, n_species), dtype='int')

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

    # a temporary helper variable
    biomet_var_list = \
        T_atm_names + RH_atm_names + T_ch_names + ['T_dew_ch'] + \
        T_leaf_names + T_soil_names + w_soil_names + PAR_names + PAR_ch_names

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
    # - `timelag_nominal`: nominal time lag in seconds
    # - `timelag_optmz`: optimized time lag in seconds
    # - `status_timelag_optmz`: status code for time lag optimization
    #   initial value is -1
    timelag_nominal = np.zeros(n_smpl_per_day) * np.nan
    timelag_optmz = np.zeros(n_smpl_per_day) * np.nan
    status_timelag_optmz = np.zeros(n_smpl_per_day, dtype='int') - 1

    # @DEBUG: debug print for time optimization
    # print(datetime.datetime.now())

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

        # get sensor numbers to search in the biomet data table
        TC_no = np.int(df_chlut_current['TC_no'].values[0])
        PAR_no = np.int(df_chlut_current['PAR_no'].values[0])
        flowmeter_no = np.int(df_chlut_current['flowmeter_no'].values[0])

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
                # use 'axis=0' to average by column
                T_atm[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, T_atm_names].values, axis=0)

            # atmospheric RH
            if len(RH_atm_names) > 0:
                RH_atm[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, RH_atm_names].values, axis=0)

            # chamber temperatures
            if len(T_ch_names) > 0:
                T_ch[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, T_ch_names].values, axis=0)

            # PAR (not associated with chambers)
            if len(PAR_names) > 0:
                PAR[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, PAR_names].values, axis=0)

            # PAR associated with chambers
            if len(PAR_ch_names) > 0:
                PAR_ch[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, PAR_ch_names].values, axis=0)

            # leaf temperatures
            if len(T_leaf_names) > 0:
                T_leaf[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, T_leaf_names].values, axis=0)

            # soil temperatures
            if len(T_soil_names) > 0:
                T_soil[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, T_soil_names].values, axis=0)

            # soil water content in volumetric fraction (m^3 water m^-3 soil)
            if len(w_soil_names) > 0:
                w_soil[loop_num, :] = np.nanmean(
                    df_biomet.loc[ind_ch_biomet, w_soil_names].values, axis=0)

        # extract indices for averaging flow rates, no time lag
        ind_ch_flow = np.where((doy_flow >= ch_start[loop_num]) &
                               (doy_flow < ch_end[loop_num]))[0]
        # include the full chamber period
        n_ind_ch_flow = ind_ch_flow.size

        # flow rate is only needed for the chamber currently being measured
        if len(flow_ch_names) > 0:
            # find the column location to extract the flow rate of the current
            # chamber
            # flow_loc = [k for k, s in enumerate(flow_ch_names)
            #             if 'ch_' + str(ch_no[loop_num]) in s]
            flow_loc = [k for k, s in enumerate(flow_ch_names)
                        if 'ch_%d' % flowmeter_no in s]
            if len(flow_loc) > 0:
                flow_lpm[loop_num] = \
                    np.nanmean(df_flow.loc[ind_ch_flow,
                                           flow_ch_names[flow_loc[0]]].values)
                # convert standard liter per minute to liter per minute, if
                # applicable
                if config['flow_data_settings']['flow_rate_in_STP']:
                    # flow_lpm[loop_num] *= \
                    #     (T_ch[loop_num, ch_no[loop_num] - 1] +
                    #         phys_const['T_0']) / \
                    #     phys_const['T_0'] * \
                    #     phys_const['p_std'] / pres[loop_num]
                    flow_lpm[loop_num] *= \
                        (T_ch[loop_num, TC_no - 1] +
                            phys_const['T_0']) / \
                        phys_const['T_0'] * \
                        phys_const['p_std'] / pres[loop_num]

        # convert volumetric flow to mass flow (mol s^-1)
        # flow[loop_num] = flow_lpm[loop_num] * 1e-3 / 60. * \
        #     pres[loop_num] / phys_const['R_gas'] / \
        #     (T_ch[loop_num, ch_no[loop_num] - 1] + phys_const['T_0'])
        flow[loop_num] = flow_lpm[loop_num] * 1e-3 / 60. * \
            pres[loop_num] / phys_const['R_gas'] / \
            (T_ch[loop_num, TC_no - 1] + phys_const['T_0'])

        # convert chamber volume to mol
        # V_ch_mol[loop_num] = V_ch[loop_num] * pres[loop_num] / \
        #     phys_const['R_gas'] / \
        #     (T_ch[loop_num, ch_no[loop_num] - 1] + phys_const['T_0'])
        V_ch_mol[loop_num] = V_ch[loop_num] * pres[loop_num] / \
            phys_const['R_gas'] / \
            (T_ch[loop_num, TC_no - 1] + phys_const['T_0'])

        t_turnover[loop_num] = V_ch_mol[loop_num] / flow[loop_num]
        # turnover time in seconds, useful in flux calculation

        # indices for flux calculation
        # - '_chb' - before closure
        # - '_chc' - during closure
        # - '_chs' - starting 1 min of the closure  # disabled
        # - '_cha' - after closure

        # timelag optimization (still in active development & testing)
        dt_lmargin = 0.
        dt_rmargin = 0.
        if (df_chlut_current['optimize_timelag'].values[0] and
                run_options['timelag_method'] == 'optimized'):
            timelag_nominal[loop_num] = \
                df_chlut_current['timelag_nominal'].values[0] * 86400.
            timelag_upper_limit = \
                df_chlut_current['timelag_upper_limit'].values[0] * 86400.
            # a temporary variable
            timelag_lower_limit = \
                df_chlut_current['timelag_lower_limit'].values[0] * 86400.
            # a temporary variable

            # print('DEBUG: ', timelag_lower_limit, timelag_upper_limit)
            ind_optmz = np.where(
                (doy_conc > ch_o_b[loop_num]) &
                (doy_conc < ch_end[loop_num] +
                    timelag_upper_limit / 86400.))[0]
            time_optmz = (doy_conc[ind_optmz] - ch_start[loop_num]) * 86400.
            conc_optmz = \
                df_conc.loc[ind_optmz, species_list[spc_optmz_id]].values * \
                conc_factor[spc_optmz_id]

            dt_open_before = (df_chlut_current['ch_cls'].values[0] -
                              df_chlut_current['ch_o_b'].values[0]) * 86400.
            dt_close = (df_chlut_current['ch_o_a'].values[0] -
                        df_chlut_current['ch_cls'].values[0]) * 86400.
            dt_open_after = (df_chlut_current['ch_end'].values[0] -
                             df_chlut_current['ch_o_a'].values[0]) * 86400.

            # print('DEBUG: ', np.nanmin(time_optmz), np.nanmax(time_optmz))
            # print('DEBUG: ', time_optmz, conc_optmz)
            # print('DEBUG: ', dt_open_before, dt_close, dt_open_after)

            timelag_optmz_results = optimize_timelag(
                time_optmz, conc_optmz, t_turnover[loop_num],
                dt_open_before, dt_close, dt_open_after,
                closure_period_only=True,
                bounds=(timelag_lower_limit, timelag_upper_limit),
                guess=timelag_nominal[loop_num])
            timelag_optmz[loop_num], status_timelag_optmz[loop_num] = \
                timelag_optmz_results
            # print('DEBUG: ', timelag_optmz_results)
            timelag_in_day = timelag_optmz_results[0] / 86400.  # in day
        elif (run_options['timelag_method'] == 'prescribed' and
              df_timelag is not None):
            df_timelag_subset = \
                df_timelag.loc[df_timelag['ch_no'] == ch_no[loop_num], :]
            timelag_nominal[loop_num] = np.interp(
                ch_start[loop_num], df_timelag_subset['time_doy'].values,
                df_timelag_subset['timelag_nom'].values)
            timelag_upper_limit = np.interp(
                ch_start[loop_num], df_timelag_subset['time_doy'].values,
                df_timelag_subset['timelag_uplim'].values)
            # a temporary variable
            timelag_lower_limit = np.interp(
                ch_start[loop_num], df_timelag_subset['time_doy'].values,
                df_timelag_subset['timelag_lolim'].values)
            # a temporary variable

            # timelag_nominal[loop_num] = \
            #     df_chlut_current['timelag_nominal'].values[0] * 86400.
            # timelag_upper_limit = \
            #     df_chlut_current['timelag_upper_limit'].values[0] * 86400.
            # # a temporary variable
            # timelag_lower_limit = \
            #     df_chlut_current['timelag_lower_limit'].values[0] * 86400.
            # # a temporary variable

            # print('DEBUG: ', timelag_lower_limit, timelag_upper_limit)
            ind_optmz = np.where(
                (doy_conc > ch_o_b[loop_num]) &
                (doy_conc < ch_end[loop_num] +
                    timelag_upper_limit / 86400.))[0]
            time_optmz = (doy_conc[ind_optmz] - ch_start[loop_num]) * 86400.
            conc_optmz = \
                df_conc.loc[ind_optmz, species_list[spc_optmz_id]].values * \
                conc_factor[spc_optmz_id]

            dt_open_before = (df_chlut_current['ch_cls'].values[0] -
                              df_chlut_current['ch_o_b'].values[0]) * 86400.
            dt_close = (df_chlut_current['ch_o_a'].values[0] -
                        df_chlut_current['ch_cls'].values[0]) * 86400.
            dt_open_after = (df_chlut_current['ch_end'].values[0] -
                             df_chlut_current['ch_o_a'].values[0]) * 86400.

            # print('DEBUG: ', np.nanmin(time_optmz), np.nanmax(time_optmz))
            # print('DEBUG: ', time_optmz, conc_optmz)
            # print('DEBUG: ', dt_open_before, dt_close, dt_open_after)

            timelag_optmz_results = optimize_timelag(
                time_optmz, conc_optmz, t_turnover[loop_num],
                dt_open_before, dt_close, dt_open_after,
                closure_period_only=True,
                bounds=(timelag_lower_limit, timelag_upper_limit),
                guess=timelag_nominal[loop_num])
            timelag_optmz[loop_num], status_timelag_optmz[loop_num] = \
                timelag_optmz_results
            # print('DEBUG: ', timelag_optmz_results)
            timelag_in_day = timelag_optmz_results[0] / 86400.  # in day
        else:
            timelag_in_day = 0.

        # 'ind_ch_full' index is only used for plotting
        ind_ch_full = np.where((doy_conc > ch_o_b[loop_num]) &
                               (doy_conc < ch_end[loop_num] +
                                timelag_in_day))[0]
        ind_atmb = np.where(
            (doy_conc > ch_start[loop_num] + timelag_in_day + dt_lmargin) &
            (doy_conc < ch_o_b[loop_num] + timelag_in_day - dt_rmargin))[0]
        ind_chb = np.where(
            (doy_conc > ch_o_b[loop_num] + timelag_in_day + dt_lmargin) &
            (doy_conc < ch_cls[loop_num] + timelag_in_day - dt_rmargin))[0]
        ind_chc = np.where(
            (doy_conc > ch_cls[loop_num] + timelag_in_day + dt_lmargin) &
            (doy_conc < ch_o_a[loop_num] + timelag_in_day - dt_rmargin))[0]
        # Note: after the line is switched, regardless of the time lag,
        # the analyzer will sample the next line.
        # This is the reason that a time lag is not added to the terminal time.
        ind_cha = np.where(
            (doy_conc > ch_o_a[loop_num] + timelag_in_day + dt_lmargin) &
            (doy_conc < ch_atm_a[loop_num]))[0]
        ind_atma = np.where(
            (doy_conc > ch_atm_a[loop_num] + timelag_in_day + dt_lmargin) &
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
                # - `t_bl_chb`: median or mean time for chamber open period
                #   (before closure), in seconds
                # - `t_bl_cha`: median or mean time for chamber open period
                #   (after closure), in seconds
                if (species_settings[species_list[spc_id]][
                        'baseline_correction'] in ['mean', 'average']):
                    bl_calc_func = np.nanmean
                else:
                    bl_calc_func = np.nanmedian

                t_bl_chb = bl_calc_func(
                    doy_conc[ind_chb] - ch_start[loop_num]) * 86400.
                t_bl_cha = bl_calc_func(
                    doy_conc[ind_cha] - ch_start[loop_num]) * 86400.
                conc_bl_chb = bl_calc_func(
                    df_conc.loc[ind_chb, species_list[spc_id]].values) * \
                    conc_factor[spc_id]

                if (species_settings[species_list[spc_id]][
                        'baseline_correction'] in ['none', 'None', None]):
                    conc_bl_cha = conc_bl_chb
                else:
                    conc_bl_cha = bl_calc_func(
                        df_conc.loc[ind_cha, species_list[spc_id]].values) * \
                        conc_factor[spc_id]
                    # if `conc_bl_cha` is not a finite value, set it equal to
                    # `conc_bl_chb`. Thus `k_bl` will be zero.
                    if np.isnan(conc_bl_cha):
                        conc_bl_cha = conc_bl_chb
                        t_bl_cha = chc_time[-1]

                k_bl = (conc_bl_cha - conc_bl_chb) / \
                    (t_bl_cha - t_bl_chb)
                b_bl = conc_bl_chb - k_bl * t_bl_chb

                # subtract the baseline to correct for instrument drift
                # (assuming linear drift)
                conc_bl = k_bl * chc_time + b_bl

                # linear fit
                # -------------------------------------------------------------
                # see the supp. info of Sun et al. (2016) JGR-Biogeosci.
                y_fit = (chc_conc - conc_bl) * flow[loop_num] / A_ch[loop_num]
                x_fit = np.exp(- (chc_time - chc_time[0] +
                                  dt_lmargin * 8.64e4) / t_turnover[loop_num])

                # boolean index array for finite concentration values
                ind_conc_fit = np.isfinite(y_fit)

                # number of valid observations
                n_obs[loop_num, spc_id] = np.sum(ind_conc_fit)

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
                # @TODO: replace Theil-Sen estimator with RANSAC method??
                # the original algorithm of Theil-Sen method uses numpy.sort()
                # and is thus time consuming
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
                t_fit = (chc_time - chc_time[0] + dt_lmargin * 8.64e4) / \
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
                conc_bl_pts[spc_id, :] = conc_bl_chb, conc_bl_cha

            # used for plotting the baseline
            t_bl_pts[:] = t_bl_chb, t_bl_cha

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
                    axes[i].axvline(x=timelag_in_day * 86400.,
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
                                       ' (%s)' % conc_unit_names[i])
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

    header = create_output_header(
        'flux', species_settings['species_list'], biomet_var_list)

    # quality flags for fluxes
    qc_cols = ['qc_' + s for s in species_settings['species_list']]

    # add number of valid observations of concentrations
    n_obs_cols = ['n_obs_' + s for s in species_settings['species_list']]

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
    df_flux['t_lag_nom'] = timelag_nominal
    df_flux['t_lag_optmz'] = timelag_optmz
    df_flux['status_tlag'] = status_timelag_optmz
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

    # flagging using Dixon's Q test for outlier detection on small samples
    # 1 - if outlier exists in the flux values from three fitting methods
    # 0 - if no outlier exists
    # Note: the flagging system is at its best suggestive, not categorical
    for spc_id in range(n_species):
        arr_flag_test = np.hstack((
            flux_lin[:, [spc_id]], flux_rlin[:, [spc_id]],
            flux_nonlin[:, [spc_id]]))
        for k in range(n_smpl_per_day):
            try:
                dixon_test_res = dixon_test(arr_flag_test[k, :])
            except ValueError:
                qc_flags[k, spc_id] = 0
            else:
                if dixon_test_res == [None, None]:
                    qc_flags[k, spc_id] = 0
                else:
                    qc_flags[k, spc_id] = 1

    for spc_id in range(n_species):
        # concentrations
        df_flux[species_list[spc_id] + '_atmb'] = conc_atmb[:, spc_id]
        df_flux['sd_%s_atmb' % species_list[spc_id]] = sd_conc_atmb[:, spc_id]
        df_flux[species_list[spc_id] + '_chb'] = conc_chb[:, spc_id]
        df_flux['sd_%s_chb' % species_list[spc_id]] = sd_conc_chb[:, spc_id]
        df_flux[species_list[spc_id] + '_cha'] = conc_cha[:, spc_id]
        df_flux['sd_%s_cha' % species_list[spc_id]] = sd_conc_cha[:, spc_id]
        df_flux[species_list[spc_id] + '_atma'] = conc_atma[:, spc_id]
        df_flux['sd_%s_atma' % species_list[spc_id]] = sd_conc_atma[:, spc_id]
        df_flux[species_list[spc_id] + '_chc_iqr'] = conc_chc_iqr[:, spc_id]
        # fluxes
        df_flux['f%s_lin' % species_list[spc_id]] = flux_lin[:, spc_id]
        df_flux['se_f%s_lin' % species_list[spc_id]] = se_flux_lin[:, spc_id]
        df_flux['f%s_rlin' % species_list[spc_id]] = flux_rlin[:, spc_id]
        df_flux['se_f%s_rlin' % species_list[spc_id]] = se_flux_rlin[:, spc_id]
        df_flux['f%s_nonlin' % species_list[spc_id]] = flux_nonlin[:, spc_id]
        df_flux['se_f%s_nonlin' % species_list[spc_id]] = \
            se_flux_nonlin[:, spc_id]
        # others
        df_flux['qc_%s' % species_list[spc_id]] = qc_flags[:, spc_id]
        df_flux['n_obs_%s' % species_list[spc_id]] = n_obs[:, spc_id]

    # rounding off to reduce output file size
    # '%.6f' is the accuracy of single-precision floating numbers
    # do not round off day of year variables or chamber descriptors
    df_flux = df_flux.round(
        {key: 6 for key in df_flux.columns.values
         if key not in ['doy_utc', 'doy_local', 'ch_no',
                        'ch_label', 'A_ch', 'V_ch'] + qc_cols + n_obs_cols})

    df_flux.to_csv(output_fname, sep=',', na_rep='NaN', index=False)
    # no need to have 'row index', therefore, set `index=False`

    print('Raw data on the day %s processed.' % run_date_str)
    print('Data table saved to %s' % output_fname)

    # output curve fitting diagnostics
    # =========================================================================
    if run_options['save_fitting_diagnostics']:
        if data_dir['output_filename_prefix'] != '':
            diag_fname = output_dir + '/diag/' + \
                data_dir['output_filename_prefix'] + \
                '_diag_%s.csv' % run_date_str
        else:
            diag_fname = output_dir + '/diag/' + \
                'diag_' + run_date_str + '.csv'

        header_diag = \
            create_output_header('diag', species_settings['species_list'])

        # create output dataframe for fitting diagnostics
        # `dtype` not needed since it self-adapts to the assigned columns
        df_diag = pd.DataFrame(index=range(n_smpl_per_day),
                               columns=header_diag)

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
            df_diag['delta_rlin_' + species_list[spc_id]] = \
                delta_rlin[:, spc_id]

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
             if (key not in ['doy_utc', 'doy_local', 'ch_no'] and
                 'p_' not in key)})

        df_diag.to_csv(diag_fname, sep=',', na_rep='NaN', index=False)
        # no need to have 'row index', therefore, set `index=False`

        print('Curve fitting diagnostics saved to %s' % diag_fname)

    # generate daily summary plots
    if run_options['save_daily_plots']:
        dailyplot_fontsize = 9
        hr_local = (ch_time - np.round(ch_time[0])) * 24.
        if config['biomet_data_settings']['time_in_UTC']:
            tz_str = 'UTC'
        else:
            tz_str = 'UTC+%d' % site_parameters['time_zone']

        arr_ch_label = np.array(ch_label)
        unique_ch_label = np.unique(arr_ch_label)
        fig_daily, axes_daily = plt.subplots(
            nrows=n_species, ncols=len(unique_ch_label),
            sharex=True, sharey=False,
            figsize=(2.5 * len(unique_ch_label) + 1., 2 * n_species + 0.5))
        for j in range(n_species):
            for k, lb_ch in enumerate(unique_ch_label):
                axes_daily[j, k].errorbar(
                    hr_local[arr_ch_label == lb_ch],
                    flux_lin[arr_ch_label == lb_ch, j],
                    yerr=se_flux_lin[arr_ch_label == lb_ch, spc_id] * 2.,
                    c='#d62728', fmt='o', markeredgecolor='None', markersize=5,
                    linestyle='-', lw=1.5, capsize=0, label='linear')
                axes_daily[j, k].errorbar(
                    hr_local[arr_ch_label == lb_ch],
                    flux_rlin[arr_ch_label == lb_ch, j],
                    yerr=se_flux_rlin[arr_ch_label == lb_ch, spc_id] * 2.,
                    c='#1f77b4', fmt='d', markeredgecolor='None', markersize=5,
                    linestyle='--', lw=1.5, capsize=0, label='robust linear')
                axes_daily[j, k].errorbar(
                    hr_local[arr_ch_label == lb_ch],
                    flux_nonlin[arr_ch_label == lb_ch, j],
                    yerr=se_flux_nonlin[arr_ch_label == lb_ch, spc_id] * 2.,
                    c='#7f7f7f', fmt='P', markeredgecolor='None', ms=5,
                    linestyle='-.', lw=1.5, capsize=0, label='nonlinear')
                axes_daily[j, k].tick_params(labelsize=dailyplot_fontsize)
                plt.setp(axes_daily[j, k].get_xticklabels(), visible=True)

        # set y labels
        for j in range(n_species):
            axes_daily[j, 0].set_ylabel(
                species_settings['species_names'][j] +
                ' (%s)' % flux_unit_names[j], fontsize=dailyplot_fontsize)

        # set common column titles and x-axes
        for k, lb_ch in enumerate(unique_ch_label):
            axes_daily[0, k].set_title('\n' + lb_ch,
                                       fontsize=dailyplot_fontsize)
            axes_daily[-1, k].set_xlim((0, 24))
            axes_daily[-1, k].xaxis.set_major_locator(
                ticker.MultipleLocator(6))
            axes_daily[-1, k].xaxis.set_minor_locator(
                ticker.MultipleLocator(2))
            axes_daily[-1, k].set_xlabel('Hour (%s)' % tz_str,
                                         fontsize=dailyplot_fontsize)

        # figure legend
        fig_daily.suptitle(run_date_str, x=0.01, horizontalalignment='left',
                           fontsize=dailyplot_fontsize)
        fig_daily.legend(handles=axes_daily[0, 0].lines[-3:],
                         labels=['linear', 'robust linear', 'nonlinear'],
                         loc='upper right', ncol=3, handlelength=3,
                         fontsize=dailyplot_fontsize,
                         frameon=False, framealpha=0.5)

        fig_daily.tight_layout()
        fig_daily.savefig(daily_plots_dir + 'daily_flux_%s.png' % run_date_str)

        # important! release the memory after figure is saved
        fig_daily.clf()
        plt.close()

        print('Daily flux summary plots generated.')

    print('\n------\n')
    return None


def main():
    # Echo program starting
    # =========================================================================
    print('Starting data processing...')
    dt_start = datetime.datetime.now()
    print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
    print('Checking specifications:')
    print('numpy version = %s\n' % np.__version__ +
          'pandas version = %s\n' % pd.__version__ +
          'matplotlib version = %s\n' % mpl.__version__ +
          'Config file is set as `%s`' % args.config)

    # Load config files
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

    # customize matplotlib plotting style
    if config['run_options']['plot_style'] is not None:
        plt.style.use(config['run_options']['plot_style'])

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

    # Load data files
    # =========================================================================
    if config['run_options']['load_data_by_day']:
        # this branch loads data by daily chunks
        biomet_data_flist = glob.glob(config['data_dir']['biomet_data'])
        biomet_query_list, biomet_date_series = extract_date_substr(
            biomet_data_flist,
            date_format=config['data_dir']['biomet_data.date_format'])

        # read leaf data (outside of the loop)
        if config['data_dir']['separate_leaf_data']:
            print('Notice: Leaf area data are stored separately ' +
                  'from chamber configuration.')
            # if leaf data are in their own files, read from files
            df_leaf = load_tabulated_data('leaf', config)
            # check data size; if no data entry in it, terminate the program
            if df_leaf is None:
                raise RuntimeError('No leaf area data file is found.')
            elif df_leaf.shape[0] == 0:
                raise RuntimeError('No entry in the leaf area data.')
            # check timestamp existence
            if 'timestamp' not in df_leaf.columns.values:
                raise RuntimeError(
                    'No time variable found in the leaf area data.')
        else:
            # if leaf data are not in their own files, set them to None
            df_leaf = None

        # read timelag data (outside of the loop)
        if config['data_dir']['use_timelag_data']:
            # if use external timelag data
            print('Notice: Use external timelag data.')
            df_timelag = load_tabulated_data('timelag', config)
            # check data size; if no data entry in it, terminate the program
            if df_timelag is None:
                raise RuntimeError('No timelag data file is found.')
            elif df_timelag.shape[0] == 0:
                raise RuntimeError('No entry in the timelag data.')
            # check timestamp existence
            if 'timestamp' not in df_timelag.columns.values:
                raise RuntimeError(
                    'No time variable found in the timelag data.')
        else:
            # if not using external timelag data
            df_timelag = None

        for i in range(len(biomet_query_list)):
            biomet_ts_query = biomet_query_list[i]
            conc_ts_query = biomet_date_series[i].strftime(
                config['data_dir']['conc_data.date_format'])
            flow_ts_query = biomet_date_series[i].strftime(
                config['data_dir']['flow_data.date_format'])
            # read biomet data
            df_biomet = \
                load_tabulated_data('biomet', config, query=biomet_ts_query)
            # check data size; if no data entry in it, skip
            if df_biomet is None:
                print('No biomet data file is found on day %s. Skip.' %
                      biomet_ts_query)
                continue
            elif df_biomet.shape[0] == 0:
                print('No entry in the biomet data on day %s. Skip.' %
                      biomet_ts_query)
            # check timestamp existence
            if 'timestamp' not in df_biomet.columns.values:
                print('No time variable found in the biomet data ' +
                      'on day %s. Skip.' % biomet_ts_query)

            # read concentration data
            if config['data_dir']['separate_conc_data']:
                # if concentration data are in their own files, read from files
                df_conc = \
                    load_tabulated_data('conc', config, query=conc_ts_query)
                # check data size; if no data entry in it, skip
                if df_conc is None:
                    print('No concentration data file is found on day %s. ' %
                          conc_ts_query + 'Skip.')
                    continue
                elif df_conc.shape[0] == 0:
                    print('No entry in the concentration data on day %s. ' %
                          conc_ts_query + 'Skip.')
                # check timestamp existence
                if 'timestamp' not in df_conc.columns.values:
                    print('No time variable found in the concentration data' +
                          'on day %s. Skip.' % conc_ts_query)
            else:
                # if concentration data are not in their own files
                # create aliases for biomet data and the parsed time variable
                df_conc = df_biomet

            # read flow data
            if config['data_dir']['separate_flow_data']:
                # if flow data are in their own files, read from files
                df_flow = \
                    load_tabulated_data('flow', config, query=flow_ts_query)
                # check data size; if no data entry in it, skip
                if df_flow is None:
                    print('No flow data file is found on day %s. Skip.' %
                          flow_ts_query)
                elif df_flow.shape[0] == 0:
                    print('No entry in the flow data on day %s. Skip.' %
                          flow_ts_query)
                # check timestamp existence
                if 'timestamp' not in df_flow.columns.values:
                    print('No time variable found in the flow rate data ' +
                          'on day %s. Skip.' % flow_ts_query)
            else:
                # if flow data are not in their own files, create aliases for
                # biomet data and the parsed time variable
                df_flow = df_biomet

            year_biomet = df_biomet.loc[0, 'timestamp'].year
            if year_biomet is None:
                year_biomet = config['biomet_data_settings']['year_ref']

            year_conc = df_conc.loc[0, 'timestamp'].year
            if year_conc is None:
                year_conc = config['conc_data_settings']['year_ref']

            if (year_biomet != year_conc and
                    config['data_dir']['separate_conc_data']):
                raise RuntimeError('Year numbers do not match between ' +
                                   'biomet data and concentration data.')

            # Calculate fluxes, and output plots and the processed data
            # =================================================================
            print('Calculating fluxes...')

            doy = (biomet_date_series[i] -
                   pd.Timestamp('%s-01-01' % year_biomet)) / \
                pd.Timedelta(days=1)
            year = year_biomet

            # calculate fluxes
            flux_calc(df_biomet, df_conc, df_flow, df_leaf, df_timelag,
                      doy, year, config, chamber_config)
    else:
        # this branch loads all the data at once
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
            print('Notice: Concentration data are extracted from biomet ' +
                  'data, because they are not stored in their own files.')

        # read flow data
        if config['data_dir']['separate_flow_data']:
            # if flow data are in their own files, read from files
            df_flow = load_tabulated_data('flow', config, query=ts_query)
            # check data size; if no data entry in it, terminate the program
            if df_flow is None:
                raise RuntimeError('No flow data file is found.')
            elif df_flow.shape[0] == 0:
                raise RuntimeError('No entry in the flow data.')
            # check timestamp existence
            if 'timestamp' not in df_flow.columns.values:
                raise RuntimeError(
                    'No time variable found in the flow rate data.')
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
            df_leaf = load_tabulated_data('leaf', config)
            # check data size; if no data entry in it, terminate the program
            if df_leaf is None:
                raise RuntimeError('No leaf area data file is found.')
            elif df_leaf.shape[0] == 0:
                raise RuntimeError('No entry in the leaf area data.')
            # check timestamp existence
            if 'timestamp' not in df_leaf.columns.values:
                raise RuntimeError(
                    'No time variable found in the leaf area data.')
        else:
            # if leaf data are not in their own files, set them to None
            df_leaf = None

        # read timelag data
        if config['data_dir']['use_timelag_data']:
            # if use external timelag data
            print('Notice: Use external timelag data.')
            df_timelag = load_tabulated_data('timelag', config)
            # check data size; if no data entry in it, terminate the program
            if df_timelag is None:
                raise RuntimeError('No timelag data file is found.')
            elif df_timelag.shape[0] == 0:
                raise RuntimeError('No entry in the timelag data.')
            # check timestamp existence
            if 'timestamp' not in df_timelag.columns.values:
                raise RuntimeError(
                    'No time variable found in the timelag data.')
        else:
            # if not using external timelag data
            df_timelag = None

        year_biomet = df_biomet.loc[0, 'timestamp'].year
        if year_biomet is None:
            year_biomet = config['biomet_data_settings']['year_ref']

        year_conc = df_conc.loc[0, 'timestamp'].year
        if year_conc is None:
            year_conc = config['conc_data_settings']['year_ref']

        if (year_biomet != year_conc and
                config['data_dir']['separate_conc_data']):
            raise RuntimeError('Year numbers do not match between ' +
                               'biomet data and concentration data.')

        # Calculate fluxes, and output plots and the processed data
        # =====================================================================
        print('Calculating fluxes...')

        doy_start = np.floor(df_biomet['time_doy'].min())  # NaN skipped
        doy_end = np.ceil(df_biomet['time_doy'].max())  # NaN skipped
        year = year_biomet

        # calculate fluxes day by day
        for doy in np.arange(doy_start, doy_end):
            flux_calc(df_biomet, df_conc, df_flow, df_leaf, df_timelag,
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
