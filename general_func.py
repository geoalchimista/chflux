# -*- coding: utf-8 -*-
# use python 3

import numpy as np
import pandas as pd
import scipy.optimize as optmz


def chamber_lookup_table_func(doy, return_all=False):
    """
    Return a chamber meta information look-up table (pandas.DataFrame). 

    Parameters
    ----------
    doy : float
        Days since Jan 1 00:00 of the current year.
    return_all : boolean, optional
        By default `False`, which means only the chamber lookup table is returned.
        If set as `True`, the function also returns other related information (see "Returns"). 

    Returns
    -------
    chlut : pandas.DataFrame
        Chamber meta information lookup table, including these columns:
        * 'ch_no': the sequence of chamber numbers in a full cycle
        * 'ch_label': strings used to label the chambers
        * 'flowmeter_no': ID numbers of flowmeters corresponding to the chambers; assign -1 if not existent
        * 'TC_no': ID numbers of temperature readings corresponding to the chamber numbers; assign -1 if not existent
        * 'PAR_no': ID numbers of PAR readings corresponding to the chamber numbers; assign -1 if not existent
        * 'A_ch': chamber-covered surface areas (for soil flux) or leaf areas (for leaf flux)
        * 'A_ch_std': standard deviations of chamber-covered surface areas (for soil flux) or leaf areas (for leaf flux)
        * 'V_ch': chamber volumes
        * 'ch_start': starting times of chamber sampling with respect to the start of each cycle, in fraction of a day
        * 'ch_o_b': times when chambers open, with respect to 'ch_start', in fraction of a day
        * 'ch_cls': times when chambers close, with respect to 'ch_start', in fraction of a day
        * 'ch_o_a': times when chambers reopen after a closure period, with respect to 'ch_start', in fraction of a day
        * 'ch_end': times when chamber sampling ends, with respect to 'ch_start', in fraction of a day
    Note: you may change the intermediate steps that generate `chlut` as you like, since they are only used for 
        getting the lookup table. As long as the returned `chlut` has the above columns properly assigned, it should 
        work well with the main program (`flux_calc.py`). 

    If `return_all == True`, return also these variables:
    `n_ch` : integer
        The number of chambers being alternated in this schedule. 
        Note: a chamber may be sampled more than once in a full cycle, for example, chamber sequence 1-2-3-3-2-1. 
    `smpl_cycle_len` : float
        The duration of a full cycle over all chambers, in fraction of a day
    `n_cycle_per_day` : integer
        The number of full sampling cycles in a day. 
    `next_schedule_switch` : float
        The time (in day of year) when the sampling schedule switched to the next one. 
    
    Raises
    ------
    None
    """
    # chamber parameters
    # there are 6 unique chambers in this example
    # to be assigned: chamber volumes and surface (leaf) areas, `V_ch` and `A_ch`
    V_ch = np.zeros(6) * np.nan
    A_ch = np.zeros(6) * np.nan
    A_ch_std = np.zeros(6) * np.nan

    lc_height = 30.0e-2    # leaf chamber (ABT Sorime, fabrique en France), in m
    lc_radius = 12.0e-2
    skirt_height = np.array([11., 6., 11.5]) * 1e-2
    # heights of plastic skirt at the bottom (m), approximated with paraboloid
    n_leaves = np.array([87, 67,])  # leaf number in each branch chamber
    A_leaf = 8.1 * 1e-4  # average area per leaf (m^2)
    A_leaf_std = 1.5 * 1e-4 # standard deviation of leaf area (m^2)
    sc_collar_height = np.array([5.40, 5.00, 9.45]) * 1e-2  # soil chamber collar height (m)

    A_ch[0:2] = A_leaf * n_leaves  # leaf area in each branch chamber (m^2)
    A_ch[2] = np.pi * lc_radius ** 2 + 2 * np.pi * lc_radius * lc_height  # a blank one, use inner surface area
    A_ch[3:6] = 317.8 * 1e-4  # soil area covered by the chamber (m^2), from LI-8100A manual
    A_ch_std[0:2] = A_leaf_std * n_leaves  # standard deviations of leaf area in each branch chamber (m^2)
    A_ch_std[2:] = 0.  # for soil surface areas, no standard deviation is assigned
    V_ch[0:3] = (lc_height + skirt_height * 0.5) * np.pi * lc_radius ** 2  # branch chamber volumes (m^3)
    V_ch[3:6] = 4076.1 * 1e-6 + sc_collar_height * A_ch[3:6]  # soil chamber volume (m^3), bowl + collar volume

    """
    Chamber schedule
    ----------------
    For chamber #1
    * Background calibration of analyzer: hh:00:00 - hh:02:00
    * Atmospheric line: hh:02:00 - hh:03:00
    * Chamber opening (before closure): hh:03:00 - hh:05:00
    * Chamber closure: hh:05:00 - hh:13:00
    * Chamber reopening (after closure): hh:13:00 - hh:15:00

    Then add 15 minutes for the schedule of the next chamber measurements...
    """
    if doy < 105.5:  # from the beginning to 04/16/2013 12:00 (UTC)     
        chlut = pd.DataFrame([1,2,3,4,5,6], columns=['ch_no'])
        chlut['ch_label'] = ['LC1', 'LC2', 'LC3', 'SC1', 'SC2', 'SC3']
        chlut['flowmeter_no'] = chlut['ch_no']
        chlut['TC_no'] = chlut['ch_no']
        chlut['PAR_no'] = [1, 2, 3, -1, -1, -1]
        chlut['A_ch'] = A_ch[chlut['ch_no']-1]
        chlut['A_ch_std'] = A_ch_std[chlut['ch_no']-1]
        chlut['V_ch'] = V_ch[chlut['ch_no']-1]
        chlut['ch_start'] = (2. + 15. * np.arange(6)) / 1440.  # minutes converted to day
        chlut['ch_o_b'] = np.ones(6) / 1440. # with respect to ch_start
        chlut['ch_cls'] = np.ones(6) * 3. / 1440. # with respect to ch_start
        chlut['ch_o_a'] = np.ones(6) * 11. / 1440. # with respect to ch_start
        chlut['ch_end'] = np.ones(6) * 13. / 1440. # with respect to ch_start
        chlut['ch_atm_a'] = np.ones(6) * 13. / 1440. # with respect to ch_start
        # if switch back to atmospheric line after chamber reopening, set the time with the key `chlut['ch_atm_a']`
        # else, set the key to the same as `chlut['ch_end']`

        smpl_cycle_len = 1.5 / 24.  # 1.5 hours per cycle
        n_cycle_per_day = np.int(1. / smpl_cycle_len)  # 16 full cycles per day
        n_ch = chlut['ch_no'].size
        next_schedule_switch = 105.5

    elif doy >= 105.5 and doy < 115.9:  # from 04/16/2013 12:00 to 04/26/2013 21:40 (UTC)
        chlut = pd.DataFrame([1,2,4,1,2,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC1', 'LC2', 'SC1', 'LC1', 'LC2', 'SC2']
        chlut['flowmeter_no'] = chlut['ch_no']
        chlut['TC_no'] = chlut['ch_no']
        chlut['PAR_no'] = [1, 2, -1, 1, 2, -1]
        chlut['A_ch'] = A_ch[chlut['ch_no']-1]
        chlut['A_ch_std'] = A_ch_std[chlut['ch_no']-1]
        chlut['V_ch'] = V_ch[chlut['ch_no']-1]
        chlut['ch_start'] = (2. + 15. * np.arange(6)) / 1440.  # minutes converted to day
        chlut['ch_o_b'] = np.ones(6) / 1440. # with respect to ch_start
        chlut['ch_cls'] = np.ones(6) * 3. / 1440. # with respect to ch_start
        chlut['ch_o_a'] = np.ones(6) * 11. / 1440. # with respect to ch_start
        chlut['ch_end'] = np.ones(6) * 13. / 1440. # with respect to ch_start
        chlut['ch_atm_a'] = np.ones(6) * 13. / 1440. # with respect to ch_start
        # if switch back to atmospheric line after chamber reopening, set the time with the key `chlut['ch_atm_a']`
        # else, set the key to the same as `chlut['ch_end']`

        smpl_cycle_len = 1.5 / 24.  # 1.5 hours per cycle
        n_cycle_per_day = np.int(1. / smpl_cycle_len)  # 16 full cycles per day
        n_ch = chlut['ch_no'].size
        next_schedule_switch = 115.9

    elif doy >= 115.9 and doy < 127. + 1./24.:  # from 04/26/2013 21:40 to 05/08/2013 01:00 (UTC)
        chlut = pd.DataFrame([1,2,4,3,1,2,5,3], columns=['ch_no'])
        chlut['ch_label'] = ['LC1', 'LC2', 'SC1', 'LC3', 'LC1', 'LC2', 'SC2', 'LC3']
        chlut['flowmeter_no'] = chlut['ch_no']
        chlut['TC_no'] = chlut['ch_no']
        chlut['PAR_no'] = [1, 2, -1, 3, 1, 2, -1, 3]
        chlut['A_ch'] = A_ch[chlut['ch_no']-1]
        chlut['A_ch_std'] = A_ch_std[chlut['ch_no']-1]
        chlut['V_ch'] = V_ch[chlut['ch_no']-1]
        chlut['ch_start'] = (2. + 15. * np.arange(8)) / 1440.  # minutes converted to day
        chlut['ch_o_b'] = np.ones(8) / 1440. # with respect to ch_start
        chlut['ch_cls'] = np.ones(8) * 3. / 1440. # with respect to ch_start
        chlut['ch_o_a'] = np.ones(8) * 11. / 1440. # with respect to ch_start
        chlut['ch_end'] = np.ones(8) * 13. / 1440. # with respect to ch_start
        chlut['ch_atm_a'] = np.ones(8) * 13. / 1440. # with respect to ch_start
        # if switch back to atmospheric line after chamber reopening, set the time with the key `chlut['ch_atm_a']`
        # else, set the key to the same as `chlut['ch_end']`

        smpl_cycle_len = 2. / 24.  # 2 hours per cycle
        n_cycle_per_day = np.int(1. / smpl_cycle_len)  # 12 full cycle per day
        n_ch = chlut['ch_no'].size
        next_schedule_switch = 127. + 1./24.

    elif doy >= 127. + 1./24.:  # from 05/08/2013 01:00 (UTC) to the end
        chlut = pd.DataFrame([1,2,4,6,1,2,5,6], columns=['ch_no'])
        chlut['ch_label'] = ['LC1', 'LC2', 'SC1', 'LC3', 'LC1', 'LC2', 'SC2', 'LC3']
        chlut['flowmeter_no'] = chlut['ch_no']
        chlut['TC_no'] = chlut['ch_no']
        chlut['PAR_no'] = [1, 2, -1, 3, 1, 2, -1, 3]
        chlut['A_ch'] = A_ch[chlut['ch_no']-1]
        chlut['A_ch_std'] = A_ch_std[chlut['ch_no']-1]
        chlut['V_ch'] = V_ch[chlut['ch_no']-1]
        chlut['ch_start'] = (2. + 15. * np.arange(8)) / 1440.  # minutes converted to day
        chlut['ch_o_b'] = np.ones(8) / 1440. # with respect to ch_start
        chlut['ch_cls'] = np.ones(8) * 3. / 1440. # with respect to ch_start
        chlut['ch_o_a'] = np.ones(8) * 11. / 1440. # with respect to ch_start
        chlut['ch_end'] = np.ones(8) * 13. / 1440. # with respect to ch_start
        chlut['ch_atm_a'] = np.ones(8) * 13. / 1440. # with respect to ch_start
        # if switch back to atmospheric line after chamber reopening, set the time with the key `chlut['ch_atm_a']`
        # else, set the key to the same as `chlut['ch_end']`

        smpl_cycle_len = 2. / 24.  # 2 hours per cycle
        n_cycle_per_day = np.int(1. / smpl_cycle_len)  # 12 full cycle per day
        n_ch = chlut['ch_no'].size
        next_schedule_switch = np.nan
    
    if return_all:
        return(chlut, n_ch, smpl_cycle_len, n_cycle_per_day, next_schedule_switch)
    else:
        return(chlut)

def chamber_flow_rates(doy, ch_no):
    """
    Return chamber flow rates. 
    
    Use this function when flow rates are not in the biomet data tables. 
    """
    if ch_no <= 3:
        flow_lpm = 1.5
    if ch_no > 3:
        flow_lpm = 2.
    is_flow_STP = False
    return(flow_lpm, is_flow_STP)

def time_lag_opt_func():
    return 0

def volume_eff_opt_func():
    return 0
'''
def conc_func(flow_rate, turnover_time, ):
    return 0
'''

def conc_func(p, t):
    """
    Calculate the concentration changes in chamber closure period as a function of time.

    Parameters
    ----------
    p : parameter array with two elements
        p[0]: fitted flux
        p[1]: timelag / turnover time
    t : time variable (normalized by the turnover time)

    Returns
    -------
    y : concentration changes in chamber closure period. 
    """
    y = p[0] * (1. - np.exp(-t + p[1]))
    return(y)

def resid_conc_func(p, t, y):
    """
    Calculate the residuals of fitted concentration changes in chamber closure period as a function of time.

    Parameters
    ----------
    p : parameter array with two elements
        p[0]: fitted flux
        p[1]: timelag / turnover time
    t : time variable (normalized by the turnover time)
    y : observations

    Returns
    -------
    resid : residuals of fitted concentration changes in chamber closure period. 
    """
    resid = p[0] * (1. - np.exp(-t + p[1])) - y
    return(resid)

def IQR_func(x):
    """ Calculate the interquartile range of an array. """
    if np.sum(np.isfinite(x)) > 0:
        q1, q3 = np.nanpercentile(x, [25,75])
        return(q3 - q1)
    else:
        return(np.nan)


def p_sat_h2o(temp, ice=False, kelvin=False, method='gg'):
    """
    Calculate saturation vapor pressure over water or ice at a temperature.

    Parameters
    ----------
    temp : float or `numpy.ndarray`
        Temperature, in Celsius degree by default.
    ice : boolean, optional
        Calculate saturation vapor pressure on ice if enabled.
    kelvin : boolean, optional
        Temperature input is in Kelvin if enabled.
    method : string, optional
        Method used to evaluate saturation vapor pressure.
        'gg': default, Goff-Gratch equation (1946). [GG46]_
        'buck': Buck Research Instruments L.L.C. (1996). [B96]_
        'cimo': CIMO Guide (2008). [WMO]_

    Returns
    -------
    e_sat : float or `numpy.ndarray`
        Saturation vapor pressure in Pascal.

    Raises
    ------
    ValueError
        If keyword 'ice' is enabled but temperature is above 0 C or 273.15 K.

    References
    ----------
    .. [GG46] Goff, J. A., and Gratch, S. (1946). Low-­pressure properties of 
              water from -160 to 212 F, in Transactions of the American Society 
              of Heating and Ventilating Engineers, pp 95-­122, presented at 
              the 52nd Annual Meeting of the American Society of Heating and 
              Ventilating Engineers, New York.
    .. [B96]  Buck Research Instruments L.L.C. (1996). Buck Research CR-1A 
              User's Manual, Appendix 1.
    .. [WMO]  World Meteorological Organization. (2008). Guide to 
              Meteorological Instruments and Methods of Observation, 
              Appendix 4B, WMO-No. 8 (CIMO Guide), Geneva.

    Examples
    --------
    >>> print(p_sat_h2o(25))
    3165.19563338

    >>> print(p_sat_h2o([0, 5, 15, 25]))
    [  610.33609993   871.31372986  1703.28100711  3165.19563338]

    >>> print(p_sat_h2o(25, method='buck'))
    3168.53141228

    >>> print(p_sat_h2o(273.15, kelvin=True))
    610.336099933

    >>> print(p_sat_h2o(-15, ice=True))
    165.014773924

    >>> print(p_sat_h2o(273.15, kelvin=True, ice=True, method='cimo'))
    165.287132017
    """
    
    T_k = np.array(temp, dtype='d') + (not kelvin) * 273.15
    # force temperature to be in Kelvin
    # pressure = np.array(pressure, dtype='d')

    if (np.sum(T_k > 273.16) and ice):
        # The triple point of water is 273.16 K
        raise ValueError('Temperature error, no ice exists. ')

    if (not ice):
        # equations for liquid water
        if method == 'buck':
            T_c = T_k - 273.15      # temperature in Celsius degree
            e_sat = 6.1121 * np.exp( (18.678 - T_c / 234.5) * 
                T_c / (257.14 + T_c) ) * 100
        elif method == 'cimo':
            T_c = T_k - 273.15      # temperature in Celsius degree
            e_sat = 6.112 * np.exp( 17.62 * T_c / (243.12 + T_c) ) * 100
        else:
            # Goff-Gratch equation by default
            u_T = 373.16/T_k
            v_T = T_k/373.16
            e_sat = ( - 7.90298 * (u_T - 1) + 5.02808 * np.log10(u_T)
                - 1.3816e-7 * (10**(11.344 * (1 - v_T)) - 1)
                + 8.1328e-3 * (10**(- 3.49149 * (u_T - 1)) - 1) 
                + np.log10(1013.246) )
            e_sat = 10**e_sat * 100
    else:
        # equations for ice
        if method == 'buck':
            T_c = T_k - 273.15      # temperature in Celsius degree
            e_sat = 6.1115 * np.exp( (23.036 - T_c / 333.7) * 
                T_c / (279.82 + T_c) ) * 100
        elif method == 'cimo':
            T_c = T_k - 273.15      # temperature in Celsius degree
            e_sat = 6.112 * np.exp( 22.46 * T_c / (272.62 + T_c) ) * 100
        else:
            # Goff-Gratch equation by default
            u_T = 273.16/T_k
            v_T = T_k/273.16
            e_sat = ( - 9.09718 * (u_T - 1) - 3.56654 * np.log10(u_T) 
              + 0.876793 * (1 - v_T) + np.log10(6.1071) )
            e_sat = 10**e_sat * 100

    return e_sat

def dew_temp(vap_pres):
    """
    Calculate dew temperature from water concentration. 

    Takes only one value as input, does not take arrays.
    """ 
    def __dew_temp_func(T, e_sat):
        return(p_sat_h2o(T) - e_sat)
    return optmz.newton(__dew_temp_func, x0=25, args=(vap_pres,))
