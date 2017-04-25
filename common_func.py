"""
Common functions used in flux calculation

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
from collections import namedtuple
import warnings

import numpy as np
from scipy import optimize
import scipy.constants.constants as sci_const
import pandas as pd


# Physical constants
# Do not modify unless you are in a different universe.
# - 'p_std': 1 standard atmospheric pressure in Pascal
# - 'R_gas': The universal gas constant in J mol^-1 K^-1
# - 'T_0': zero Celsius in Kelvin
# - 'air_conc_std': Air concentration (mol m^-3) at STP condition
phys_const = {
    'T_0': sci_const.zero_Celsius,
    'p_std': sci_const.atm,
    'R_gas': sci_const.R,
    'air_conc_std': sci_const.atm / (sci_const.R * sci_const.zero_Celsius), }
T_0 = phys_const['T_0']


# the metaclass for subclassing
ChamberLookupTableMeta = namedtuple(
    'ChamberLookupTableMeta',
    ['schedule_start', 'schedule_end', 'n_ch', 'smpl_cycle_len',
     'n_cycle_per_day', 'unit_of_time', 'df'])


class ChamberLookupTable(ChamberLookupTableMeta):
    """
    Chamber Information Lookup Table

    Parameters
    ----------
    chamber_config : dict
        Chamber configuration dictionary loaded from file
    time_query : float, str, or pandas.Timestamp
        A time variable for inquiry (day of year number, or datetime string,
        or pandas.Timestamp)
    year : int, optional
        If `time_query` is given as day of year number, must specify the year.

    Attributes
    ----------
    schedule_start : panda.Timestamp
        Starting time of the current schedule
    schedule_end : panda.Timestamp
        End time of the current schedule
    n_ch : int
        Number of chambers
    smpl_cycle_len : float
        Length of a full sampling cycle through all chambers [in day]
    n_cycle_per_day : int
        Number of cycles per day
    unit_of_time : str
        Unit of time in the chamber settings table ('s', 'min', or 'h')
    df : pandas.DataFrame
        Dataframe for the chamber settings table

    """
    def __new__(cls, chamber_config, time_query):
        pass
    # TO BE CONTINUED
    # def __new__(cls, chamber_config, doy):
    #     extract chamber_lookup_table_dict from chamber_config and doy
    #     self = super().__init__(cls, **chamber_lookup_table_dict)
    #     return self (an instance of this class)


def chamber_lookup_table_func(doy, chamber_config):
    """
    Return a chamber meta information look-up table.

    """
    # define returned data template
    ChamberLookupTableResult = namedtuple(
        'ChamberLookupTableResult',
        ['schedule_start', 'schedule_end', 'n_ch', 'smpl_cycle_len',
         'n_cycle_per_day', 'unit_of_time', 'df'])

    for sch_id in chamber_config:
        if type(chamber_config[sch_id]['schedule_start']) is str:
            sch_start = pd.Timestamp(chamber_config[sch_id]['schedule_start'])
            sch_end = pd.Timestamp(chamber_config[sch_id]['schedule_end'])
            sch_start_doy = sch_start.dayofyear - 1. + \
                sch_start.hour / 24. + sch_start.minute / 1440. + \
                sch_start.second / 86400.
            sch_end_doy = sch_end.dayofyear - 1. + \
                sch_end.hour / 24. + sch_end.minute / 1440. + \
                sch_end.second / 86400.
        else:
            sch_start_doy = chamber_config[sch_id]['schedule_start']
            sch_end_doy = chamber_config[sch_id]['schedule_end']
        if (sch_start_doy <= doy < sch_end_doy):
            current_schedule = chamber_config[sch_id]
            break
    else:
        warnings.warn('No valid chamber schedule found on the day %s.' %
                      str(doy), RuntimeWarning)
        return None

    # initialize an empty dictionary and then convert it to a namedtuple
    chamber_lookup_table = {}
    for key in ['schedule_start', 'schedule_end', 'n_ch', 'smpl_cycle_len',
                'n_cycle_per_day', 'unit_of_time', ]:
        chamber_lookup_table[key] = current_schedule[key]

    df = pd.DataFrame()

    for key in ['ch_no', 'A_ch', 'A_ch_std', 'V_ch', 'ch_label',
                'is_leaf_chamber', 'flowmeter_no', 'TC_no', 'PAR_no',
                'ch_start', 'ch_o_b', 'ch_cls', 'ch_o_a',
                'ch_end', 'ch_atm_a', 'optimize_timelag', 'timelag_nominal',
                'timelag_upper_limit', 'timelag_lower_limit']:
        df[key] = current_schedule[key]

    if chamber_lookup_table['unit_of_time'] in ['second', 'sec', 's']:
        time_unit_conversion_factor = 60. * 60. * 24.
    elif chamber_lookup_table['unit_of_time'] in ['minute', 'min', 'm']:
        time_unit_conversion_factor = 60. * 24.
    elif chamber_lookup_table['unit_of_time'] in ['hour', 'hr', 'h']:
        time_unit_conversion_factor = 24.
    else:
        time_unit_conversion_factor = 1

    # convert the unit of all time variables specifying the schedule to day
    # this does not apply to `schedule_start` and `schedule end` since
    # they both are in day of year
    chamber_lookup_table['smpl_cycle_len'] /= time_unit_conversion_factor
    df[['ch_start', 'ch_o_b', 'ch_cls', 'ch_o_a', 'ch_end', 'ch_atm_a',
        'timelag_nominal', 'timelag_upper_limit', 'timelag_lower_limit']] /= \
        time_unit_conversion_factor

    chamber_lookup_table['df'] = df

    # convert to a namedtuple
    # `**` is the 'splat operator' for unpacking dictionaries
    chamber_lookup_table = ChamberLookupTableResult(**chamber_lookup_table)

    return chamber_lookup_table


def optimize_timelag(time, conc, t_turnover,
                     dt_open_before, dt_close, dt_open_after,
                     dt_left_margin=0., dt_right_margin=0.,
                     closure_period_only=False, bounds=None, guess=None):
    """
    The time lag optimization function.

    Parameters
    ----------
    time : array_like
        Time since switching to the current chamber line, in seconds.
    conc : array_like
        Concentrations
    t_turnover : float
        The turnover time, `V_ch_mol` [mol] divided by `f_ch` [mol s^-1].
    dt_open_before : float
        Chamber opening period before closure measurement, in seconds.
    dt_open_after : float
        Chamber opening period after closure measurement, in seconds.
    dt_left_margin : optional, float
        Left margin on the closure period series to exclude, in seconds.
    dt_right_margin : optional, float
        Right margin on the closure period series to exclude, in seconds.
    closure_period_only : optional, bool
        If True (default), use the closure period only in evaluating the cost
        function for time lag optimization. If False, use the closure period
        and the opening period before it to evaluate the cost function.
    bounds : optional, tuple
        Lower and upper bounds of the time lag, in seconds. Only when both
        values are given does it call a bounded optimization.
    guess: optional, float
        The initial guess value for the time lag, in seconds. If not given, the
        default value will be 0.

    Returns
    -------
    timelag : float
        The optimized time lag value in seconds.
    status_timelag : int
        Status code for the time lag optimization procedure.
        * 0 -- Convergence
        * 1 -- Failure to converge.

    """

    def __timelag_resid_func(t_lag, time, conc, t_turnover,
                             dt_open_before, dt_close, dt_open_after,
                             dt_left_margin, dt_right_margin,
                             closure_period_only=False):
        """
        The timelag optimization function to minimize.

        Parameters
        ----------
        t_lag : float
            Time lag, in sec
        time : array_like
            Time since switching to the current chamber line, in seconds.
        conc : array_like
            Concentrations.
        t_turnover : float
            The turnover time, `V_ch_mol` [mol] divided by `f_ch` [mol s^-1].

        Returns
        -------
        MSR : float
            Mean squared difference.

        """
        # all index arrays should only contain the indices of finite values
        _ind_chb = np.where(
            (time >= t_lag + dt_left_margin) &
            (time < t_lag + dt_open_before - dt_right_margin) &
            np.isfinite(time) & np.isfinite(conc))
        _ind_chc = np.where(
            (time >= t_lag + dt_open_before + dt_left_margin) &
            (time < t_lag + dt_open_before + dt_close - dt_right_margin) &
            np.isfinite(time) & np.isfinite(conc))
        _ind_cha = np.where(
            (time >= t_lag + dt_open_before + dt_close + dt_left_margin) &
            (time < t_lag + dt_open_before + dt_close + dt_open_after -
                dt_right_margin) &
            np.isfinite(time) & np.isfinite(conc))

        _median_chb = np.nanmedian(conc[_ind_chb])
        _median_cha = np.nanmedian(conc[_ind_cha])
        _t_mid_chb = np.nanmedian(time[_ind_chb])
        _t_mid_cha = np.nanmedian(time[_ind_cha])

        # baseline
        _k_bl = (_median_cha - _median_chb) / (_t_mid_cha - _t_mid_chb)
        _b_bl = _median_chb - _k_bl * _t_mid_chb
        _conc_bl = _k_bl * time + _b_bl

        _x_obs = 1. - np.exp(- (time[_ind_chc] - t_lag - dt_open_before) /
                             t_turnover)
        _y_obs = conc[_ind_chc] - _conc_bl[_ind_chc]

        if _x_obs.size == 0:
            return(np.nan)
        # if no valid observations in chamber closure period, return NaN value
        # this will terminate the optimization procedure, and returns a
        # 'status code' of 1 in `optimize.minimize`

        # _slope, _intercept, _r_value, _p_value, _sd_slope = \
        #     stats.linregress(_x_obs, _y_obs)
        # _slope, _intercept, _, _, _ = stats.linregress(_x_obs, _y_obs)
        _slope = np.sum(_y_obs * _x_obs) / np.sum(_x_obs * _x_obs)
        _intercept = 0.
        _y_fitted = _slope * _x_obs + _intercept

        if closure_period_only:
            MSR = np.nansum((_y_fitted - _y_obs) ** 2) / \
                (_ind_chc[0].size - 2)  # mean squared residual
        else:
            _conc_fitted = _slope * \
                (1. - np.exp(- (time - t_lag - dt_open_before) /
                             t_turnover)) + _intercept + _conc_bl
            _conc_fitted[(time < t_lag + dt_open_before) |
                         (time > t_lag + dt_open_before + dt_close)] = \
                _conc_bl[(time < t_lag + dt_open_before) |
                         (time > t_lag + dt_open_before + dt_close)]
            resid = conc - _conc_fitted
            # do not include the chamber open period after closure
            resid_trunc = resid[time <= t_lag + dt_open_before + dt_close]
            # degree of freedom = 3
            MSR = np.nansum(resid_trunc * resid_trunc) / \
                (np.sum(np.isfinite(resid_trunc)) - 3.)
        return(MSR)

    # do time lag optimization
    # get the bounds
    # legal input of bounds: 1. must be finite numbers; 2. the lower bound
    # must be less than the upper bound
    if bounds is None:
        flag_bounded_optimization = False
    else:
        try:
            timelag_lolim, timelag_uplim = bounds
        except TypeError:
            flag_bounded_optimization = False
            warnings.warn('Illegal bounds given to timelag optimization!' +
                          'Default to unbounded optimization method.',
                          RuntimeWarning)
        else:
            if np.isfinite(timelag_lolim) and np.isfinite(timelag_uplim):
                flag_bounded_optimization = timelag_lolim < timelag_uplim
            else:
                flag_bounded_optimization = False
                warnings.warn('Illegal bounds given to timelag optimization!' +
                              ' Default to unbounded optimization method.',
                              RuntimeWarning)

    # get initial guess value
    if guess is None:
        timelag_guess = 0.
    else:
        try:
            timelag_guess = float(guess)
        except ValueError:
            warnings.warn('Illegal timelag guess! Default to 0.',
                          RuntimeWarning)
            timelag_guess = 0.
        except TypeError:
            warnings.warn('Illegal timelag guess! Default to 0.',
                          RuntimeWarning)
            timelag_guess = 0.
        if not np.isfinite(timelag_guess):
            warnings.warn('Illegal timelag guess! Default to 0.',
                          RuntimeWarning)
            timelag_guess = 0.

    if flag_bounded_optimization:
        # bounded optimization uses `scipy.optimize.minimize_scalar`
        timelag_results = optimize.minimize_scalar(
            __timelag_resid_func,
            bounds=(timelag_lolim, timelag_uplim),
            args=(time, conc, t_turnover,
                  dt_open_before, dt_close, dt_open_after,
                  dt_left_margin, dt_right_margin,
                  closure_period_only),
            method='bounded', options={'xatol': 1e-6})
        timelag = timelag_results.x
        status_timelag = 0 if timelag_results.success else 1
    else:
        # unbounded optimization uses `scipy.optimize.minimize`
        timelag_results = optimize.minimize(
            __timelag_resid_func, x0=timelag_guess,
            args=(time, conc, t_turnover,
                  dt_open_before, dt_close, dt_open_after,
                  dt_left_margin, dt_right_margin,
                  closure_period_only),
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
        timelag = timelag_results.x[0]
        status_timelag = timelag_results.status

    return timelag, status_timelag


def volume_eff_optmz_func():
    # @TODO: to add this function
    return None


def conc_func(p, t):
    """
    Calculate the changes in concentration in chamber closure period as a
    function of time.

    Parameters
    ----------
    p : list or array with two elements
        Parameter array
            p[0]: fitted flux
            p[1]: timelag / turnover time
    t : array_like
        Normalized time variable (by the turnover time).

    Returns
    -------
    y : array_like
        Concentration changes in chamber closure period.

    """
    y = p[0] * (1. - np.exp(-t + p[1]))
    return y


def jacobian_conc_func(p, t):
    """
    Calculate the Jacobian matrix of the function of concentration changes.

    Parameters
    ----------
    p : list or array with two elements
        Parameter array
            p[0]: fitted flux
            p[1]: timelag / turnover time
    t : array_like
        Normalized time variable (by the turnover time).

    Returns
    -------
    jac : array_like
        Jacobian matrix with dimension N * 2, where N is the size of `t`.

    """
    jac = np.vstack((1. - np.exp(-t + p[1]), -p[0] * np.exp(-t + p[1]))).T
    return jac


def resid_conc_func(p, t, y):
    """
    Calculate the residuals of fitted concentration changes in chamber closure
    period as a function of time.

    Parameters
    ----------
    p : list or array with two elements
        Parameter array
            p[0]: fitted flux
            p[1]: timelag / turnover time
    t : array_like
        Normalized time variable (by the turnover time).
    y : array_like
        Observations of concentration changes in chamber closure period.

    Returns
    -------
    resid : array_like
        Residuals of fitted concentration changes in chamber closure period.

    """
    resid = p[0] * (1. - np.exp(-t + p[1])) - y
    return resid


def resist_mean(x, IQR_range=1.5):
    """
    Calculate outlier-resistant mean of the sample using Tukey's outlier test.

    Caveat: Does support calculation along an axis, unlike `numpy.mean()`.

    Parameters
    ----------
    x : array_like
        The sample.
    IQR_range : float, optional
        Parameter to control the inlier range defined by
            [ Q_1 - IQR_range * (Q_3 - Q_1), Q_3 - IQR_range * (Q_3 - Q_1) ]
        By default the parameter is 1.5, the original value used by John Tukey.

    Returns
    -------
    x_rmean : float
        The resistant mean of the sample with outliers removed.

    References
    ----------
    .. [T77] John W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.

    """
    x = np.array(x)
    if np.sum(np.isfinite(x)) <= 1:
        return np.nanmean(x)
    else:
        x_q1, x_q3 = np.nanpercentile(x, [25, 75])
        x_iqr = x_q3 - x_q1
        x_uplim = x_q3 + IQR_range * x_iqr
        x_lolim = x_q1 - IQR_range * x_iqr
        x_rmean = np.nanmean(x[(x >= x_lolim) & (x <= x_uplim)])
        return x_rmean


def resist_std(x, IQR_range=1.5):
    """
    Calculate outlier-resistant standard deviation of the sample using
    Tukey's outlier test.

    Caveat: Does support calculation along an axis, unlike `numpy.std()`.

    Parameters
    ----------
    x : array_like
        The sample.
    IQR_range : float, optional
        Parameter to control the inlier range defined by
            [ Q_1 - IQR_range * (Q_3 - Q_1), Q_3 - IQR_range * (Q_3 - Q_1) ]
        By default the parameter is 1.5, the original value used by John Tukey.

    Returns
    -------
    x_rstd : float
        The resistant standard deviation of the sample with outliers removed.
        Degree of freedom = 1 is enforced for the sample standard deviation.

    References
    ----------
    .. [T77] John W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.

    """
    x = np.array(x)
    if np.sum(np.isfinite(x)) <= 1:
        return(np.nanstd(x, ddof=1))
    else:
        x_q1, x_q3 = np.nanpercentile(x, [25, 75])
        x_iqr = x_q3 - x_q1
        x_uplim = x_q3 + IQR_range * x_iqr
        x_lolim = x_q1 - IQR_range * x_iqr
        x_rstd = np.nanstd(x[(x >= x_lolim) & (x <= x_uplim)], ddof=1)
        return x_rstd


def dixon_test(x, left=True, right=True, q_conf='q95'):
    """
    Use Dixon's Q test to identify one or two outliers. The test is based upon
    two assumptions: (1) data must be normally distributed; (2) the test may
    only be used once to a dataset and not repeated.

    Adapted from: <http://sebastianraschka.com/Articles/2014_dixon_test.html>
    (Retrieved 23 Apr 2017).

    Parameters
    ----------
    x : array_like
        Data points. Must be a list or a one dimensional array.
    left : bool, optional
        If True, test the minimum value.
    right : bool, optional
        If True, test the maximum value.
        (At least one of the two, `left` or `right`, must be True.)
    q_conf : str, optional
        Confidence level: 'q95' -- 95% confidence (default). Others supported
        are 'q90' (90% C.I.) and 'q99' (99% C.I.).

    Returns
    -------
    outliers : list
        A list of two values containing the outliers or None. The first element
        corresponds to the minimum value, and the second element corresponds
        to the maximum value. If the tested value is not an outlier, return
        None at its position.

    References
    ----------
    .. [1] Dean, R. B. and Dixon, W. J. (1951). Simplified Statistics for Small
       Numbers of Observations. Anal. Chem., 23(4), 636—638.
    .. [2] Dixon, W. J. (1953). Processing data for outliers Reference.
       J. Biometrics, 9, 74–89.
    .. [3] Rorabacher, D. B. (1991). Statistical Treatment for Rejection of
       Deviant Values: Critical Values of Dixon Q Parameter and Related
       Subrange Ratios at the 95 percent Confidence Level. Anal. Chem., 63(2),
       139–146.

    """
    # critical Q value table
    q_dicts = {'q90': [0.941, 0.765, 0.642, 0.560, 0.507, 0.468, 0.437,
                       0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
                       0.320, 0.313, 0.306, 0.300, 0.295, 0.290, 0.285,
                       0.281, 0.277, 0.273, 0.269, 0.266, 0.263, 0.260],
               'q95': [0.970, 0.829, 0.710, 0.625, 0.568, 0.526, 0.493,
                       0.466, 0.444, 0.426, 0.410, 0.396, 0.384, 0.374,
                       0.365, 0.356, 0.349, 0.342, 0.337, 0.331, 0.326,
                       0.321, 0.317, 0.312, 0.308, 0.305, 0.301, 0.290],
               'q99': [0.994, 0.926, 0.821, 0.740, 0.680, 0.634, 0.598,
                       0.568, 0.542, 0.522, 0.503, 0.488, 0.475, 0.463,
                       0.452, 0.442, 0.433, 0.425, 0.418, 0.411, 0.404,
                       0.399, 0.393, 0.388, 0.384, 0.38, 0.376, 0.372]}
    # cast to numpy array and remove NaNs
    x_arr = np.array(x)
    x_arr = x_arr[np.isfinite(x_arr)]
    # minimum and maximum data sizes allowed
    min_size = 3
    max_size = len(q_dicts[q_conf]) + min_size - 1
    if len(x_arr) < min_size:
        raise ValueError('Sample size too small: ' +
                         'at least %d data points are required' % min_size)
    elif len(x_arr) > max_size:
        raise ValueError('Sample size too large')

    if not (left or right):
        raise ValueError('At least one of the two options, ' +
                         '`left` or `right`, must be True.')

    q_crit = q_dicts[q_conf][len(x_arr) - 3]

    # for small dataset, the built-in `sorted()` is faster than `np.sort()`
    x_sorted = sorted(x_arr)

    x_range = x_sorted[-1] - x_sorted[0]
    if x_range == 0:
        outliers = [None, None]
    else:
        Q_min = abs((x_sorted[1] - x_sorted[0]) / x_range)
        Q_max = abs((x_sorted[-1] - x_sorted[-2]) / x_range)
        outliers = [
            x_sorted[0] if (Q_min > q_crit) and (Q_min >= Q_max) else None,
            x_sorted[-1] if (Q_max > q_crit) and (Q_max >= Q_min) else None]

    return outliers


def IQR_func(x, axis=None):
    """
    Calculate the interquartile range of an array.

    Parameters
    ----------
    x : array_like
        The sample.
    axis : int, optional
        Axis along which the percentiles are computed. Default is to ignore
        and compute the flattened array.
        (Same as the `axis` argument in `numpy.nanpercentile()`.)

    Returns
    -------
    IQR : float or array_like
        The interquartile range of an array.

    """
    if np.sum(np.isfinite(x)) > 0:
        q1, q3 = np.nanpercentile(x, [25., 75.], axis=axis)
        return q3 - q1
    else:
        return np.nan


def p_sat_h2o(temp, ice=False, kelvin=False, method='gg'):
    """
    Calculate saturation vapor pressure over water or ice at a temperature.

    by Wu Sun <wu.sun@ucla.edu>, 14 Sep 2014

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    ice : bool, optional
        Calculate saturation vapor pressure on ice if enabled.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    method : str, optional
        Method used to evaluate saturation vapor pressure.
        'gg': default, Goff-Gratch equation (1946). [GG46]_
        'buck': Buck Research Instruments L.L.C. (1996). [B96]_
        'cimo': CIMO Guide (2008). [WMO]_

    Returns
    -------
    e_sat : float or array_like
        Saturation vapor pressure in Pascal.

    Raises
    ------
    ValueError
        If keyword 'ice' is enabled but temperature is above 0 C or 273.15 K.

    References
    ----------
    .. [GG46] Goff, J. A., and Gratch, S. (1946). Low-­pressure properties of
       water from -160 to 212 F, in Transactions of the American Society of
       Heating and Ventilating Engineers, pp 95-­122, presented at the 52nd
       Annual Meeting of the American Society of Heating and Ventilating
       Engineers, New York.
    .. [B96] Buck Research Instruments L.L.C. (1996). *Buck Research CR-1A
       User's Manual*, Appendix 1.
    .. [WMO] World Meteorological Organization. (2008). *Guide to
       Meteorological Instruments and Methods of Observation*, Appendix 4B,
       WMO-No. 8 (CIMO Guide), Geneva.

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

    >>> print(p_sat_h2o(258.15, kelvin=True, ice=True, method='cimo'))
    165.287132017

    """
    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    if (np.sum(T_k > 273.16) and ice):
        # The triple point of water is 273.16 K
        raise ValueError('Temperature error, no ice exists.')

    if not ice:
        if method == 'buck':
            T_c = T_k - T_0  # temperature in Celsius degree
            e_sat = 6.1121 * np.exp((18.678 - T_c / 234.5) *
                                    T_c / (257.14 + T_c)) * 100
        elif method == 'cimo':
            T_c = T_k - T_0  # temperature in Celsius degree
            e_sat = 6.112 * np.exp(17.62 * T_c / (243.12 + T_c)) * 100
        else:
            # Goff-Gratch equation by default
            u_T = 373.16 / T_k
            v_T = T_k / 373.16
            e_sat = (- 7.90298 * (u_T - 1) + 5.02808 * np.log10(u_T) -
                     1.3816e-7 * (10 ** (11.344 * (1 - v_T)) - 1) +
                     8.1328e-3 * (10 ** (- 3.49149 * (u_T - 1)) - 1) +
                     np.log10(1013.246))
            e_sat = 10 ** e_sat * 100
    else:
        if method == 'buck':
            T_c = T_k - T_0  # temperature in Celsius degree
            e_sat = 6.1115 * np.exp((23.036 - T_c / 333.7) *
                                    T_c / (279.82 + T_c)) * 100
        elif method == 'cimo':
            T_c = T_k - T_0  # temperature in Celsius degree
            e_sat = 6.112 * np.exp(22.46 * T_c / (272.62 + T_c)) * 100
        else:
            # Goff-Gratch equation by default
            u_T = 273.16 / T_k
            v_T = T_k / 273.16
            e_sat = (- 9.09718 * (u_T - 1) - 3.56654 * np.log10(u_T) +
                     0.876793 * (1 - v_T) + np.log10(6.1071))
            e_sat = 10 ** e_sat * 100

    return e_sat


def dew_temp(e_sat, guess=25., kelvin=False, method='gg'):
    """
    Calculate dew temperature from water concentration.

    Parameters
    ----------
    e_sat : float
        Saturation vapor pressure in Pascal. Takes only a single value,
        no array allowed.
    guess : float, optional
        An initial guess for the dew temperature to infer.
    kelvin : bool, optional
        Dew temperature calculated is in Kelvin if enabled.
    method : str, optional
        Method used to evaluate saturation vapor pressure.
        'gg': default, Goff-Gratch equation (1946). [GG46]_
        'buck': Buck Research Instruments L.L.C. (1996). [B96]_
        'cimo': CIMO Guide (2008). [WMO]_

    Returns
    -------
    T_dew : float
        Dew temperature.

    Examples
    --------
    >>> dew_temp(3165)
    24.998963153421172

    >>> dew_temp(610)
    -0.0075798295337097081

    >>> dew_temp(3165, kelvin=True)
    298.14896315342116

    """
    def __e_sat_residual(T, e_sat, kelvin, method):
        return p_sat_h2o(T, kelvin=kelvin, method=method) - e_sat

    if kelvin:
        guess += T_0

    T_dew = optimize.newton(__e_sat_residual, x0=guess,
                            args=(e_sat, kelvin, method))

    return T_dew
