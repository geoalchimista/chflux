"""
Common functions used in flux calculation

(c) Wu Sun <wu.sun@ucla.edu> 2016-2017

"""
from collections import namedtuple
import warnings

import numpy as np
from scipy import optimize
import scipy.constants.constants as sci_const
import pandas as pd
import yaml


# Physical constants
# Do not modify unless you are on a differnt planet or in a different universe.
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


def chamber_lookup_table_func(doy, chamber_config=None):
    """
    Return a chamber meta information look-up table.

    """
    # define returned data template
    ChamberLookupTableResult = namedtuple(
        'ChamberLookupTableResult',
        ['schedule_start', 'schedule_end', 'n_ch', 'smpl_cycle_len',
         'n_cycle_per_day', 'unit_of_time', 'df'])

    if chamber_config is None:
        with open('chamber.yaml', 'r') as fo:
            chamber_config = yaml.load(fo)

    for key in chamber_config:
        if (chamber_config[key]['schedule_start'] <= doy <
                chamber_config[key]['schedule_end']):
            current_schedule = chamber_config[key]
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
                'flowmeter_no', 'TC_no', 'PAR_no',
                'ch_start', 'ch_o_b', 'ch_cls', 'ch_o_a',
                'ch_end', 'ch_atm_a']:
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
    df[['ch_start', 'ch_o_b', 'ch_cls', 'ch_o_a', 'ch_end', 'ch_atm_a']] /= \
        time_unit_conversion_factor

    chamber_lookup_table['df'] = df

    # convert to a namedtuple
    # `**` is the 'splat operator' for unpacking dictionaries
    chamber_lookup_table = ChamberLookupTableResult(**chamber_lookup_table)

    return chamber_lookup_table


def timelag_optmz_func():
    # @TODO: to add this function
    return None


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
    x_rstd : float or array_like
        The resistant standard deviation of the sample with outliers removed.

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
