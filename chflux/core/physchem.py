"""
=========================================================
Basic physics and chemistry (:mod:`chflux.core.physchem`)
=========================================================

.. module:: chflux.core.physchem

This module contains several functions for basic physics and chemistry.

.. autosummary::
   :toctree: generated/

   e_sat            -- Saturation vapor pressure of water.
   dew_temp         -- Dew temperature of air.
   convert_flowrate -- Convert flow rate from STP to ambient conditions.
"""
import warnings

import numpy as np
from scipy import optimize

from chflux.lib import const


__all__ = ['e_sat', 'dew_temp', 'convert_flowrate']


def e_sat(temp, kelvin=False):
    """
    Calculate saturation vapor pressure of water at a certain temperature.

    Parameters
    ----------
    temp : float or numpy.ndarray
        Temperature, in Celsius degree by default.
    kelvin : bool, optional
        `temp` input is treated as in Kelvin if True. Default is False.

    Returns
    -------
    e_sat : float or array_like
        Saturation vapor pressure in Pascal.

    References
    ----------
    .. [GG46] Goff, J. A., and Gratch, S. (1946). Low-­pressure properties of
       water from -160 to 212 F, in Transactions of the American Society of
       Heating and Ventilating Engineers, pp 95-­122, presented at the 52nd
       Annual Meeting of the American Society of Heating and Ventilating
       Engineers, New York.

    Examples
    --------
    >>> print(e_sat(25.))
    3165.19563338

    >>> print(e_sat(np.array([0., 5., 15., 25.])))
    [  610.33609993   871.31372986  1703.28100711  3165.19563338]

    >>> print(e_sat(273.15, kelvin=True))
    610.336099933
    """
    # force temperature to be in Kelvin
    T_k = temp + (not kelvin) * const.T_0

    # Goff-Gratch equation by default
    u_T = 373.16 / T_k
    v_T = T_k / 373.16
    log10_esat = -7.90298 * (u_T - 1.) + 5.02808 * np.log10(u_T) - \
        1.3816e-7 * (10. ** (11.344 * (1. - v_T)) - 1.) + \
        8.1328e-3 * (10. ** (- 3.49149 * (u_T - 1.)) - 1.) + \
        np.log10(1013.246) + 2.  # add 2 to convert from hPa to Pa
    e_sat = 10. ** log10_esat

    return e_sat


def dew_temp(e_air, guess=25., kelvin=False):
    """
    Calculate dew temperature of air from water vapor partial pressure.

    Parameters
    ----------
    e_air : float
        Water vapor partial pressure in Pascal.
    guess : float, optional
        An initial guess of the dew temperature to be determined.
    kelvin : bool, optional
        The returned dew temperature is in Kelvin if True. Default is False.

    Returns
    -------
    T_dew : float
        Dew temperature.

    Examples
    --------
    >>> print('%f' % dew_temp(3165.2))
    25.000023

    >>> print('%f' % dew_temp(610.3))
    -0.000814

    >>> print('%f' % dew_temp(3165.2, kelvin=True))
    298.150023
    """
    def __e_sat_residual(T, e_air, kelvin):
        return e_sat(T, kelvin=kelvin) - e_air

    if kelvin:
        guess += const.T_0

    try:
        T_dew = optimize.newton(__e_sat_residual, x0=guess,
                                args=(e_air, kelvin))
    except RuntimeError:
        warnings.warn('Dew temperature does not converge!', RuntimeWarning)
        T_dew = np.nan

    return T_dew


def convert_flowrate(flow_slpm, temp, pressure=const.p_std, kelvin=False):
    """
    Convert flow rate from STP to the ambient condition of temperature and
    pressure.

    Parameters
    ----------
    flow_slpm : float or numpy.ndarray
        Flow rate in standard liter per minute under STP condition.
    temp : float or numpy.ndarray
        Temperature, in Celsius degree by default.
    pressure : float or numpy.ndarray, optional
        Pressure [Pa]. Default is standard atmospheric pressure.
    kelvin : bool, optional
        `temp` input is treated as in Kelvin if True. Default is False.

    Returns
    -------
    float or numpy.ndarray
        Flow rate in liter per minute under ambient condition.

    Examples
    --------
    >>> print('%f' % convert_flowrate(3., 25., 9.7e4)) # 3 SLPM, 25 C, 970 hPa
    3.420580
    """
    return flow_slpm * (temp / const.T_0 + (not kelvin)) * const.p_std / \
        pressure
