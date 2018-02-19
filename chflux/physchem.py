"""
====================================================
Basic physics and chemistry (:mod:`chflux.physchem`)
====================================================

.. module:: chflux.physchem

This module contains several functions for basic physics and chemistry.

.. autosummary::
   :toctree: generated/

   e_sat     -- Saturation vapor pressure of water.
   dew_temp  -- Dew temperature of air.
"""
import numpy as _np
from scipy import optimize as _optimize

from chflux import const


__all__ = ['e_sat', 'dew_temp']


# by Wu Sun <wu.sun@ucla.edu>, 14 Sep 2014
def e_sat(temp, ice=False, kelvin=False, method='gg'):
    """
    Calculate saturation vapor pressure over water or ice at a temperature.

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    ice : bool, optional
        Calculate saturation vapor pressure on ice if enabled.
    kelvin : bool, optional
        Temperature input is taken in the Kelvin scale if enabled.
    method : str, optional
        Method used to evaluate saturation vapor pressure.

        - ``'gg'``: default, Goff-Gratch equation (1946). [GG46]_
        - ``'buck'``: Buck Research Instruments L.L.C. (1996). [B96]_
        - ``'cimo'``: CIMO Guide (2008). [WMO]_

    Returns
    -------
    e_sat : float or array_like
        Saturation vapor pressure in Pascal.

    Raises
    ------
    ValueError
        If keyword ``ice`` is enabled but ``temp`` is above 0.01 C or 273.16 K.

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
    >>> print(e_sat(25))
    3165.19563338

    >>> print(e_sat([0, 5, 15, 25]))
    [  610.33609993   871.31372986  1703.28100711  3165.19563338]

    >>> print(e_sat(25, method='buck'))
    3168.53141228

    >>> print(e_sat(273.15, kelvin=True))
    610.336099933

    >>> print(e_sat(-15, ice=True))
    165.014773924

    >>> print(e_sat(258.15, kelvin=True, ice=True, method='cimo'))
    165.287132017
    """
    T_k = _np.array(temp, dtype='d') + (not kelvin) * const.T_0
    # force temperature to be in Kelvin

    if (_np.sum(T_k > 273.16) and ice):
        # The triple point of water is 273.16 K
        raise ValueError('Temperature error; no ice exists.')

    if not ice:
        if method == 'buck':
            T_c = T_k - const.T_0  # temperature in Celsius degree
            e_sat = 6.1121 * _np.exp((18.678 - T_c / 234.5) *
                                     T_c / (257.14 + T_c)) * 100.
        elif method == 'cimo':
            T_c = T_k - const.T_0  # temperature in Celsius degree
            e_sat = 6.112 * _np.exp(17.62 * T_c / (243.12 + T_c)) * 100.
        else:
            # Goff-Gratch equation by default
            u_T = 373.16 / T_k
            v_T = T_k / 373.16
            e_sat = (- 7.90298 * (u_T - 1.) + 5.02808 * _np.log10(u_T) -
                     1.3816e-7 * (10. ** (11.344 * (1. - v_T)) - 1.) +
                     8.1328e-3 * (10. ** (- 3.49149 * (u_T - 1.)) - 1.) +
                     _np.log10(1013.246))
            e_sat = 10. ** e_sat * 100.
    else:
        if method == 'buck':
            T_c = T_k - const.T_0  # temperature in Celsius degree
            e_sat = 6.1115 * _np.exp((23.036 - T_c / 333.7) *
                                     T_c / (279.82 + T_c)) * 100.
        elif method == 'cimo':
            T_c = T_k - const.T_0  # temperature in Celsius degree
            e_sat = 6.112 * _np.exp(22.46 * T_c / (272.62 + T_c)) * 100.
        else:
            # Goff-Gratch equation by default
            u_T = 273.16 / T_k
            v_T = T_k / 273.16
            e_sat = (- 9.09718 * (u_T - 1.) - 3.56654 * _np.log10(u_T) +
                     0.876793 * (1. - v_T) + _np.log10(6.1071))
            e_sat = 10. ** e_sat * 100.

    return e_sat


def dew_temp(e_air, guess=25., kelvin=False, method='gg'):
    """
    Calculate dew temperature of air from water vapor partial pressure.

    Parameters
    ----------
    e_air : float
        Saturation vapor pressure in Pascal.
    guess : float, optional
        An initial guess of the dew temperature to be determined.
    kelvin : bool, optional
        Dew temperature returned is in Kelvin if enabled.
    method : str, optional
        Method used to evaluate saturation vapor pressure.

        - ``'gg'``: default, Goff-Gratch equation (1946). [GG46]_
        - ``'buck'``: Buck Research Instruments L.L.C. (1996). [B96]_
        - ``'cimo'``: CIMO Guide (2008). [WMO]_

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
    def __e_sat_residual(T, e_air, kelvin, method):
        return e_sat(T, kelvin=kelvin, method=method) - e_air

    if kelvin:
        guess += const.T_0

    try:
        T_dew = _optimize.newton(__e_sat_residual, x0=guess,
                                 args=(e_air, kelvin, method))
    except RuntimeError:
        T_dew = _np.nan

    return T_dew
