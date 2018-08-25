"""
=========================================================
Basic physics and chemistry (:mod:`chflux.core.physchem`)
=========================================================

.. currentmodule:: chflux.core.physchem

.. autosummary::
   :toctree: generated/

   e_sat            -- Saturation vapor pressure of water.
   dew_temp         -- Dew temperature of air.
   convert_flowrate -- Convert flow rate from STP to ambient conditions.
"""
import warnings

import numpy as np
from scipy import optimize

from chflux.core import const

__all__ = ['e_sat', 'dew_temp', 'convert_flowrate']


def e_sat(temp, kelvin: bool = False):
    """
    Calculate saturation vapor pressure of water at a certain temperature.

    Parameters
    ----------
    temp : float or numpy.ndarray
        Temperature, in Celsius degree by default.
    kelvin : bool, optional
        ``temp`` is treated as in Kelvin if ``True``. Default is ``False``.

    Returns
    -------
    e_sat : float or array_like
        Saturation vapor pressure [Pa].

    References
    ----------
    .. [GG46] Goff, J. A., and Gratch, S. (1946). Low-pressure properties of
       water from -160 to 212 F, in Transactions of the American Society of
       Heating and Ventilating Engineers, pp 95--122, presented at the 52nd
       Annual Meeting of the American Society of Heating and Ventilating
       Engineers, New York.

    Examples
    --------
    >>> e_sat(25.)
    3165.1956333836806

    >>> e_sat(np.array([0., 5., 15., 25.]))
    array([  610.33609993,   871.31372986,  1703.28100711,  3165.19563338])

    >>> e_sat(273.15, kelvin=True)
    610.33609993341383
    """
    # force temperature to be in Kelvin
    T_k = temp + (not kelvin) * const.T_0

    # Goff-Gratch equation by default
    u_T = 373.16 / T_k
    v_T = T_k / 373.16
    log10_e_sat = -7.90298 * (u_T - 1.) + 5.02808 * np.log10(u_T) - \
        1.3816e-7 * (10. ** (11.344 * (1. - v_T)) - 1.) + \
        8.1328e-3 * (10. ** (- 3.49149 * (u_T - 1.)) - 1.) + \
        np.log10(1013.246) + 2.  # add 2 to convert from hPa to Pa
    e_sat = 10. ** log10_e_sat

    return e_sat


def dew_temp(e_air: float, guess: float = 25., kelvin: bool = False) -> float:
    """
    Calculate dew temperature of air from water vapor partial pressure.

    Parameters
    ----------
    e_air : float
        Water vapor partial pressure [Pa].
    guess : float, optional
        An initial guess of the dew temperature to be determined. Default is 25
        C or 298.15 K.
    kelvin : bool, optional
        The returned dew temperature is in Kelvin if ``True``. Default is
        ``False``.

    Returns
    -------
    T_dew : float
        Dew temperature.

    Examples
    --------
    >>> dew_temp(3165.2)
    25.000023142197445

    >>> dew_temp(610.3)
    -0.00081395945033536498

    >>> dew_temp(3165.2, kelvin=True)
    298.15002314219754
    """
    def __e_sat_residual(T: float, e_air: float, kelvin: bool):
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


def convert_flowrate(flow_slm, temp, pressure=const.atm, kelvin: bool = False):
    """
    Convert flow rate from STP to the ambient condition of temperature and
    pressure.

    Parameters
    ----------
    flow_slm : float or numpy.ndarray
        Flow rate in standard liter per minute under STP condition.
    temp : float or numpy.ndarray
        Temperature, in Celsius degree by default.
    pressure : float or numpy.ndarray, optional
        Pressure [Pa]. Default is the standard atmospheric pressure.
    kelvin : bool, optional
        ``temp`` is treated as in Kelvin if ``True``. Default is ``False``.

    Returns
    -------
    float or numpy.ndarray
        Flow rate in liter per minute under ambient conditions.

    Examples
    --------
    >>> convert_flowrate(3., 25., 9.7e4)  # 3 SLM, 25 C, 970 hPa
    3.420579918137197
    """
    return flow_slm * (temp / const.T_0 + (not kelvin)) * const.atm / \
        pressure
