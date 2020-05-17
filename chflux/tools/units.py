import math
from typing import Tuple

__all__ = ['parse_concentration_units', 'time_conversion_factor']


def parse_concentration_units(unit_value: float) -> Tuple[str, str]:
    """
    Parse concentration and flux units from a numerical value of unit prefix.

    Parameters
    ----------
    unit_value : float
        A numerical representation of unit prefix, e.g., ``1e-6`` for "micro-".

    Returns
    -------
    units : 2-tuple of str
        A pair of strings where the first is the concentration unit and the
        second is the flux unit in LaTeX representation.
    """
    base_unit_concentration = 'mol mol$^{-1}$'
    base_unit_flux = 'mol m$^{-2}$ s$^{-1}$'
    if math.isclose(unit_value, 1e-15):
        prefix = 'f'
    elif math.isclose(unit_value, 1e-12):
        prefix = 'p'
    elif math.isclose(unit_value, 1e-9):
        prefix = 'n'
    elif math.isclose(unit_value, 1e-6):
        prefix = '$\\mu$'
    elif math.isclose(unit_value, 1e-3):
        prefix = 'm'
    elif math.isclose(unit_value, 1):
        prefix = ''
    else:
        return 'undefined unit', 'undefined unit'

    return prefix + base_unit_concentration, prefix + base_unit_flux


def time_conversion_factor(unit: str) -> float:
    """Return a multiplier that converts a given unit of time to day(s)."""
    if unit in ["seconds", "second", "sec", "s"]:
        conv_fac = 1. / (60. * 60. * 24.)
    elif unit in ["minutes", "minute", "min", "m"]:
        conv_fac = 1. / (60. * 24.)
    elif unit in ["hours", "hour", "hr", "h"]:
        conv_fac = 1. / 24.
    else:
        conv_fac = 1.0
    return conv_fac
