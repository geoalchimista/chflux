import math
from typing import List, Tuple

__all__ = ['parse_concentration_units', 'time_conversion_factor']


def parse_concentration_units(unit_values: List[float]) -> List[
        Tuple[str, str]]:
    """
    Parse concentration and flux units from numerical values of unit prefixes.

    Parameters
    ----------
    unit_values : list of floats
        A list of numerical representation of unit prefixes.

    Returns
    -------
    units : list of pairs
        For each pair in the list, the first element is the concentration unit
        name and the second element is the flux unit name.
    """
    def _convert_unit(unit_value):
        base_unit_conc = 'mol mol$^{-1}$'
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
            prefix = None

        if prefix is not None:
            return prefix + base_unit_conc, prefix + base_unit_flux
        else:
            return 'undefined unit', 'undefined unit'

    return list(map(_convert_unit, unit_values))


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
