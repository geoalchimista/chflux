"""A collection of tools for unit conversions and representations."""
import math

__all__ = ['parse_units']


def parse_units(unit_values):
    """
    Parse concentration and flux units from numerical values.

    Parameters
    ----------
    unit_values : list of float
        A list of numerical representation of units.

    Returns
    -------
    units : list of tuple pairs
        For each tuple pair in the list, the first element is the concentration
        unit name and the second element is the flux unit name.
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
