import math

from chflux.core import const


def test_const():
    assert math.isclose(const.T_0, 273.15)
    assert math.isclose(const.R_gas, 8.3145, rel_tol=1e-3)
    assert math.isclose(const.atm, 101325.0)
    assert math.isclose(const.air_concentration, 44.615, rel_tol=1e-3)
