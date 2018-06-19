import math

from chflux.core import physchem


def test_physchem():
    # test e_sat
    assert math.isclose(physchem.e_sat(25.), 3165.0, rel_tol=1e-3)
    assert math.isclose(physchem.e_sat(0.), 610.0, rel_tol=1e-3)

    # test dew_temp
    assert math.isclose(physchem.dew_temp(3165.0, kelvin=True),
                        298.15, rel_tol=1e-4)
    assert math.isclose(physchem.dew_temp(610.0, kelvin=True),
                        273.15, rel_tol=1e-4)

    # test convert_flowrate
    assert math.isclose(physchem.convert_flowrate(5., 0.), 5.)  # tautology
    assert math.isclose(physchem.convert_flowrate(5., 25.),
                        5. * 298.15 / 273.15)
    assert math.isclose(physchem.convert_flowrate(5., 0., 8e4),
                        5. * 1.01325e5 / 8e4)
