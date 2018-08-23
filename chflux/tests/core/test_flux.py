import math

import numpy as np
import numpy.linalg as LA
from chflux.core import flux

fco2 = -4.0  # CO2 uptake [µmol m^-2 s^-1]
co2_a = 400.  # ambient CO2 [µmol mol^-1]
tlag = 5  # timelag [s]
turnover = 180  # turnover time [s]
t = np.arange(300)
flow = 0.002043702918883586  # flow rate in mol s^-1, eqv. to 3 L min^-1
area = 3e-2  # leaf area [m^2], eqv. to 300 cm^2


def test_flux():
    # conc_fun
    p = (fco2 * area / flow, 0.)
    t_norm = t / turnover
    dco2 = flux.conc_fun(p, t_norm)  # co2 drawdown
    ss_dco2 = flux.conc_fun(p, np.inf)  # steady state co2 drawdown

    assert abs(flux.conc_fun(p, 0.0)) < 1e-16
    assert (math.isfinite(ss_dco2) and
            math.isclose(ss_dco2, p[0])) or \
        (not math.isfinite(ss_dco2))

    noise = np.random.random(dco2.size) * 10.0 - 5.0
    # noisy data
    dco2_with_noise = dco2 + noise
    # noisy data with missing points
    dco2_with_nan = dco2_with_noise.copy()
    dco2_with_nan[[21, 34, 55, 89, 144, 233]] = np.nan

    # residuals_conc_fun
    assert LA.norm(flux.residuals_conc_fun(p, t_norm, dco2)) < 1e-16
    assert LA.norm(
        flux.residuals_conc_fun(p, t_norm, dco2_with_noise) + noise
    ) < 1e-14 * dco2_with_noise.size

    # linfit
    # test 1: no missing data
    linfit_result1 = flux.linfit(dco2_with_noise, t, turnover, area, flow)
    assert linfit_result1.n_obs == dco2_with_noise.size
    assert math.isclose(linfit_result1.flux, -linfit_result1.slope)
    assert linfit_result1.rmse < 5.0
    assert linfit_result1.rvalue > 0.9
    assert linfit_result1.pvalue < 1e-6
    assert math.isclose(linfit_result1.flux, fco2, rel_tol=0.05)

    # test 2: with missing data
    linfit_result2 = flux.linfit(dco2_with_nan, t, turnover, area, flow)
    assert linfit_result2.n_obs == np.isfinite(dco2_with_nan).sum()
    assert math.isclose(linfit_result2.flux, -linfit_result2.slope)
    assert linfit_result2.rmse < 5.0
    assert linfit_result2.rvalue > 0.9
    assert linfit_result2.pvalue < 1e-6
    assert math.isclose(linfit_result2.flux, fco2, rel_tol=0.05)

    # rlinfit
    # test 1: no missing data
    rlinfit_result1 = flux.rlinfit(dco2_with_noise, t, turnover, area, flow)
    assert rlinfit_result1.n_obs == dco2_with_noise.size
    assert math.isclose(rlinfit_result1.flux, -rlinfit_result1.slope)
    assert rlinfit_result1.rmse < 5.0
    assert rlinfit_result1.rvalue > 0.9
    assert math.isclose(rlinfit_result1.flux, fco2, rel_tol=0.05)
    # test 2: with missing data
    rlinfit_result2 = flux.rlinfit(dco2_with_nan, t, turnover, area, flow)
    assert rlinfit_result2.n_obs == np.isfinite(dco2_with_nan).sum()
    assert rlinfit_result2.rmse < 5.0
    assert rlinfit_result2.rvalue > 0.9
    assert math.isclose(rlinfit_result2.flux, -rlinfit_result2.slope)
    assert math.isclose(rlinfit_result2.flux, fco2, rel_tol=0.05)

    # nonlinfit
    # test 1: no missing data
    nonlinfit_result1 = flux.nonlinfit(dco2_with_noise,
                                       t, turnover, area, flow)
    assert nonlinfit_result1.n_obs == dco2_with_noise.size
    assert nonlinfit_result1.rmse < 5.0
    assert nonlinfit_result1.rvalue > 0.9
    assert math.isclose(nonlinfit_result1.flux, fco2, rel_tol=0.05)
    # test 2: with missing data
    nonlinfit_result2 = flux.nonlinfit(dco2_with_nan, t, turnover, area, flow)
    assert nonlinfit_result2.n_obs == np.isfinite(dco2_with_nan).sum()
    assert nonlinfit_result2.rmse < 5.0
    assert nonlinfit_result2.rvalue > 0.9
    assert math.isclose(nonlinfit_result2.flux, fco2, rel_tol=0.05)

    # remove_baseline
    amb = (400, 406)  # assume a drift rate of 0.02 µmol mol^-1 s^-1
    t_amb = (0, 300)
    co2_with_drift = co2_a + dco2_with_noise + 0.02 * t
    dco2_corrected = flux.remove_baseline(co2_with_drift, t, amb, t_amb)
    assert LA.norm(dco2_corrected - dco2_with_noise) < 1e-14 * t.size
