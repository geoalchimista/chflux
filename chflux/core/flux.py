r"""
=================================================
Flux calculation module (:mod:`chflux.core.flux`)
=================================================

.. module:: chflux.core.flux

This module contains flux calculation functions.

Curve-fitting functions
=======================

These functions are used by the :func:`nonlinfit` flux calculator.

.. autosummary::
   :toctree: generated/

   conc_fun
   jacobian_conc_fun
   residuals_conc_fun

Flux calculators
================

All flux calculators use the same equation:

:math:`C(t) - C(0) = \dfrac{F\cdot A}{q} [1 - \exp(-q\cdot t / V)]`

where

- :math:`C(t) - C(0)` is the concentration change during the flux measurement
- :math:`F` is the flux to be calculated
- :math:`A` is the area to which the flux is normalized
- :math:`q` is the flow rate passing through the chamber
- :math:`V` is the chamber volume

The only difference is *how* the unknown :math:`F` is determined from different
statistical procedures.

.. autosummary::
   :toctree: generated/

   linfit
   rlinfit
   nonlinfit

Helper functions
================

.. autosummary::
   :toctree: generated/

   remove_baseline
"""
import math
from collections import namedtuple

import numpy as np
from numpy import linalg as LA
from scipy import stats, optimize

__all__ = ['conc_fun', 'jacobian_conc_fun', 'residuals_conc_fun',
           'linfit', 'rlinfit', 'nonlinfit', 'remove_baseline']

LinearFitResults = namedtuple(
    'LinearFitResults',
    ['flux', 'se_flux', 'n_obs', 'conc_fitted', 't_fitted',
     'delta_conc_fitted', 'rmse', 'rvalue', 'slope', 'intercept', 'pvalue',
     'stderr'])

RobustLinearFitResults = namedtuple(
    'RobustLinearFitResults',
    ['flux', 'se_flux', 'n_obs', 'conc_fitted', 't_fitted',
     'delta_conc_fitted', 'rmse', 'rvalue', 'slope', 'intercept',
     'lo_slope', 'up_slope'])

NonlinearFitResults = namedtuple(
    'NonlinearFitResults',
    ['flux', 'se_flux', 'n_obs', 'conc_fitted', 't_fitted',
     'delta_conc_fitted', 'rmse', 'rvalue', 'p0', 'p1', 'se_p0', 'se_p1'])


def conc_fun(p, t):
    r"""
    Calculate the evolution of headspace concentration during chamber closure.

    Parameters
    ----------
    p : array_like
        A collection of two parameters:

        - ``p[0]``: flux [mol m\ :sup:`-2`\  s\ :sup:`-1`\ ] *
          area [m\ :sup:`2`\ ] / flow rate [mol s\ :sup:`-1`\ ]
        - ``p[1]``: timelag [s] / turnover time [s]

    t : array_like
        A nondimensional time variable normalized by the turnover time.

    Returns
    -------
    array_like
        Concentration changes in the chamber closure period.
    """
    return p[0] * (1. - np.exp(-t + p[1]))


def jacobian_conc_fun(p, t):
    r"""
    Calculate the Jacobian matrix of the function of concentration changes.

    Parameters
    ----------
    p : array_like
        A collection of two parameters:

        - ``p[0]``: flux [mol m\ :sup:`-2`\  s\ :sup:`-1`\ ] *
          area [m\ :sup:`2`\ ] / flow rate [mol s\ :sup:`-1`\ ]
        - ``p[1]``: timelag [s] / turnover time [s]

    t : array_like
        A nondimensional time variable normalized by the turnover time.

    Returns
    -------
    array_like
        Jacobian matrix of shape ``(N, 2)``, where ``N`` is the size of ``t``.
    """
    ext = np.exp(-t + p[1])
    return np.vstack((1. - ext, -p[0] * ext)).T


def residuals_conc_fun(p, t, y):
    r"""
    Calculate residuals of fitted concentration changes during chamber closure.

    Parameters
    ----------
    p : array_like
        A collection of two parameters:

        - ``p[0]``: flux [mol m\ :sup:`-2`\  s\ :sup:`-1`\ ] *
          area [m\ :sup:`2`\ ] / flow rate [mol s\ :sup:`-1`\ ]
        - ``p[1]``: timelag [s] / turnover time [s]

    t : array_like
        A nondimensional time variable normalized by the turnover time.
    y : array_like
        Observed concentration changes in the chamber closure period.

    Returns
    -------
    array_like
        Residuals of fitted concentration changes.
    """
    return conc_fun(p, t) - y


def linfit(conc, t, turnover, area, flow):
    r"""
    Calculate the flux using a linear fit.

    Parameters
    ----------
    conc : array
        Concentration changes during chamber closure. Unit is not specified,
        but the calculated flux is reported in a corresponding unit.
    t : array
        Time since the chamber is closed [s].
    turnover : float
        Chamber air turnover time [s] = volume [m\ :sup:`3`\ ] /
        flow rate [m\ :sup:`3`\  s\ :sup:`-1`\ ].
    area : float
        Area to which the flux is normalized [m\ :sup:`2`\ ].
    flow : float
        Molar flow rate of the air passing through the chamber
        [mol s\ :sup:`-1`\ ].

    Returns
    -------
    flux : float
        The flux to be determined. Its unit depends on the unit of ``conc``.
        For example, if CO\ :sub:`2`\  dry mixing ratio is given in ppmv,
        CO\ :sub:`2`\  flux will have the unit of
        µmol m\ :sup:`-2`\  s\ :sup:`-1`\ .
    se_flux : float
        Standard error of the flux. Has the same unit as ``flux``.
    n_obs : int
        Number of valid concentration observations in ``conc``.
    conc_fitted : array
        Fitted concentrations, in the same unit as ``conc``.
    t_fitted : array
        Time index for the fitted concentrations [s].
    delta_conc_fitted : float
        Change of fitted concentration at the end of the measurement.
    rmse : float
        Root-mean-square error of the fitted concentrations.
    rvalue : float
        Pearson correlation coefficient.
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line.
    pvalue : float
        Two-sided *p*-value from testing a null hypothesis that the slope is 0.
    stderr : float
        Standard error of the estimated slope.
    """
    # use only the finite values
    idx_finite = np.isfinite(conc)
    n_obs = idx_finite.sum()
    if n_obs == 0:
        return LinearFitResults(np.nan, np.nan, 0, *[np.nan] * 9)

    # linear regression
    t_fitted = t[idx_finite]
    ys = conc[idx_finite] * flow / area
    xs = np.exp(-t_fitted / turnover)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(xs, ys)

    flux = -slope
    se_flux = stderr
    conc_fitted = (slope * xs + intercept) * area / flow
    delta_conc_fitted = conc_fitted[-1] - conc_fitted[0]
    rmse = LA.norm(conc_fitted - conc[idx_finite]) / math.sqrt(n_obs)

    return LinearFitResults(flux, se_flux, n_obs, conc_fitted, t_fitted,
                            delta_conc_fitted, rmse, rvalue, slope, intercept,
                            pvalue, stderr)


def rlinfit(conc, t, turnover, area, flow):
    r"""
    Calculate the flux using a robust linear fit (Theil–Sen estimator).

    Parameters
    ----------
    conc : array
        Concentration changes during chamber closure. Unit is not specified,
        but the calculated flux is reported in a corresponding unit.
    t : array
        Time since the chamber is closed [s].
    turnover : float
        Chamber air turnover time [s] = volume [m\ :sup:`3`\ ] /
        flow rate [m\ :sup:`3`\  s\ :sup:`-1`\ ].
    area : float
        Area to which the flux is normalized [m\ :sup:`2`\ ].
    flow : float
        Molar flow rate of the air passing through the chamber
        [mol s\ :sup:`-1`\ ].

    Returns
    -------
    flux : float
        The flux to be determined. Its unit depends on the unit of ``conc``.
    se_flux : float
        Standard error of the flux. Has the same unit as ``flux``.
    n_obs : int
        Number of valid concentration observations in ``conc``.
    conc_fitted : array
        Fitted concentrations, in the same unit as ``conc``.
    t_fitted : array
        Time index for the fitted concentrations [s].
    delta_conc_fitted : float
        Change of fitted concentration at the end of the measurement.
    rmse : float
        Root-mean-square error of the fitted concentrations.
    rvalue : float
        Pearson correlation coefficient.
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line.
    lo_slope : float
        Lower bound of the :math:`\pm2\sigma` confidence interval of ``slope``.
    up_slope : float
        Upper bound of the :math:`\pm2\sigma` confidence interval of ``slope``.
    """
    # use only the finite values
    idx_finite = np.isfinite(conc)
    n_obs = idx_finite.sum()
    if n_obs == 0:
        return RobustLinearFitResults(np.nan, np.nan, 0, *[np.nan] * 9)

    # robust linear regression
    t_fitted = t[idx_finite]
    ys = conc[idx_finite] * flow / area
    xs = np.exp(-t_fitted / turnover)
    slope, intercept, lo_slope, up_slope = stats.theilslopes(
        ys, xs, alpha=0.954499736103642)  # CI: plus/minus 2 sigma

    flux = -slope
    se_flux = (up_slope - lo_slope) * 0.25
    conc_fitted = (slope * xs + intercept) * area / flow
    delta_conc_fitted = conc_fitted[-1] - conc_fitted[0]
    rmse = LA.norm(conc_fitted - conc[idx_finite]) / math.sqrt(n_obs)
    rvalue = stats.pearsonr(conc[idx_finite], conc_fitted)[0]

    return RobustLinearFitResults(flux, se_flux, n_obs, conc_fitted, t_fitted,
                                  delta_conc_fitted, rmse, rvalue, slope,
                                  intercept, lo_slope, up_slope)


def nonlinfit(conc, t, turnover, area, flow):
    r"""
    Calculate the flux using nonlinear regression.

    Parameters
    ----------
    conc : array
        Concentration changes during chamber closure. Unit is not specified,
        but the calculated flux is reported in a corresponding unit.
    t : array
        Time since the chamber is closed [s].
    turnover : float
        Chamber air turnover time [s] = volume [m\ :sup:`3`\ ] /
        flow rate [m\ :sup:`3`\  s\ :sup:`-1`\ ].
    area : float
        Area to which the flux is normalized [m\ :sup:`2`\ ].
    flow : float
        Molar flow rate of the air passing through the chamber
        [mol s\ :sup:`-1`\ ].

    Returns
    -------
    flux : float
        The flux to be determined. Its unit depends on the unit of ``conc``.
    se_flux : float
        Standard error of the flux. Has the same unit as ``flux``.
    n_obs : int
        Number of valid concentration observations in ``conc``.
    conc_fitted : array
        Fitted concentrations, in the same unit as ``conc``.
    t_fitted : array
        Time index for the fitted concentrations [s].
    delta_conc_fitted : float
        Change of fitted concentration at the end of the measurement.
    rmse : float
        Root-mean-square error of the fitted concentrations.
    rvalue : float
        Pearson correlation coefficient.
    p0 : float
        Fitted parameter ``p[0]``.
    p1 : float
        Fitted parameter ``p[1]``.
    se_p0 : float
        Standard error of the fitted parameter ``p[0]``.
    se_p1 : float
        Standard error of the fitted parameter ``p[1]``.
    """
    # use only the finite values
    idx_finite = np.isfinite(conc)
    n_obs = idx_finite.sum()
    if n_obs == 0:
        return NonlinearFitResults(np.nan, np.nan, 0, *[np.nan] * 9)

    conc_obs = conc[idx_finite]
    params_guess = [conc_obs[-1] - conc_obs[0], 0.]

    # nonlinear regression
    t_fitted = t[idx_finite]
    t_norm = t_fitted / turnover  # normalized time variable
    nlfit = optimize.least_squares(
        residuals_conc_fun, params_guess,
        bounds=([-np.inf, -1.0], [np.inf, 1.0]),
        loss='soft_l1', f_scale=0.5,
        args=(t_norm, conc_obs))

    conc_fitted = conc_fun(nlfit.x, t_norm)
    delta_conc_fitted = conc_fitted[-1] - conc_fitted[0]
    rmse = LA.norm(conc_fitted - conc_obs) / math.sqrt(n_obs)
    rvalue = stats.pearsonr(conc_obs, conc_fitted)[0]
    p0, p1 = nlfit.x[0], nlfit.x[1]

    # standard errors of estimated parameters
    # 1. `J^T J` is a Gauss-Newton approximation of the negative of the Hessian
    #    of the cost function.
    # 2. The covariance matrix of the parameter estimates is the inverse of the
    #    negative of Hessian matrix evaluated at the parameter estimates.
    neg_hess = np.dot(nlfit.jac.T, nlfit.jac)
    # # debug: check if the hessian is positive definite
    # # print(np.all(LA.eigvals(neg_hess) > 0))
    try:
        inv_neg_hess = LA.inv(neg_hess)
    except LA.LinAlgError:
        try:
            inv_neg_hess = LA.pinv(neg_hess)
        except LA.LinAlgError:
            inv_neg_hess = neg_hess + np.nan
    # calculate covariance matrix of parameter estimates
    MSE = np.nansum(nlfit.fun * nlfit.fun) / (t_fitted.size - 2)
    pcov = inv_neg_hess * MSE

    se_p0 = np.sqrt(pcov[0, 0])
    se_p1 = np.sqrt(pcov[1, 1])

    flux = p0 * flow / area
    se_flux = se_p0 * flow / area

    return NonlinearFitResults(flux, se_flux, n_obs, conc_fitted, t_fitted,
                               delta_conc_fitted, rmse, rvalue,
                               p0, p1, se_p0, se_p1)


def remove_baseline(conc, t, amb, t_amb):
    """
    Correct chamber headspace concentrations for baseline drift and detrend.

    Parameters
    ----------
    conc : array
        Concentrations during chamber closure, uncorrected.
    t : array
        Time since the chamber is closed [s].
    amb : 2-tuple
        Ambient concentrations before and after chamber measurements.
    t_amb : 2-tuple
        Times when the ambient concentrations are measured [s].

    Returns
    -------
    array_like
        Corrected and detrended concentration changes.
    """
    k_bl = (amb[1] - amb[0]) / (t_amb[1] - t_amb[0])  # basline slope
    b_bl = amb[0] - k_bl * t_amb[0]  # baseline intercept
    conc_bl = k_bl * t + b_bl  # baseline concentrations
    return conc - conc_bl
