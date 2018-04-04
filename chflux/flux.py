"""
=====================================================
Functions for calculating fluxes (:mod:`chflux.flux`)
=====================================================

.. module:: chflux.flux

This module contains functions used for calculating fluxes.

Curve-fitting functions
=======================
.. autosummary::
   :toctree: generated/

   conc_func
   jacobian_conc_func
   resid_conc_func

Flux calculators
================
.. autosummary::
   :toctree: generated/

   fit_flux_lin
   fit_flux_rlin
   fit_flux_nonlin
"""
import math
from collections import namedtuple

import numpy as np
from numpy import linalg as LA
from scipy import stats


__all__ = ['conc_func', 'jacobian_conc_func', 'resid_conc_func',
           'fit_flux_lin', 'fit_flux_rlin', 'fit_flux_nonlin']


FluxLinFitResults = namedtuple(
    'FluxLinFitResults',
    ['flux', 'se_flux', 'n_obs', 'conc_fitted', 't_fitted',
     'delta_conc_fitted', 'rmse', 'slope', 'intercept', 'rvalue', 'pvalue',
     'stderr'])

FluxNonlinFitResults = namedtuple(
    'FluxNonlinFitResults',
    ['flux', 'se_flux', 'n_obs', 'conc_fitted', 't_fitted',
     'delta_conc_fitted', 'rmse', 'p0', 'p1', 'se_p0', 'se_p1'])


def conc_func(p, t):
    r"""
    Calculate concentration changes as a function of time during a chamber
    closure period.

    Parameters
    ----------
    p : list or array
        A parameter array with exactly two elements:

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


def jacobian_conc_func(p, t):
    r"""
    Calculate the Jacobian matrix of the function of concentration changes.

    Parameters
    ----------
    p : list or array
        A parameter array with exactly two elements:

            - ``p[0]``: flux [mol m\ :sup:`-2`\  s\ :sup:`-1`\ ] *
              area [m\ :sup:`2`\ ] / flow rate [mol s\ :sup:`-1`\ ]
            - ``p[1]``: timelag [s] / turnover time [s]

    t : array_like
        A nondimensional time variable normalized by the turnover time.

    Returns
    -------
    array_like
        Jacobian matrix with the shape (N, 2), where N is the size of ``t``.
    """
    ext = np.exp(-t + p[1])  # a temporary variable
    return np.vstack((1. - ext, -p[0] * ext)).T


def resid_conc_func(p, t, y):
    r"""
    Calculate the residuals of fitted concentration changes during the chamber
    closure period.

    Parameters
    ----------
    p : list or array
        A parameter array with exactly two elements:

            - ``p[0]``: flux [mol m\ :sup:`-2`\  s\ :sup:`-1`\ ] *
              area [m\ :sup:`2`\ ] / flow rate [mol s\ :sup:`-1`\ ]
            - ``p[1]``: timelag [s] / turnover time [s]

    t : array_like
        A nondimensional time variable normalized by the turnover time.
    y : array_like
        Observations of concentration changes in chamber closure period.

    Returns
    -------
    array_like
        Residuals of fitted concentration changes.
    """
    return conc_func(p, t) - y


def fit_flux_lin(conc, t, t_turnover, area, flow):
    r"""
    Calculate flux from concentration changes and other chamber parameters,
    using linear regression.

    [EQN TO BE ADDED]  @TODO

    Parameters
    ----------
    conc : array
        Concentration changes during the chamber closure period. Unit can be
        arbitrary as long as the flux is reported in a corresponding unit.
        For example, if CO\ :sub:`2`\  concentration (more precisely, dry
        mixing ratio) is given in ppmv, then the calculated CO\ :sub:`2`\  flux
        has the unit of µmol m\ :sup:`-2`\  s\ :sup:`-1`\ .
    t : array
        Time [s] since the chamber is closed.
    t_turnover : float
        Chamber air turnover time [s] = volume [m\ :sup:`3`\ ] /
        flow rate [m\ :sup:`3`\  s\ :sup:`-1`\ ].
    area : float
        Area [m\ :sup:`2`\ ] to which the flux is normalized, for example,
        leaf area or chamber footprint area.
    flow : float
        Flow rate of the air passing through the chamber [mol s\ :sup:`-1`\ ].
        Note the unit is converted for convenience in calculations.

    Returns
    -------
    flux : float
        The flux to be determined. Its unit depends on the unit of ``conc``.
    se_flux : float
        Standard error of the flux. Its unit is the same as ``flux``.
    n_obs : int
        Number of valid concentration observations in ``conc``.
    conc_fitted : array
        Fitted concentrations, with the same unit as ``conc``.
    t_fitted : array
        Time [s] index for the fitted concentrations.
    delta_conc_fitted : float
        Change of fitted concentration at the end of the measurement.
    rmse : float
        Root-mean-square error of the fitted concentrations.
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line.
    rvalue : float
        Pearson correlation coefficient.
    pvalue : float
        Two-sided *p*-value from testing a null hypothesis that the slope is 0.
    stderr : float
        Standard error of the estimated slope.
    """
    # extract finite values as a boolean index
    idx_finite = np.isfinite(conc)
    # number of valid observations
    n_obs = idx_finite.sum()
    if n_obs == 0:
        return None  # @NOTE: placeholder; return everything with nan values

    # time index for the fitted concentrations
    t_fitted = t[idx_finite]
    # prepare x and y variables for linear regression
    y_fit = conc[idx_finite] * flow / area
    x_fit = np.exp(-t_fitted / t_turnover)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x_fit, y_fit)

    flux = -slope
    se_flux = stderr
    conc_fitted = slope * x_fit + intercept
    delta_conc_fitted = conc_fitted[-1] - conc_fitted[0]
    rmse = LA.norm(conc_fitted - conc[idx_finite]) / math.sqrt(n_obs)

    return FluxLinFitResults(flux, se_flux, n_obs, conc_fitted, t_fitted,
                             delta_conc_fitted, rmse, slope, intercept,
                             rvalue, pvalue, stderr)


def fit_flux_rlin(conc, t, t_turnover, area, flow):
    r"""
    Calculate flux from concentration changes and other chamber parameters,
    using robust linear regression.

    [EQN TO BE ADDED]  @TODO

    Parameters
    ----------
    conc : array
        Concentration changes during the chamber closure period. Unit can be
        arbitrary as long as the flux is reported in a corresponding unit.
        For example, if CO\ :sub:`2`\  concentration (more precisely, dry
        mixing ratio) is given in ppmv, then the calculated CO\ :sub:`2`\  flux
        has the unit of µmol m\ :sup:`-2`\  s\ :sup:`-1`\ .
    t : array
        Time [s] since the chamber is closed.
    t_turnover : float
        Chamber air turnover time [s] = volume [m\ :sup:`3`\ ] /
        flow rate [m\ :sup:`3`\  s\ :sup:`-1`\ ].
    area : float
        Area [m\ :sup:`2`\ ] to which the flux is normalized, for example,
        leaf area or chamber footprint area.
    flow : float
        Flow rate of the air passing through the chamber [mol s\ :sup:`-1`\ ].
        Note the unit is converted for convenience in calculations.

    Returns
    -------
    flux : float
        The flux to be determined. Its unit depends on the unit of ``conc``.
    se_flux : float
        Standard error of the flux. Its unit is the same as ``flux``.
    n_obs : int
        Number of valid concentration observations in ``conc``.
    conc_fitted : array
        Fitted concentrations, with the same unit as ``conc``.
    t_fitted : array
        Time [s] index for the fitted concentrations.
    delta_conc_fitted : float
        Change of fitted concentration at the end of the measurement.
    rmse : float
        Root-mean-square error of the fitted concentrations.
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line.
    rvalue : float
        Pearson correlation coefficient.
    pvalue : float
        Two-sided *p*-value from testing a null hypothesis that the slope is 0.
    stderr : float
        Standard error of the estimated slope.
    """
    pass


def fit_flux_nonlin(conc, t, t_turnover, area, flow):
    r"""
    Calculate flux from concentration changes and other chamber parameters,
    using nonlinear regression.

    [EQN TO BE ADDED]  @TODO

    Parameters
    ----------
    conc : array
        Concentration changes during the chamber closure period. Unit can be
        arbitrary as long as the flux is reported in a corresponding unit.
        For example, if CO\ :sub:`2`\  concentration (more precisely, dry
        mixing ratio) is given in ppmv, then the calculated CO\ :sub:`2`\  flux
        has the unit of µmol m\ :sup:`-2`\  s\ :sup:`-1`\ .
    t : array
        Time [s] since the chamber is closed.
    t_turnover : float
        Chamber air turnover time [s] = volume [m\ :sup:`3`\ ] /
        flow rate [m\ :sup:`3`\  s\ :sup:`-1`\ ].
    area : float
        Area [m\ :sup:`2`\ ] to which the flux is normalized, for example,
        leaf area or chamber footprint area.
    flow : float
        Flow rate of the air passing through the chamber [mol s\ :sup:`-1`\ ].
        Note the unit is converted for convenience in calculations.

    Returns
    -------
    flux : float
        The flux to be determined. Its unit depends on the unit of ``conc``.
    se_flux : float
        Standard error of the flux. Its unit is the same as ``flux``.
    n_obs : int
        Number of valid concentration observations in ``conc``.
    conc_fitted : array
        Fitted concentrations, with the same unit as ``conc``.
    t_fitted : array
        Time [s] index for the fitted concentrations.
    delta_conc_fitted : float
        Change of fitted concentration at the end of the measurement.
    rmse : float
        Root-mean-square error of the fitted concentrations.
    p0 : float
        Fitted parameter ``p[0]``.
    p1 : float
        Fitted parameter ``p[1]``.
    se_p0 : float
        Standard error of the fitted parameter ``p[0]``.
    se_p1 : float
        Standard error of the fitted parameter ``p[1]``.
    """
    pass


def label_chamber_period():
    pass


def extract_chamber_period():
    pass


def correct_baseline():
    pass


def average_metvar(df_orig, df_dest, id, vars, std=False, iqr=False):
    """
    id : str
        The column name for subsetting segments for averaging.
    vars : list or dict
        List or dict of variables to be averaged. If list, names do not change
        in ``df_dest``; if dict, name in df_dest will be specified by key
        values.
    std : bool, optional
        Return standard deviations if True. Variable names will be `sd_*`.
    iqr : bool, optional
        Return interquartile ranges if True. Variable names will be `iqr_*`.
    """
    pass


def calculate_flux(df, method='all'):
    pass


def predict_conc(df, flux):
    """Predict the fitted concentrations."""
    pass


# @NOTE: maybe this is not needed; I'll see
def dixon_test_flux(fluxes):
    pass
