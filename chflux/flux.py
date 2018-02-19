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
"""
import numpy as _np


__all__ = ['conc_func', 'jacobian_conc_func', 'resid_conc_func']


def conc_func(p, t):
    """
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
    return p[0] * (1. - _np.exp(-t + p[1]))


def jacobian_conc_func(p, t):
    """
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
    ext = _np.exp(-t + p[1])  # a temporary variable
    return _np.vstack((1. - ext, -p[0] * ext)).T


def resid_conc_func(p, t, y):
    """
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
