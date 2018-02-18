"""
=================================================
Basic statistical functions (:mod:`chflux.stats`)
=================================================

.. module:: chflux.stats

This module contains several functions for basic statistical calculations.

List of functions
=================
.. autosummary::
   :toctree: generated/

   interquartile  -- Interquartile range of the sample.
   resist_mean    -- Outlier-resistant mean of the sample.
   resist_std     -- Outlier-resistant standard deviation of the sample.
"""
import numpy as _np


__all__ = ['interquartile', 'resist_mean', 'resist_std']


def interquartile(x, axis=None, retqrt=False):
    """
    Calculate the interquartile range of a sample.

    Parameters
    ----------
    x : array_like
        The sample.
    axis : int, optional
        Axis along which the percentiles are computed. Default is to ignore
        and compute the flattened array.
        (Same as the ``axis`` argument in ``numpy.nanpercentile()``.)
    retqrt : bool, optional
        Return quartiles. If True, return (``iqr``, ``q1``, ``q3``), where
        ``q1`` is the first quartile and ``q3`` is the third quartile.

    Returns
    -------
    iqr : float or array_like
        The interquartile range of a sample.
    q1, q3 : float or array_like, optional
        Only returned if ``retqrt`` is True.

        The first and the third quartiles of a sample.

    Examples
    --------
    >>> interquartile(range(20))
    9.5

    >>> interquartile(range(20), retqrt=True)
    (9.5, 4.75, 14.25)
    """
    if _np.sum(_np.isfinite(x)) > 0:
        q1, q3 = _np.nanpercentile(x, [25., 75.], axis=axis)
        iqr = q3 - q1
        if retqrt:
            return iqr, q1, q3
        else:
            return q3 - q1
    else:
        return _np.nan, _np.nan, _np.nan


def resist_mean(x, inlier_range=1.5):
    """
    Calculate outlier-resistant mean of a sample using Tukey's outlier test.

    Caveat: Does support calculation along an axis.

    Parameters
    ----------
    x : array_like
        The sample.
    inlier_range : float, optional
        Parameter to control the inlier range defined by

        .. math::

            [Q_1 - \\text{inlier_range} \cdot (Q_3 - Q_1), \\
             Q_3 - \\text{inlier_range} \cdot (Q_3 - Q_1)]

        Default value is 1.5.

    Returns
    -------
    rmean : float
        The resistant mean of the sample with outliers removed.

    Examples
    --------
    >>> resist_mean(list(range(20)) + [100, 1000, 10000])
    9.5

    References
    ----------
    .. [T77] John W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.
    """
    x = _np.array(x)
    if _np.sum(_np.isfinite(x)) <= 1:
        return _np.nanmean(x)
    else:
        iqr, q1, q3 = interquartile(x, retqrt=True)
        inlier_uplim = q3 + inlier_range * iqr
        inlier_lolim = q1 - inlier_range * iqr
        rmean = _np.nanmean(x[(x >= inlier_lolim) & (x <= inlier_uplim)])
        return rmean


def resist_std(x, inlier_range=1.5):
    """
    Calculate outlier-resistant standard deviation of a sample using Tukey's
    outlier test.

    Caveat: Does support calculation along an axis.

    Parameters
    ----------
    x : array_like
        The sample.
    inlier_range : float, optional
        Parameter to control the inlier range defined by

        .. math::

            [Q_1 - \\text{inlier_range} \cdot (Q_3 - Q_1), \\
             Q_3 - \\text{inlier_range} \cdot (Q_3 - Q_1)]

        Default value is 1.5.

    Returns
    -------
    rstd : float
        The resistant standard deviation of the sample with outliers removed.
        Degree of freedom = 1 is enforced for the sample standard deviation.

    Examples
    --------
    >>> resist_std(list(range(20)))
    5.9160797830996161

    >>> resist_std(list(range(20)) + [100, 1000, 10000])
    5.9160797830996161

    References
    ----------
    .. [T77] John W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.
    """
    x = _np.array(x)
    if _np.sum(_np.isfinite(x)) <= 1:
        return(_np.nanstd(x, ddof=1))
    else:
        iqr, q1, q3 = interquartile(x, retqrt=True)
        inlier_uplim = q3 + inlier_range * iqr
        inlier_lolim = q1 - inlier_range * iqr
        rstd = _np.nanstd(x[(x >= inlier_lolim) & (x <= inlier_uplim)],
                          ddof=1)
        return rstd
