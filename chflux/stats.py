"""
=================================================
Basic statistical functions (:mod:`chflux.stats`)
=================================================

.. module:: chflux.stats

This module contains several functions for basic statistical calculations.

.. autosummary::
   :toctree: generated/

   interquartile -- Interquartile range of the sample.
   resist_mean   -- Outlier-resistant mean of the sample.
   resist_std    -- Outlier-resistant standard deviation of the sample.
   dixon_test    -- Use Dixon's Q test to identify low and high outliers.
"""
import numpy as np


__all__ = ['interquartile', 'resist_mean', 'resist_std', 'dixon_test']


def interquartile(x, axis=None, retqrt=False):
    """
    Calculate the interquartile range of a sample.

    Parameters
    ----------
    x : array_like
        The sample.
    axis : int, optional
        Axis along which the percentiles are computed. Default is to ignore
        and compute the flattened array. (Same as the ``axis`` argument in
        ``numpy.nanpercentile()``.)
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
    if np.sum(np.isfinite(x)) > 0:
        q1, q3 = np.nanpercentile(x, [25., 75.], axis=axis)
        iqr = q3 - q1
        if retqrt:
            return iqr, q1, q3
        else:
            return q3 - q1
    else:
        return np.nan, np.nan, np.nan


def resist_mean(x, inlier_range=1.5):
    r"""
    Calculate outlier-resistant mean of a sample using Tukey's outlier test.

    Note: If the sample size is too small for outlier filtering (i.e., *n* <
    4), the outlier-resistant mean is equal to the mean.

    Parameters
    ----------
    x : array_like
        The sample.
    inlier_range : float, optional
        Parameter to control the inlier range defined by

        .. math::

            [Q_1 - \text{inlier_range} \cdot (Q_3 - Q_1),
             Q_3 + \text{inlier_range} \cdot (Q_3 - Q_1)]

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
    x = np.array(x)
    if np.sum(np.isfinite(x)) <= 3:
        return np.nanmean(x)
    else:
        iqr, q1, q3 = interquartile(x, retqrt=True)
        inlier_uplim = q3 + inlier_range * iqr
        inlier_lolim = q1 - inlier_range * iqr
        rmean = np.nanmean(x[(x >= inlier_lolim) & (x <= inlier_uplim)])
        return rmean


def resist_std(x, inlier_range=1.5):
    r"""
    Calculate outlier-resistant standard deviation of a sample using Tukey's
    outlier test.

    Note: If the sample size is too small for outlier filtering (i.e., *n* <
    4), the outlier-resistant standard deviation is equal to the standard
    deviation.

    Parameters
    ----------
    x : array_like
        The sample.
    inlier_range : float, optional
        Parameter to control the inlier range defined by

        .. math::

            [Q_1 - \text{inlier_range} \cdot (Q_3 - Q_1),
             Q_3 + \text{inlier_range} \cdot (Q_3 - Q_1)]

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
    x = np.array(x)
    if np.sum(np.isfinite(x)) <= 3:
        return(np.nanstd(x, ddof=1))
    else:
        iqr, q1, q3 = interquartile(x, retqrt=True)
        inlier_uplim = q3 + inlier_range * iqr
        inlier_lolim = q1 - inlier_range * iqr
        rstd = np.nanstd(x[(x >= inlier_lolim) & (x <= inlier_uplim)],
                         ddof=1)
        return rstd


def dixon_test(x, left=True, right=True, q_conf='q95'):
    """
    Use Dixon's Q test to identify low and high outliers. The test is based
    upon two assumptions: (1) data must be normally distributed; (2) the test
    may only be used once to a dataset and not repeated.

    Adapted from: <http://sebastianraschka.com/Articles/2014_dixon_test.html>
    (Retrieved 23 Apr 2017).

    Parameters
    ----------
    x : array_like
        Data points. Must be a list or a one dimensional array.
    left : bool, optional
        If True, test the minimum value.
    right : bool, optional
        If True, test the maximum value.
        (At least one of the two, ``left`` or ``right``, must be True.)
    q_conf : str, optional
        Confidence level: ``'q95'`` -- 95% confidence interval (default).
        Others options are ``'q90'`` (90% C.I.) and ``'q99'`` (99% C.I.).

    Returns
    -------
    outliers : list
        A list of two values containing the outliers or None. The first element
        corresponds to the minimum value, and the second element corresponds
        to the maximum value. If the tested value is not an outlier, return
        ``None`` at its position.

    References
    ----------
    .. [1] Dean, R. B. and Dixon, W. J. (1951). Simplified Statistics for Small
       Numbers of Observations. Anal. Chem., 23(4), 636—638.
    .. [2] Dixon, W. J. (1953). Processing data for outliers Reference.
       J. Biometrics, 9, 74–89.
    .. [3] Rorabacher, D. B. (1991). Statistical Treatment for Rejection of
       Deviant Values: Critical Values of Dixon Q Parameter and Related
       Subrange Ratios at the 95 percent Confidence Level. Anal. Chem., 63(2),
       139–146.

    Examples
    --------
    >>> dixon_test([1.1, 2.0, 10., 3., 4.5, 5.2, 3.1, 2.3])
    [None, 10.0]
    """
    # critical Q value table
    q_dicts = {'q90': [0.941, 0.765, 0.642, 0.560, 0.507, 0.468, 0.437,
                       0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
                       0.320, 0.313, 0.306, 0.300, 0.295, 0.290, 0.285,
                       0.281, 0.277, 0.273, 0.269, 0.266, 0.263, 0.260],
               'q95': [0.970, 0.829, 0.710, 0.625, 0.568, 0.526, 0.493,
                       0.466, 0.444, 0.426, 0.410, 0.396, 0.384, 0.374,
                       0.365, 0.356, 0.349, 0.342, 0.337, 0.331, 0.326,
                       0.321, 0.317, 0.312, 0.308, 0.305, 0.301, 0.290],
               'q99': [0.994, 0.926, 0.821, 0.740, 0.680, 0.634, 0.598,
                       0.568, 0.542, 0.522, 0.503, 0.488, 0.475, 0.463,
                       0.452, 0.442, 0.433, 0.425, 0.418, 0.411, 0.404,
                       0.399, 0.393, 0.388, 0.384, 0.38, 0.376, 0.372]}
    # cast to numpy array and remove NaNs
    x_arr = np.array(x)
    x_arr = x_arr[np.isfinite(x_arr)]
    # minimum and maximum data sizes allowed
    min_size = 3
    max_size = len(q_dicts[q_conf]) + min_size - 1
    if len(x_arr) < min_size:
        raise ValueError('Sample size too small: ' +
                         'at least %d data points are required.' % min_size)
    elif len(x_arr) > max_size:
        raise ValueError('Sample size too large.')

    if not (left or right):
        raise ValueError('At least one of the two options, ' +
                         '`left` or `right`, must be True.')

    q_crit = q_dicts[q_conf][len(x_arr) - 3]

    # for small dataset, the built-in `sorted()` is faster than `np.sort()`
    x_sorted = sorted(x_arr)

    x_range = x_sorted[-1] - x_sorted[0]
    if x_range == 0:
        outliers = [None, None]
    else:
        Q_min = abs((x_sorted[1] - x_sorted[0]) / x_range)
        Q_max = abs((x_sorted[-1] - x_sorted[-2]) / x_range)
        outliers = [
            x_sorted[0] if (Q_min > q_crit) and (Q_min >= Q_max) else None,
            x_sorted[-1] if (Q_max > q_crit) and (Q_max >= Q_min) else None]

    return outliers
