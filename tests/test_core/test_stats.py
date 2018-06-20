import numpy as np
from scipy.special import erfinv

from chflux.core import stats


def test_stats():
    mu = 5.
    sigma = 10.

    np.random.seed(0xbeefcafe)
    data_1 = np.random.normal(mu, sigma, 10_000)

    np.random.seed(0xdeadbeef)
    data_2 = np.random.normal(mu, sigma, 10_000)

    # sprinkle 5% outliers on one side of the distribution
    data_2[0:500] += np.random.uniform(50., 100., 500)
    np.random.shuffle(data_2)

    # test interquartile
    z_qrt = erfinv(0.5) * np.sqrt(2)  # the theoretical z-score for q3

    assert np.isscalar(stats.interquartile(data_1))
    iqr_res_1 = stats.interquartile(data_1, ret_qrt=True)
    assert len(iqr_res_1) == 3
    assert np.isclose((iqr_res_1[1] - mu) / sigma, -z_qrt, atol=0.01)
    assert np.isclose((iqr_res_1[2] - mu) / sigma, z_qrt, atol=0.01)

    # test resist_mean
    assert abs(stats.resist_mean(data_2) - mu) < abs(np.nanmean(data_2) - mu)

    # test resist_std
    assert abs(stats.resist_std(data_2) - sigma) < abs(
        np.nanstd(data_2, ddof=1) - sigma)

    # test dixon_test
    assert stats.dixon_test([1.0, 2.0, 100.0]) == [None, 100.0]
    assert stats.dixon_test([2.0, 100.0, 123.0, 115.0, 130.0]) == [2.0, None]
