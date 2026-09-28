import numpy as np
import properscoring as ps
from scipy.stats import norm

from heatcast import climatology as clim
from heatcast import metrics as M
from heatcast.probability import (LEVELS, cdf_from_quantiles, exceed_prob,
                                  quantiles_from_cdf_grid, quantiles_from_samples)

CRIT = {
    "plains": {"tmax_min": 40.0, "hw_dep": 4.5, "severe_dep": 6.5, "hw_abs": 45.0, "severe_abs": 47.0},
    "coastal": {"tmax_min": 37.0, "hw_dep": 4.5, "severe_dep": 6.5, "hw_abs": None, "severe_abs": None},
}


def test_plains_thresholds_follow_imd_rules():
    hw, sev = clim.thresholds(np.array([38.0, 41.0, 33.0]), "plains", CRIT)
    assert np.allclose(hw, [42.5, 45.0, 40.0])     # departure rule, absolute 45 cap, 40 floor
    assert np.allclose(sev, [44.5, 47.0, 40.0])


def test_coastal_thresholds():
    hw, sev = clim.thresholds(np.array([32.0, 34.0]), "coastal", CRIT)
    assert np.allclose(hw, [37.0, 38.5])
    assert np.allclose(sev, [38.5, 40.5])


def test_labels():
    hw, sev = clim.thresholds(np.full(4, 38.0), "plains", CRIT)
    cls = clim.label(np.array([42.4, 42.5, 44.5, np.nan]), hw, sev)
    assert cls.tolist() == [0, 1, 2, -1]


def test_doy365_leap_year():
    import pandas as pd
    d = pd.to_datetime(["2024-02-28", "2024-02-29", "2024-03-01", "2023-03-01"])
    assert clim.doy365(d).tolist() == [59, 59, 60, 60]


def test_exceedance_matches_gaussian():
    q = norm.ppf(LEVELS, loc=40, scale=2)[None]
    for x in (37.0, 40.0, 42.0, 43.5):
        assert abs(exceed_prob(x, q)[0] - norm.sf(x, 40, 2)) < 0.01
    # beyond the 0.99 quantile the exponential tail keeps decreasing
    far = exceed_prob(np.array([45.0, 46.0, 48.0]), np.repeat(q, 3, 0))
    assert np.all(np.diff(far) < 0) and far[-1] >= 0


def test_cdf_monotone():
    q = np.sort(np.random.default_rng(0).normal(38, 3, (1, 21)), -1)
    xs = np.linspace(25, 50, 200)
    F = cdf_from_quantiles(xs, np.repeat(q, 200, 0))
    assert np.all(np.diff(F) >= -1e-9)


def test_crps_close_to_closed_form():
    q = norm.ppf(LEVELS, loc=40, scale=2)
    for y in (36.0, 40.0, 43.0):
        approx = M.crps_from_quantiles(q, np.array(y))
        exact = ps.crps_gaussian(y, 40, 2)
        assert abs(approx - exact) / exact < 0.08


def test_regrid_quantiles_roundtrip():
    lv9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q9 = norm.ppf(lv9, 40, 2)[None]
    q21 = quantiles_from_cdf_grid(q9, lv9)[0]
    inner = (LEVELS >= 0.1) & (LEVELS <= 0.9)
    assert np.allclose(q21[inner], norm.ppf(LEVELS[inner], 40, 2), atol=0.1)
    assert np.all(np.diff(q21) > 0)


def test_samples_to_quantiles():
    smp = np.random.default_rng(1).normal(40, 2, (1, 7, 5000))
    q = quantiles_from_samples(smp)
    assert q.shape == (1, 7, 21)
    assert abs(q[0, 0, 10] - 40) < 0.15


def test_contingency():
    c = M.contingency(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (c["hits"], c["misses"], c["false_alarms"]) == (1, 1, 1)
    assert abs(c["CSI"] - 1 / 3) < 1e-9
