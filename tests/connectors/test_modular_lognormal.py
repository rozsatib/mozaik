import numpy as np

from mozaik.connectors.modular import _lognormal_event_weight_sums


class ConstantDistribution:
    def __init__(self, value):
        self.value = value

    def copy(self, seed):
        return self

    def next(self, n):
        return np.zeros(n) + self.value


def test_lognormal_event_weight_sums_preserves_base_weight_with_zero_sigma():
    weights = _lognormal_event_weight_sums(
        ConstantDistribution(0.25),
        sigma=0.0,
        counts=[1, 2, 4],
        seed=11,
    )

    np.testing.assert_allclose(weights, [0.25, 0.5, 1.0])


def test_lognormal_event_weight_sums_has_requested_mean():
    base_weight = 0.25
    weights = _lognormal_event_weight_sums(
        ConstantDistribution(base_weight),
        sigma=1.0,
        counts=np.ones(100000, dtype=int),
        seed=17,
    )

    np.testing.assert_allclose(np.mean(weights), base_weight, rtol=0.02)


def test_lognormal_event_weight_sums_has_lognormal_shape():
    base_weight = 0.25
    sigma = 0.75
    weights = _lognormal_event_weight_sums(
        ConstantDistribution(base_weight),
        sigma=sigma,
        counts=np.ones(100000, dtype=int),
        seed=23,
    )

    log_factors = np.log(weights / base_weight)
    np.testing.assert_allclose(np.mean(log_factors), -0.5 * sigma * sigma, atol=0.01)
    np.testing.assert_allclose(np.std(log_factors), sigma, atol=0.01)


def test_lognormal_event_weight_sums_wraps_large_seed():
    weights = _lognormal_event_weight_sums(
        ConstantDistribution(0.25),
        sigma=1.0,
        counts=[1, 2, 4],
        seed=2**32 - 2,
    )

    assert len(weights) == 3
    assert np.all(weights > 0)
