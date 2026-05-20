import math

import numpy as np

from mozaik.tools.misc import (
    find_neuron,
    normal_function,
    sample_from_bin_distribution,
)


def test_sample_from_bin_distribution_returns_reproducible_indices():
    bins = np.array([1.0, 3.0, 6.0])

    samples_a = sample_from_bin_distribution(bins, 12, seed=7)
    samples_b = sample_from_bin_distribution(bins, 12, seed=7)

    assert len(samples_a) == 12
    assert np.array_equal(samples_a, samples_b)
    assert np.all(samples_a >= 0)
    assert np.all(samples_a < len(bins))


def test_sample_from_bin_distribution_handles_empty_bins():
    assert sample_from_bin_distribution(np.array([]), 5, seed=3) == []


def test_normal_function_matches_standard_normal_peak():
    value = normal_function(np.array([0.0]), mean=0.0, sigma=1.0)[0]

    np.testing.assert_approx_equal(
        value, 1.0 / math.sqrt(2.0 * math.pi), significant=12
    )


def test_normal_function_is_symmetric_around_mean():
    x = np.array([-0.5, 0.5])
    values = normal_function(x, mean=0.0, sigma=2.0)

    np.testing.assert_allclose(values[0], values[1])


def test_find_neuron_returns_expected_named_positions():
    positions = np.array(
        [
            [-1.0, 1.0, -1.0, 1.0, 0.0],
            [1.0, 1.0, -1.0, -1.0, 0.0],
        ]
    )

    assert find_neuron("top_left", positions) == 0
    assert find_neuron("top_right", positions) == 1
    assert find_neuron("bottom_left", positions) == 2
    assert find_neuron("bottom_right", positions) == 3
    assert find_neuron("center", positions) == 4
