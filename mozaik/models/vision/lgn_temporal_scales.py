"""Temporal RF scale policies, independent of position and eccentricity.

Select ``lognormal``, ``uniform`` or ``custom`` with temporal_scale_mode.
To try another distribution, edit custom() below and select ``custom``.
Each function returns one positive time-dilation factor per cell.
"""

import numpy as np
from scipy.special import ndtri

# Experimental source: Cai, DeAngelis & Freeman (1997), Fig. 5C, adult LGN
# temporal response durations. https://doi.org/10.1152/jn.1997.78.2.1045
# The data are digitized histogram bin centres and frequencies. Duration D
# is the width of the temporal response's Hilbert envelope at 1/e of its peak.
# Fit a log-normal duration distribution: log(D) ~ Normal(mu_D, sigma**2).
# Fix mu_D = log(79.6 ms), using the reported population geometric mean, and
# estimate sigma by maximizing the histogram-weighted log-likelihood of the
# bin probabilities. Each bin probability is the fitted CDF difference between
# its edges; omitted zero-height bins are restored before fitting.
#
# Convert duration to a dimensionless temporal dilation using scale = D / D0,
# where D0 = 67.25 ms is the reference RF's duration measured in the same way.
# Therefore log(scale) ~ Normal(log(79.6 / 67.25), sigma**2).
# Sampling is conditioned on the central 99% of the parent distribution, a
# model choice implemented with inverse-CDF sampling rather than clipping.
CANONICAL_LOGNORMAL_PARAMETERS = {
    "mu": 0.1685970743683889,
    "sigma": 0.3059765581600333,
    "lower_quantile": 0.005,
    "upper_quantile": 0.995,
}


def lognormal(number, rng):
    """Current canonical policy: independent, truncated log-normal scales."""
    p = CANONICAL_LOGNORMAL_PARAMETERS
    quantiles = rng.uniform(p["lower_quantile"], p["upper_quantile"], size=number)
    return np.exp(p["mu"] + p["sigma"] * ndtri(quantiles))


def uniform(number, rng):
    """The same temporal RF for every cell: all scales are one."""
    return np.ones(number)


def custom(number, rng):
    """Edit this function to define your own temporal-scale distribution.

    Initially identical to uniform(). Use the supplied rng for random draws,
    e.g. rng.uniform(0.5, 1.5, size=number), to keep runs reproducible.
    Return an array of length number, containing only positive finite scales.
    The function is called separately for ON and OFF populations.
    """
    return np.ones(number)


def sample(number, rng, policy):
    """Choose the temporal policy. No spatial information enters this switch."""
    if policy == "lognormal":
        scales = lognormal(number, rng)
    elif policy == "uniform":
        scales = uniform(number, rng)
    elif policy == "custom":
        scales = custom(number, rng)
    else:
        raise ValueError(
            "temporal_scale_mode must be 'lognormal', 'uniform' or 'custom'"
        )

    scales = np.asarray(scales, dtype=float)
    if scales.shape != (number,) or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError(
            "Temporal policy must return one positive finite scale per cell"
        )
    return scales
