import numpy as np
from collections import OrderedDict


def meshgrid3D(x, y, z):
    r"""A slimmed-down version of http://www.scipy.org/scipy/numpy/attachment/ticket/966/meshgrid.py"""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    mult_fact = np.ones((len(x), len(y), len(z)))
    nax = np.newaxis
    return x[:, nax, nax] * mult_fact, \
           y[nax, :, nax] * mult_fact, \
           z[nax, nax, :] * mult_fact

def stRF_kernel_2d(duration=200.0, dt=1000.0/120.0, size=10.0,
                   scale_factor=10.0, p=OrderedDict()):
    r"""
    scale_factor = pixel/degree
    """
    degree_per_pixel = 1/float(scale_factor)
    #p = RF_parameters()
    x = np.arange(-size/2. + degree_per_pixel/2,
                     size/2. + degree_per_pixel/2,
                     degree_per_pixel)
    t = np.arange(0.0, duration, dt)
    xm, ym, tm = meshgrid3D(x, x, t)
    kernel = stRF_2d(xm, ym, tm, p)
    return kernel


def stRF_2d(x, y, t, p):
    r"""
    x, y, and t should all be 3D arrays, produced by meshgrid3D.
    If we need to optimize, it would be quicker to do G() on a 1D t array and
    F_2d() on 2D x and y arrays, and then multiply them, as Jens did in his
    original implementation.
    Timing gives 0.44 s for Jens' implementation, and 2.9 s for this one.
    
    """

    fcm, tmc, fsm, tms = _stRF_2d_components(x, y, t, p)

    # Linear Receptive Field
    #rf = (fcm*tmc - fsm*tms)/(fcm - fsm).max()
    rf = (fcm*tmc - fsm*tms)

    x_res = x[1,0,0] - x[0,0,0]
    _assert_spatial_component_areas(fcm[:,:,0], fsm[:,:,0], x_res, p)

    if p.subtract_mean:
        for i in range(0,np.shape(rf)[2]): # lets normalize each time slice separately
            rf[:,:,i] = rf[:,:,i] - rf[:,:,i].mean()
        #rf = rf - rf.mean()
    return rf


def _stable_gamma_component(t, K, c, t0, n):
    t = np.asarray(t, dtype=float)
    u = c * (t - t0)
    out = np.zeros_like(t)

    valid = (u > 0) & (K > 0)
    if np.any(valid):
        log_p = np.log(K) + n * np.log(u[valid]) - u[valid] - n * np.log(n) + n
        out[valid] = np.exp(log_p)

    return out


def G(t, K1, K2, c1, c2, t1, t2, n1, n2):
    p1 = _stable_gamma_component(t, K1, c1, t1, n1)
    p2 = _stable_gamma_component(t, K2, c2, t2, n2)
    return p1 - p2


def F_2d(x, y, A, sigma):
    return A * np.exp(-(x**2 + y**2) / (2*sigma**2))


def _stRF_temporal_components(t, p):
    return (
        G(t, p.K1, p.K2, p.c1, p.c2, p.t1, p.t2, p.n1, p.n2),
        G(t - p.td, p.K1, p.K2, p.c1, p.c2, p.t1, p.t2, p.n1, p.n2),
    )


def _stRF_2d_components(x, y, t, p):
    tmc, tms = _stRF_temporal_components(t, p)
    fcm = F_2d(x, y, p.Ac, p.sigma_c)
    fsm = F_2d(x, y, p.As, p.sigma_s)
    return fcm, tmc, fsm, tms


def _assert_spatial_component_areas(fcm, fsm, spatial_resolution, p):
    for name, component, amplitude, sigma in (
        ("center", fcm, p.Ac, p.sigma_c),
        ("surround", fsm, p.As, p.sigma_s),
    ):
        component_sum = (
            np.prod([axis.sum() for axis in component])
            if isinstance(component, tuple)
            else component.sum()
        )
        actual = component_sum * spatial_resolution**2
        expected = 2 * np.pi * sigma**2 * amplitude
        error = abs(actual - expected)
        assert error / max(actual, expected) < 0.5, (
            "Synthesized %s of RF doesn't fit the supplied sigma and amplitude "
            "(%f-%f=%f), check visual field size and model size!"
            % (name, actual, expected, error)
        )


def stRF_2d_factorize(x, y, t, p):
    r"""Return spatial/temporal factors for :func:`stRF_2d`.

    ``x`` and ``y`` are the one-dimensional coordinates supplied as the first
    and second spatial arguments to ``stRF_2d``. The returned spatial arrays
    therefore follow the same first-axis/second-axis ordering as the dense
    callable.
    """

    tmc, tms = _stRF_temporal_components(t, p)
    fcm = (
        F_2d(np.asarray(x), 0.0, p.Ac, p.sigma_c),
        F_2d(np.asarray(y), 0.0, 1.0, p.sigma_c),
    )
    fsm = (
        F_2d(np.asarray(x), 0.0, p.As, p.sigma_s),
        F_2d(np.asarray(y), 0.0, 1.0, p.sigma_s),
    )
    _assert_spatial_component_areas(fcm, fsm, x[1] - x[0], p)

    factors = [(fcm, tmc), ((-fsm[0], fsm[1]), tms)]
    if p.subtract_mean:
        ones = np.ones_like(fcm[1])
        for spatial, temporal, mean in (
            (fcm, tmc, -fcm[0].mean() * fcm[1].mean()),
            (fsm, tms, fsm[0].mean() * fsm[1].mean()),
        ):
            factors.append(((np.full_like(spatial[0], mean), ones), temporal))
    return tuple(factors)


stRF_2d.factorize = stRF_2d_factorize


def dog_optimal_spatial_frequency(Ac, As, sigma_c, sigma_s):
    r"""Return the nonzero optimum of the Cai97 spatial DoG.

    ``F_2d`` uses peak Gaussian amplitudes and conventional standard
    deviations.  Consequently the returned frequency is in cycles per unit of
    ``sigma_c`` and ``sigma_s`` (cycles/degree when the sigmas are in degrees).
    """

    values = {
        "Ac": Ac,
        "As": As,
        "sigma_c": sigma_c,
        "sigma_s": sigma_s,
    }
    for name, value in values.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be a positive finite scalar" % name) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be a positive finite scalar" % name)
        values[name] = value

    Ac = values["Ac"]
    As = values["As"]
    sigma_c = values["sigma_c"]
    sigma_s = values["sigma_s"]
    if sigma_c == sigma_s:
        raise ValueError("sigma_c and sigma_s must be unequal")

    frequency_squared = (
        np.log(As)
        - np.log(Ac)
        + 4.0 * (np.log(sigma_s) - np.log(sigma_c))
    ) / (2.0 * np.pi**2 * (sigma_s**2 - sigma_c**2))
    if not np.isfinite(frequency_squared) or frequency_squared <= 0.0:
        raise ValueError(
            "Cai97 DoG parameters do not define a positive real nonzero optimum"
        )
    return float(np.sqrt(frequency_squared))
