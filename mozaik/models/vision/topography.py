"""Analytical LGN topography for the eccentricity-dependent visual mode.

LGN positions use Cartesian visual-field coordinates in degrees.  The density
fit is an area density with respect to ``dx dy``; it contains no polar
Jacobian.  The receptive-field fit returns the conventional Gaussian standard
deviation used by :func:`mozaik.models.vision.cai97.F_2d`; its sampled
counterpart adds the measured Gaussian scatter of that fit, so RF size is
log-normal about the line rather than a deterministic function of eccentricity.

The density fit is empirically poorly constrained above approximately 60
degrees, and the RF-size fit was reported below 25 degrees.  Larger supported
eccentricities are extrapolated without clipping and are reported once when
the provider is initialized.
"""

import numbers

import numpy
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

import mozaik

logger = mozaik.getMozaikLogger()

_DENSITY_CENTRAL_AMPLITUDE = 200.54919
_DENSITY_CENTRAL_SCALE_DEG = 2.0738426
_DENSITY_PERIPHERAL_AMPLITUDE = 55.730403
_DENSITY_PERIPHERAL_SCALE_DEG = 16.692502

_FULL_CORTICAL_AREA_MM2 = 760.0
_CENTRAL_AREAL_MAGNIFICATION_MM2_PER_DEG2 = 3.6
_FULL_MAP_ANGLE_RAD = 2.0 * numpy.pi
_MAX_SUPPORTED_ECCENTRICITY_DEG = 90.0
_VISUAL_FIELD_CENTER_TOLERANCE_DEG = 1e-12
_DENSITY_EMPIRICAL_LIMIT_DEG = 60.0
_RF_SIZE_EMPIRICAL_LIMIT_DEG = 25.0

# Pooled residual standard deviation of the Linsenmeier et al. (1982) cat
# X-cell semi-log regression of RF centre size on eccentricity, refitted on the
# 46 digitized points below 25 degrees (44 degrees of freedom).  Units are
# log10 degrees; it is deliberately constant across eccentricity.
_RF_CENTER_SIZE_LOG10_RESIDUAL_SD = 0.093765

# Half-width, in residual standard deviations, of the symmetric interval the
# centre-size scatter is drawn from.  Truncation bounds the RF-size support, so
# the smallest and largest centre sigma a configuration can produce are known
# before any cell is drawn.  The derived visual-space resolution follows from
# that bound and is therefore independent of seed and population size.
_RF_CENTER_SIZE_TRUNCATION_SD = 3.0

_TOPOGRAPHY_PARAMETER_NAMES = {
    "cap_eccentricity",
    "full_max_eccentricity",
    "beta",
}


def _finite_array(value, name):
    try:
        array = numpy.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not numpy.all(numpy.isfinite(array)):
        raise ValueError(f"{name} must contain finite real values")
    return array


def _nonnegative_eccentricity(eccentricity_deg):
    eccentricity = _finite_array(eccentricity_deg, "eccentricity_deg")
    if numpy.any(eccentricity < 0.0):
        raise ValueError("eccentricity_deg must be nonnegative")
    return eccentricity


def _positive_cap(cap_eccentricity_deg):
    if (
        isinstance(cap_eccentricity_deg, bool)
        or not isinstance(cap_eccentricity_deg, numbers.Real)
        or not numpy.isfinite(cap_eccentricity_deg)
        or cap_eccentricity_deg <= 0.0
    ):
        raise ValueError("cap_eccentricity_deg must be positive and finite")
    return float(cap_eccentricity_deg)


def _scalar_or_array(value):
    value = numpy.asarray(value)
    if value.ndim == 0:
        return float(value)
    return value


def relative_lgn_cell_density_at_eccentricity(eccentricity_deg):
    """Return the uncapped radial LGN density fit.

    Parameters
    ----------
    eccentricity_deg : float or array-like
        Nonnegative visual eccentricity in degrees.

    Returns
    -------
    float or numpy.ndarray
        Relative density, nominally in cells per square visual degree.
    """

    eccentricity = _nonnegative_eccentricity(eccentricity_deg)
    density = _DENSITY_CENTRAL_AMPLITUDE * numpy.exp(
        -eccentricity / _DENSITY_CENTRAL_SCALE_DEG
    ) + _DENSITY_PERIPHERAL_AMPLITUDE * numpy.exp(
        -eccentricity / _DENSITY_PERIPHERAL_SCALE_DEG
    )
    return _scalar_or_array(density)


def relative_lgn_cell_density(x_deg, y_deg):
    """Return uncapped LGN area-density weights at Cartesian visual positions."""

    x = _finite_array(x_deg, "x_deg")
    y = _finite_array(y_deg, "y_deg")
    try:
        x, y = numpy.broadcast_arrays(x, y)
    except ValueError as exc:
        raise ValueError("x_deg and y_deg must be broadcast-compatible") from exc
    return relative_lgn_cell_density_at_eccentricity(numpy.hypot(x, y))


def capped_relative_lgn_cell_density_at_eccentricity(
    eccentricity_deg, cap_eccentricity_deg
):
    """Return the radial LGN density fit plateaued inside an explicit cap."""

    eccentricity = _nonnegative_eccentricity(eccentricity_deg)
    cap = _positive_cap(cap_eccentricity_deg)
    return relative_lgn_cell_density_at_eccentricity(numpy.maximum(eccentricity, cap))


def capped_relative_lgn_cell_density(x_deg, y_deg, cap_eccentricity_deg):
    """Return explicitly capped density weights at Cartesian visual positions."""

    x = _finite_array(x_deg, "x_deg")
    y = _finite_array(y_deg, "y_deg")
    try:
        x, y = numpy.broadcast_arrays(x, y)
    except ValueError as exc:
        raise ValueError("x_deg and y_deg must be broadcast-compatible") from exc
    return capped_relative_lgn_cell_density_at_eccentricity(
        numpy.hypot(x, y), cap_eccentricity_deg
    )


def rf_center_sigma(eccentricity_deg, cap_eccentricity_deg=None):
    """Return the eccentricity-dependent Cai97 centre sigma in visual degrees."""

    eccentricity = _nonnegative_eccentricity(eccentricity_deg)
    if cap_eccentricity_deg is not None:
        cap = _positive_cap(cap_eccentricity_deg)
        eccentricity = numpy.maximum(eccentricity, cap)
    sigma_deg = numpy.power(10.0, 0.014 * eccentricity - 0.758) / numpy.sqrt(2.0)
    if not numpy.all(numpy.isfinite(sigma_deg)):
        raise ValueError("eccentricity_deg is outside the numerically supported range")
    return _scalar_or_array(sigma_deg)


def _log10_residual_sd(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not numpy.isfinite(value)
        or value < 0.0
    ):
        raise ValueError("log10_residual_sd must be nonnegative and finite")
    return float(value)


def _truncation_sd(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not numpy.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("truncation_sd must be positive and finite")
    return float(value)


def rf_center_sigma_scatter_factors(
    log10_residual_sd=_RF_CENTER_SIZE_LOG10_RESIDUAL_SD,
    truncation_sd=_RF_CENTER_SIZE_TRUNCATION_SD,
):
    """Return the smallest and largest multiplicative scatter factors.

    The scatter is drawn from a normal residual truncated symmetrically at
    ``+/- truncation_sd`` standard deviations, so the factor applied to the
    regression line is confined to
    ``[10**(-k*s), 10**(+k*s)]``.  Both ends are needed up front: the lower
    factor fixes the finest centre sigma a configuration can produce and hence
    the derived stimulus resolution, and the upper factor fixes the largest
    receptive-field support and hence the worst-case kernel size.
    """

    residual_sd = _log10_residual_sd(log10_residual_sd)
    half_width = _truncation_sd(truncation_sd) * residual_sd
    return float(numpy.power(10.0, -half_width)), float(numpy.power(10.0, half_width))


def sample_rf_center_sigma(
    eccentricity_deg,
    rng,
    log10_residual_sd=_RF_CENTER_SIZE_LOG10_RESIDUAL_SD,
    cap_eccentricity_deg=None,
    truncation_sd=_RF_CENTER_SIZE_TRUNCATION_SD,
):
    """Draw Cai97 centre sigmas scattered about the RF-size regression line.

    The digitized Linsenmeier et al. (1982) cat X cells scatter around the
    semi-log fit with a Gaussian residual in ``log10`` units,

        log10(r_c) = a + b * eccentricity + N(0, log10_residual_sd**2),

    so ``r_c`` -- and with it ``sigma_c = r_c / sqrt(2)`` -- is log-normal
    about the line.  The residual standard deviation is pooled over the fitted
    range and does not vary with eccentricity, so the multiplicative scatter is
    identical at every eccentricity.  This is what lets the population reach
    preferred spatial frequencies above the deterministic foveal optimum.

    The residual is drawn from that normal conditioned on
    ``|residual| <= truncation_sd * log10_residual_sd``, by inverting the
    normal CDF on the corresponding quantile interval.  Conditioning rather
    than clipping keeps the drawn density continuous: clipping would place a
    point mass of identical cells at each bound, and the lower bound is exactly
    the highest-preferred-frequency end of the population.

    ``log10_residual_sd=0.0`` reproduces :func:`rf_center_sigma` exactly.
    """

    residual_sd = _log10_residual_sd(log10_residual_sd)
    truncation = _truncation_sd(truncation_sd)
    mean_sigma_deg = rf_center_sigma(eccentricity_deg, cap_eccentricity_deg)
    upper_quantile = float(ndtr(truncation))
    quantiles = rng.uniform(
        1.0 - upper_quantile, upper_quantile, size=numpy.shape(mean_sigma_deg)
    )
    residual = residual_sd * ndtri(quantiles)
    sigma_deg = mean_sigma_deg * numpy.power(10.0, residual)
    if not numpy.all(numpy.isfinite(sigma_deg)) or numpy.any(sigma_deg <= 0.0):
        raise ValueError("sampled centre sigma is outside the supported range")
    return _scalar_or_array(sigma_deg)


def _finite_real(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not numpy.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _empirical_mapping_cap_eccentricity_deg(full_max_eccentricity_deg, beta):
    """Resolve the empirical mapping cap needed by the unified cap policy."""

    target = _FULL_CORTICAL_AREA_MM2 / (
        _FULL_MAP_ANGLE_RAD
        * _CENTRAL_AREAL_MAGNIFICATION_MM2_PER_DEG2
        * full_max_eccentricity_deg**2
    )
    maximum_integral = 0.5
    if target > maximum_integral:
        raise ValueError(
            "full_max_eccentricity is incompatible with the fixed cortical "
            "calibration: the empirical cap equation has no solution in (0, 1]"
        )

    def residual(q):
        return q**2 / 2.0 + q**beta / (2.0 - beta) * (1.0 - q ** (2.0 - beta)) - target

    q = brentq(
        residual,
        0.0,
        1.0,
        xtol=1e-14,
        rtol=4.0 * numpy.finfo(float).eps,
    )
    return float(q * full_max_eccentricity_deg)


def _visual_field_value(visual_field, name):
    if not hasattr(visual_field, name):
        raise ValueError(
            f"visual_field.{name} is required; eccentricity mode currently "
            "requires a fixation-centred visual rectangle"
        )
    try:
        value = float(getattr(visual_field, name))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"visual_field.{name} must be finite; eccentricity mode currently "
            "requires a fixation-centred visual rectangle"
        ) from exc
    if not numpy.isfinite(value):
        raise ValueError(
            f"visual_field.{name} must be finite; eccentricity mode currently "
            "requires a fixation-centred visual rectangle"
        )
    return value


def _topography_parameters(parameters):
    if not hasattr(parameters, "keys"):
        raise TypeError(
            "topography parameters must supply cap_eccentricity, "
            "full_max_eccentricity, and beta"
        )
    actual_names = set(parameters.keys())
    if actual_names != _TOPOGRAPHY_PARAMETER_NAMES:
        missing = sorted(_TOPOGRAPHY_PARAMETER_NAMES - actual_names)
        unexpected = sorted(actual_names - _TOPOGRAPHY_PARAMETER_NAMES)
        raise KeyError(
            "Invalid topography parameter keys; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return (
        parameters["cap_eccentricity"],
        parameters["full_max_eccentricity"],
        parameters["beta"],
    )


class RadiallySymmetricLGNTopography:
    """Immutable provider for eccentricity-dependent LGN properties.

    The visual domain is a fixation-centred rectangle, while modelled LGN RF
    centres occupy a disk with radius ``max_eccentricity_deg`` no larger than
    the rectangle's inscribed disk.
    An explicit cap plateaus both density and centre sigma.  ``None`` leaves
    these retinal functions uncapped; an independently resolved empirical cap
    is retained for cortical mapping.
    """

    __slots__ = (
        "max_eccentricity_deg",
        "user_cap_eccentricity_deg",
        "_full_max_eccentricity_deg",
        "_beta",
        "_empirical_mapping_cap_eccentricity_deg",
        "_resolved_mapping_cap_eccentricity_deg",
        "_frozen",
    )

    def __init__(self, visual_field, parameters, maximum_eccentricity_deg=None):
        location_x = _visual_field_value(visual_field, "location_x")
        location_y = _visual_field_value(visual_field, "location_y")
        size_x = _visual_field_value(visual_field, "size_x")
        size_y = _visual_field_value(visual_field, "size_y")

        if (
            abs(location_x) > _VISUAL_FIELD_CENTER_TOLERANCE_DEG
            or abs(location_y) > _VISUAL_FIELD_CENTER_TOLERANCE_DEG
        ):
            raise ValueError(
                "visual_field.location_x and visual_field.location_y must be "
                "zero within 1e-12 degrees; eccentricity mode currently "
                "requires a fixation-centred visual rectangle"
            )
        if size_x <= 0.0 or size_y <= 0.0:
            raise ValueError(
                "visual_field.size_x and visual_field.size_y must be positive "
                "and finite; eccentricity mode currently requires a "
                "fixation-centred visual rectangle"
            )

        visual_field_radius_deg = min(size_x, size_y) / 2.0
        if maximum_eccentricity_deg is None:
            max_eccentricity_deg = visual_field_radius_deg
        else:
            max_eccentricity_deg = _finite_real(
                maximum_eccentricity_deg, "maximum_eccentricity_deg"
            )
            if max_eccentricity_deg <= 0.0:
                raise ValueError("maximum_eccentricity_deg must be positive and finite")
            if max_eccentricity_deg > visual_field_radius_deg:
                raise ValueError(
                    "maximum_eccentricity_deg must not exceed the visual-field "
                    f"radius ({visual_field_radius_deg:.12g} degrees)"
                )
        if max_eccentricity_deg >= _MAX_SUPPORTED_ECCENTRICITY_DEG:
            raise ValueError(
                "E_max >= 90 degrees; eccentricity mode requires "
                "maximum_eccentricity_deg below 90 degrees"
            )

        cap, full_max_eccentricity, beta = _topography_parameters(parameters)
        full_max_eccentricity = _finite_real(
            full_max_eccentricity, "full_max_eccentricity"
        )
        if full_max_eccentricity <= 0.0:
            raise ValueError("full_max_eccentricity must be positive and finite")
        beta = _finite_real(beta, "beta")
        if not 0.0 < beta < 2.0:
            raise ValueError("beta must satisfy 0 < beta < 2")

        empirical_cap = _empirical_mapping_cap_eccentricity_deg(
            full_max_eccentricity, beta
        )
        if cap is None:
            user_cap = None
            resolved_mapping_cap = empirical_cap
        else:
            user_cap = _positive_cap(cap)
            if user_cap < empirical_cap:
                raise ValueError(
                    "cap_eccentricity must be at least the empirical mapping "
                    f"cap ({empirical_cap:.12g} degrees)"
                )
            if user_cap > max_eccentricity_deg:
                raise ValueError(
                    "cap_eccentricity must not exceed E_max "
                    f"({max_eccentricity_deg:.12g} degrees)"
                )
            resolved_mapping_cap = user_cap

        object.__setattr__(self, "max_eccentricity_deg", max_eccentricity_deg)
        object.__setattr__(self, "user_cap_eccentricity_deg", user_cap)
        object.__setattr__(self, "_full_max_eccentricity_deg", full_max_eccentricity)
        object.__setattr__(self, "_beta", beta)
        object.__setattr__(
            self, "_empirical_mapping_cap_eccentricity_deg", empirical_cap
        )
        object.__setattr__(
            self,
            "_resolved_mapping_cap_eccentricity_deg",
            resolved_mapping_cap,
        )
        object.__setattr__(self, "_frozen", True)

        if max_eccentricity_deg > _DENSITY_EMPIRICAL_LIMIT_DEG:
            logger.warning(
                "LGN density values above approximately 60 degrees are "
                "extrapolated (E_max=%g degrees); values are not clipped.",
                max_eccentricity_deg,
            )
        if max_eccentricity_deg > _RF_SIZE_EMPIRICAL_LIMIT_DEG:
            logger.warning(
                "LGN receptive-field sizes above 25 degrees are extrapolated "
                "(E_max=%g degrees) and may not be accurate.",
                max_eccentricity_deg,
            )

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{self.__class__.__name__} is immutable after construction"
            )
        object.__setattr__(self, name, value)

    def relative_density_at_eccentricity(self, eccentricity_deg):
        """Return capped or uncapped relative density at eccentricity."""

        if self.user_cap_eccentricity_deg is None:
            return relative_lgn_cell_density_at_eccentricity(eccentricity_deg)
        return capped_relative_lgn_cell_density_at_eccentricity(
            eccentricity_deg, self.user_cap_eccentricity_deg
        )

    def relative_density_xy(self, x_deg, y_deg):
        """Return capped or uncapped relative density at Cartesian positions."""

        if self.user_cap_eccentricity_deg is None:
            return relative_lgn_cell_density(x_deg, y_deg)
        return capped_relative_lgn_cell_density(
            x_deg, y_deg, self.user_cap_eccentricity_deg
        )

    def center_sigma_deg(self, eccentricity_deg):
        """Return the effective conventional centre sigma in visual degrees."""

        return rf_center_sigma(eccentricity_deg, self.user_cap_eccentricity_deg)

    def sample_center_sigma_deg(
        self,
        eccentricity_deg,
        rng,
        log10_residual_sd=_RF_CENTER_SIZE_LOG10_RESIDUAL_SD,
        truncation_sd=_RF_CENTER_SIZE_TRUNCATION_SD,
    ):
        """Return centre sigmas drawn about the effective centre-sigma line."""

        return sample_rf_center_sigma(
            eccentricity_deg,
            rng,
            log10_residual_sd,
            self.user_cap_eccentricity_deg,
            truncation_sd,
        )

    def center_sigma_bounds_deg(
        self,
        log10_residual_sd=_RF_CENTER_SIZE_LOG10_RESIDUAL_SD,
        truncation_sd=_RF_CENTER_SIZE_TRUNCATION_SD,
        minimum_eccentricity_deg=0.0,
    ):
        """Return the smallest and largest centre sigma this domain can produce.

        The bounds combine the extremes of the eccentricity domain with the
        truncated scatter, so they depend only on configuration.  No cell has
        been drawn yet and none needs to be: the finest centre sigma is the
        line value at the smallest supported eccentricity scaled by the lower
        scatter factor, and the coarsest is the line value at ``E_max`` scaled
        by the upper factor.

        Deriving the visual-space resolution from the lower bound rather than
        from the realized minimum is what makes that resolution reproducible
        across seeds and independent of ``number_per_polarity``.
        """

        if (
            isinstance(minimum_eccentricity_deg, bool)
            or not isinstance(minimum_eccentricity_deg, numbers.Real)
            or not numpy.isfinite(minimum_eccentricity_deg)
            or minimum_eccentricity_deg < 0.0
        ):
            raise ValueError("minimum_eccentricity_deg must be nonnegative and finite")
        minimum_eccentricity_deg = float(minimum_eccentricity_deg)
        if minimum_eccentricity_deg > self.max_eccentricity_deg:
            raise ValueError("minimum_eccentricity_deg must not exceed E_max")

        lower_factor, upper_factor = rf_center_sigma_scatter_factors(
            log10_residual_sd, truncation_sd
        )
        minimum_line_sigma = rf_center_sigma(
            minimum_eccentricity_deg, self.user_cap_eccentricity_deg
        )
        maximum_line_sigma = rf_center_sigma(
            self.max_eccentricity_deg, self.user_cap_eccentricity_deg
        )
        return (
            float(minimum_line_sigma) * lower_factor,
            float(maximum_line_sigma) * upper_factor,
        )

    def validate_visual_position(self, x_deg, y_deg):
        """Raise ``ValueError`` unless all positions lie within the LGN disk."""

        x = _finite_array(x_deg, "x_deg")
        y = _finite_array(y_deg, "y_deg")
        try:
            x, y = numpy.broadcast_arrays(x, y)
        except ValueError as exc:
            raise ValueError("x_deg and y_deg must be broadcast-compatible") from exc
        eccentricity = numpy.hypot(x, y)
        if numpy.any(eccentricity > self.max_eccentricity_deg):
            raise ValueError(
                "visual position lies outside the LGN RF-centre domain: "
                f"hypot(x_deg, y_deg) must be <= "
                f"{self.max_eccentricity_deg:.12g} degrees"
            )
