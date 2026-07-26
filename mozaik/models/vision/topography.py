"""Analytical LGN topography for the eccentricity-dependent visual mode.

LGN positions use Cartesian visual-field coordinates in degrees.  The density
fit is an area density with respect to ``dx dy``; it contains no polar
Jacobian.  The receptive-field fit returns the conventional Gaussian standard
deviation used by :func:`mozaik.models.vision.cai97.F_2d`.

The density fit is empirically poorly constrained above approximately 60
degrees, and the RF-size fit was reported below 25 degrees.  Larger supported
eccentricities are extrapolated without clipping and are reported once when
the provider is initialized.
"""

import numbers

import numpy
from scipy.optimize import brentq

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
    centres occupy the inscribed disk with radius ``max_eccentricity_deg``.
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

    def __init__(self, visual_field, parameters):
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

        max_eccentricity_deg = min(size_x, size_y) / 2.0
        if max_eccentricity_deg >= _MAX_SUPPORTED_ECCENTRICITY_DEG:
            raise ValueError(
                "visual_field produces E_max >= 90 degrees; eccentricity mode "
                "requires an inscribed LGN domain with E_max < 90 degrees"
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
