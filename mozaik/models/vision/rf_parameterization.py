"""Resolve Cai97 RF parameters independently of the sampled population plan."""

import copy
import numbers

import numpy as np
from parameters import ParameterSet

from mozaik.models.vision import cai97


def validate_rf_parameterization(profile, numeric_parameters):
    if profile is None:
        return
    fields = {
        "func_params",
        "spatial_mapping",
        "temporal_mapping",
        "reference_temporal_frequency_hz",
        "temporal_frequency_offset_hz",
        "temporal_frequency_slope_hz",
        "spatial_support",
        "surround_sigma_multiple",
        "support_quantum_deg",
        "duration_rule",
        "duration_t1_multiple",
    }
    if set(profile) != fields:
        raise ValueError(f"Invalid RF parameterization fields: {set(profile) ^ fields}")
    choices = {
        "spatial_mapping": ("eccentricity", "nominal_frequency"),
        "temporal_mapping": ("sampled_scale", "affine_frequency"),
        "spatial_support": ("population_plan", "rounded_surround"),
        "duration_rule": ("population_plan", "t1_multiple"),
    }
    for name, allowed in choices.items():
        if profile[name] not in allowed:
            raise ValueError(f"Invalid RF parameterization {name}: {profile[name]!r}")
    for name in fields - set(choices) - {"func_params"}:
        value = profile[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"RF parameterization {name} must be finite")
        if name != "temporal_frequency_offset_hz" and value <= 0:
            raise ValueError(f"RF parameterization {name} must be positive")
    base = profile.func_params
    missing = set(numeric_parameters) | {"subtract_mean"}
    missing -= set(base)
    if missing:
        raise ValueError(
            f"RF parameterization func_params is missing: {sorted(missing)}"
        )
    for name in numeric_parameters:
        value = base[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"RF parameterization func_params.{name} must be finite")
    if base.sigma_c <= 0 or base.sigma_s <= 0 or base.subtract_mean is not False:
        raise ValueError(
            "RF parameterization requires positive sigmas and subtract_mean=False"
        )


def resolve_cai97_cell_parameters(
    reference,
    center_sigma,
    surround_sigma,
    temporal_scale,
    width,
    height,
    duration,
    profile=None,
):
    """Resolve one cell; ``reference`` remains the population-selection reference."""
    base = reference if profile is None else profile.func_params
    parameters = ParameterSet(copy.deepcopy(base))
    if profile is not None and profile.spatial_mapping == "nominal_frequency":
        target = cai97.dog_optimal_spatial_frequency(
            reference.Ac, reference.As, center_sigma, surround_sigma
        )
        preferred = cai97.dog_optimal_spatial_frequency(
            base.Ac, base.As, base.sigma_c, base.sigma_s
        )
        spatial_scale = preferred / target
        parameters.sigma_c *= spatial_scale
        parameters.sigma_s *= spatial_scale
    else:
        parameters.sigma_c = center_sigma
        parameters.sigma_s = (
            surround_sigma
            if profile is None
            else (center_sigma / base.sigma_c) * base.sigma_s
        )
    divisor = None
    if profile is not None and profile.temporal_mapping == "affine_frequency":
        target = profile.reference_temporal_frequency_hz / temporal_scale
        divisor = (
            target - profile.temporal_frequency_offset_hz
        ) / profile.temporal_frequency_slope_hz
        if not np.isfinite(divisor) or divisor <= 0:
            raise ValueError(
                f"RF parameterization cannot map temporal preference {target:g} Hz"
            )
        temporal_scale = 1.0 / divisor
    for name in ("t1", "t2", "td"):
        parameters[name] *= temporal_scale
    for name in ("c1", "c2"):
        parameters[name] /= temporal_scale
    if profile is not None and profile.spatial_support == "rounded_surround":
        width = (
            np.ceil(
                profile.surround_sigma_multiple
                * parameters.sigma_s
                / profile.support_quantum_deg
            )
            * profile.support_quantum_deg
        )
        height = width
    if profile is not None and profile.duration_rule == "t1_multiple":
        duration = profile.duration_t1_multiple * abs(base.t1)
        duration = duration * temporal_scale if divisor is None else duration / divisor
    if not all(np.isfinite(v) and v > 0 for v in (width, height, duration)):
        raise ValueError("Resolved RF support and duration must be positive and finite")
    return parameters, float(width), float(height), float(duration)
