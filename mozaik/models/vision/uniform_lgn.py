"""Legacy Cartesian LGN parameters on the current per-cell RF architecture."""

import copy

from parameters import ParameterSet

from mozaik.models.vision.spatiotemporalfilter import (
    EccentricityDependentSpatioTemporalFilterRetinaLGN,
    SpatioTemporalFilterRetinaLGN,
)


def uniform_lgn_parameters(legacy_parameters):
    """Preserve a current legacy preset and add independent temporal scaling.

    The current legacy RF already uses sample-count normalization. Convert
    only its spatial-mean luminance scaler to spatial-sum units. No historical
    Cai97 substitution, neuron change, noise/gain refit, or compensation occurs.
    """
    parameters = copy.deepcopy(legacy_parameters)
    mode = parameters.pop("temporal_scale_mode", "uniform")
    if mode == "constant":
        mode = "uniform"  # Read profiles saved before this policy was renamed.
    scale = parameters.pop("temporal_scale", 1.0)
    if "temporal_scale_distribution" in parameters:
        raise ValueError(
            "The canonical temporal distribution is fixed in lgn_temporal_scales.py; "
            "select 'custom' and edit custom() to use another distribution"
        )
    object.__new__(SpatioTemporalFilterRetinaLGN).check_parameters(parameters)
    size, density = parameters.pop("size"), parameters.pop("density")
    pixel_size = parameters.receptive_field.pop("spatial_resolution")
    parameters.update(
        {
            "spatial_layout": ParameterSet({"size": size, "density": density}),
            "temporal_scale_mode": mode,
            "temporal_scale": scale,
            "temporal_scale_distribution": None,
            "minimum_eccentricity_deg": 0.0,
            "maximum_eccentricity_deg": None,
            "minimum_samples_per_center_sigma": parameters.receptive_field.func_params.sigma_c
            / pixel_size,
            "visual_space_pixel_size_deg": pixel_size,
            "topography": ParameterSet(
                {"cap_eccentricity": None, "full_max_eccentricity": 90.0, "beta": 1.59}
            ),
            "center_size_log10_residual_sd": 0.0,
            "center_size_truncation_sd": 3.0,
            "debug_legacy_luminance_normalization": False,
            "debug_historical_cai97_receptive_field": False,
            "luminance_scaler_spatial_mean_units": True,
            "rf_scaler_historical_normalization_units": False,
            "receptive_field_parameterization": None,
            "temporal_compensation": None,
        }.items()
    )
    nonlinear = parameters.gain_control.non_linear_gain
    for name in ("luminance_gain", "luminance_scaler"):
        value = nonlinear.pop(name)
        nonlinear[f"{name}_ON"] = value
        nonlinear[f"{name}_OFF"] = value
    noise = parameters.pop("noise")
    parameters["noise"] = ParameterSet(
        {polarity: copy.deepcopy(noise) for polarity in ("X_ON", "X_OFF")}
    )
    return parameters


class UniformSpatioTemporalFilterRetinaLGN(
    EccentricityDependentSpatioTemporalFilterRetinaLGN
):
    """Uniform spatial RFs and legacy placement with per-cell temporal scales.

    Accept the old uniform LGN parameter schema plus optional temporal settings.
    Shared placement/noise RNG usage remains the same as the legacy component,
    so switching architectures preserves cortical initialization and wiring.
    """

    legacy_uniform_placement = True

    def __init__(self, model, parameters):
        super().__init__(model, uniform_lgn_parameters(parameters))

    def check_parameters(self, parameters):
        if "density" in parameters or "size" in parameters:
            parameters = uniform_lgn_parameters(parameters)
        super().check_parameters(parameters)
