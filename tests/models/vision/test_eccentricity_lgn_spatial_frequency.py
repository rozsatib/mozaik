"""LGN spike-output spatial-frequency validation."""

import copy

import numpy as np
from parameters import ParameterSet
from scipy.optimize import least_squares

import mozaik
from mozaik.models import Model
from mozaik.models.vision import cai97
from mozaik.models.vision import spatiotemporalfilter
from mozaik.models.vision.spatiotemporalfilter import (
    EccentricityDependentSpatioTemporalFilterRetinaLGN,
)
from mozaik.space import VisualRegion
from mozaik.stimuli.vision.topographica_based import (
    FullfieldDriftingSinusoidalGrating,
)
from tests.models.vision.test_legacy_lgn_regression import (
    LEGACY_MODEL_PARAMETERS,
)

FRAME_DURATION_MS = 7.0
# Use a frame-aligned approximately 2 Hz grating. The 36-frame transient
# exceeds the 200 ms RF kernel, leaving two complete cycles for F1 analysis.
TEMPORAL_PERIOD_FRAMES = 72
TRANSIENT_DISCARD_FRAMES = 36
ANALYSIS_CYCLES = 2
TEMPORAL_FREQUENCY_HZ = 1000.0 / (TEMPORAL_PERIOD_FRAMES * FRAME_DURATION_MS)
TRANSIENT_DISCARD_MS = TRANSIENT_DISCARD_FRAMES * FRAME_DURATION_MS
STIMULUS_DURATION_MS = TRANSIENT_DISCARD_MS + (
    ANALYSIS_CYCLES * TEMPORAL_PERIOD_FRAMES * FRAME_DURATION_MS
)
REPRESENTATIVE_ECCENTRICITIES_DEG = (0.0, 4.0)
SPATIAL_FREQUENCY_TOLERANCE_OCTAVES = 0.06


def _eccentricity_parameters():
    parameters = copy.deepcopy(
        LEGACY_MODEL_PARAMETERS["sheets"]["retina_lgn"]["params"]
    )
    parameters["number_per_polarity"] = len(REPRESENTATIVE_ECCENTRICITIES_DEG)
    parameters["minimum_samples_per_center_sigma"] = 8.0
    parameters["topography"] = {
        "cap_eccentricity": None,
        "full_max_eccentricity": 90.0,
        "beta": 1.59,
    }
    parameters["temporal_scale_distribution"] = {
        "mu": 0.1685970743683889,
        "sigma": 0.3059765581600333,
        "lower_quantile": 0.005,
        "upper_quantile": 0.995,
    }
    del parameters["density"]
    del parameters["size"]
    del parameters["receptive_field"]["spatial_resolution"]
    shared_noise = parameters.pop("noise")
    parameters["noise"] = {
        rf_type: copy.deepcopy(shared_noise) for rf_type in ("X_ON", "X_OFF")
    }
    nonlinear = parameters["gain_control"]["non_linear_gain"]
    shared_gain = nonlinear.pop("luminance_gain")
    shared_scaler = nonlinear.pop("luminance_scaler")
    nonlinear.update(
        {
            "luminance_gain_ON": shared_gain,
            "luminance_scaler_ON": shared_scaler,
            "luminance_gain_OFF": shared_gain,
            "luminance_scaler_OFF": shared_scaler,
        }
    )
    return ParameterSet(parameters)


def _spike_f1(spike_times_ms, presentation_offset_ms):
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    selected = (spike_times_ms >= presentation_offset_ms + TRANSIENT_DISCARD_MS) & (
        spike_times_ms < presentation_offset_ms + STIMULUS_DURATION_MS
    )
    spike_times_seconds = spike_times_ms[selected] / 1000.0
    analysis_duration_seconds = (STIMULUS_DURATION_MS - TRANSIENT_DISCARD_MS) / 1000.0
    return float(
        2.0
        / analysis_duration_seconds
        * np.abs(
            np.sum(np.exp(-2j * np.pi * TEMPORAL_FREQUENCY_HZ * spike_times_seconds))
        )
    )


def _log_quadratic(log_frequency, maximum, peak, curvature):
    return maximum - curvature * (log_frequency - peak) ** 2


def _fit_log_quadratic(frequencies, responses):
    log_frequencies = np.log2(np.asarray(frequencies, dtype=float))
    responses = np.asarray(responses, dtype=float)
    response_span = np.ptp(responses)
    log_frequency_span = np.ptp(log_frequencies)
    initial = np.array(
        [
            np.max(responses),
            log_frequencies[np.argmax(responses)],
            response_span / (log_frequency_span / 2.0) ** 2,
        ]
    )
    lower_bounds = np.array([0.0, log_frequencies[0], 0.0])
    upper_bounds = np.array(
        [
            2.0 * np.max(responses),
            log_frequencies[-1],
            100.0 * response_span / log_frequency_span**2,
        ]
    )
    robust_scale = max(0.05 * response_span, np.finfo(float).eps)
    fit = least_squares(
        lambda parameters: _log_quadratic(log_frequencies, *parameters) - responses,
        initial,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=robust_scale,
    )
    assert fit.success, fit.message
    return fit.x


# TODO: Delete this diagnostic plotting function once the characterization is
# no longer needed during development of this test.
def _plot_spatial_frequency_characterization(
    output_directory,
    spatial_frequencies,
    spike_f1,
    theoretical_frequencies,
    fitted_parameters,
    fitted_optima,
):
    import matplotlib.pyplot as plt

    fit_frequencies = np.geomspace(
        spatial_frequencies[0], spatial_frequencies[-1], 1000
    )
    fit_log_frequencies = np.log2(fit_frequencies)
    figure, axis = plt.subplots(figsize=(8, 5))
    for cell_index, eccentricity in enumerate(REPRESENTATIVE_ECCENTRICITIES_DEG):
        samples = axis.plot(
            spatial_frequencies,
            spike_f1[cell_index],
            linestyle="none",
            marker="o",
            label=(
                f"E={eccentricity:g} deg samples; "
                f"theory={theoretical_frequencies[cell_index]:.4g} c/deg"
            ),
        )[0]
        color = samples.get_color()
        fitted_response = _log_quadratic(
            fit_log_frequencies, *fitted_parameters[cell_index]
        )
        axis.plot(
            fit_frequencies,
            fitted_response,
            color=color,
            label=(
                f"E={eccentricity:g} deg robust log-quadratic; "
                f"peak={fitted_optima[cell_index]:.4g} c/deg"
            ),
        )
        axis.plot(
            fitted_optima[cell_index],
            fitted_parameters[cell_index][0],
            color=color,
            marker="*",
            markersize=12,
        )
        axis.axvline(
            theoretical_frequencies[cell_index],
            color=color,
            linestyle="--",
            alpha=0.5,
        )
        axis.axvline(
            fitted_optima[cell_index],
            color=color,
            linestyle=":",
            alpha=0.7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Spatial frequency (cycles/degree)")
    axis.set_ylabel("spike F1 (spikes/s)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.suptitle("Eccentricity LGN theoretical and spike-output SF characterization")
    figure.tight_layout()
    output_path = (
        output_directory
        / "eccentricity_lgn_spike_spatial_frequency_characterization.png"
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(
        "LGN spike spatial-frequency characterization plot: " f"{output_path.resolve()}"
    )


def _report_spatial_frequency_characterization(
    output_directory,
    spatial_frequencies,
    spike_f1,
    theoretical_frequencies,
    fitted_parameters,
    fitted_optima,
    octave_errors,
):
    _plot_spatial_frequency_characterization(
        output_directory,
        spatial_frequencies,
        spike_f1,
        theoretical_frequencies,
        fitted_parameters,
        fitted_optima,
    )
    for cell_index, eccentricity in enumerate(REPRESENTATIVE_ECCENTRICITIES_DEG):
        grid_optimum = float(spatial_frequencies[np.argmax(spike_f1[cell_index])])
        print(
            "LGN spike SF characterization: "
            f"E={eccentricity:g} deg, "
            f"theory={theoretical_frequencies[cell_index]:.6g} c/deg, "
            f"grid optimum={grid_optimum:.6g} c/deg, "
            "robust log-quadratic optimum="
            f"{fitted_optima[cell_index]:.6g} c/deg, "
            f"error={octave_errors[cell_index]:.6g} octaves"
        )


def _theoretical_frequency(cell):
    parameters = cell.receptive_field.func_params
    return cai97.dog_optimal_spatial_frequency(
        parameters.Ac,
        parameters.As,
        parameters.sigma_c,
        parameters.sigma_s,
    )


def _verify_continuous_dog_optimum(cell):
    parameters = cell.receptive_field.func_params
    analytical = _theoretical_frequency(cell)
    frequencies = np.linspace(
        max(analytical * 0.25, 1e-6),
        analytical * 2.0,
        200_001,
    )
    response = (
        2.0
        * np.pi
        * (
            parameters.Ac
            * parameters.sigma_c**2
            * np.exp(-2.0 * np.pi**2 * parameters.sigma_c**2 * frequencies**2)
            - parameters.As
            * parameters.sigma_s**2
            * np.exp(-2.0 * np.pi**2 * parameters.sigma_s**2 * frequencies**2)
        )
    )
    numerical = frequencies[np.argmax(response)]
    assert abs(numerical - analytical) <= frequencies[1] - frequencies[0]


def test_lgn_spike_spatial_frequency_characterization(tmp_path, monkeypatch):
    from pyNN import nest

    fixed_positions = np.array(
        [
            REPRESENTATIVE_ECCENTRICITIES_DEG,
            np.zeros(len(REPRESENTATIVE_ECCENTRICITIES_DEG)),
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        spatiotemporalfilter,
        "_sample_lgn_positions",
        lambda topography, number, rng: fixed_positions.copy(),
    )

    model_parameters = copy.deepcopy(LEGACY_MODEL_PARAMETERS)
    model_parameters["sheets"]["retina_lgn"][
        "params"
    ] = _eccentricity_parameters().as_dict()
    parameters = ParameterSet(model_parameters)
    mozaik.setup_mpi(parameters.mpi_seed, parameters.pynn_seed)
    model = Model(nest, 1, parameters)
    model.visual_field = VisualRegion(
        location_x=0.0,
        location_y=0.0,
        size_x=12.0,
        size_y=12.0,
    )

    try:
        retina = EccentricityDependentSpatioTemporalFilterRetinaLGN(
            model, parameters.sheets.retina_lgn.params
        )
        retina.sheets["X_ON"].pop.record("spikes")

        representative_cells = retina.input_cells["X_ON"]
        theoretical_frequencies = np.array(
            [_theoretical_frequency(cell) for cell in representative_cells]
        )
        for cell in representative_cells:
            _verify_continuous_dog_optimum(cell)

        # Concentrate samples around each predicted optimum while retaining
        # progressively coarser coverage of the surrounding response curve.
        relative_frequency_samples = np.array(
            [
                0.50,
                0.78,
                0.92,
                1.00,
                1.08,
                1.22,
                1.50,
            ]
        )
        spatial_frequencies = np.unique(
            np.concatenate(
                [
                    frequency * relative_frequency_samples
                    for frequency in theoretical_frequencies
                ]
            )
        )
        spike_f1 = np.zeros(
            (
                len(REPRESENTATIVE_ECCENTRICITIES_DEG),
                len(spatial_frequencies),
            )
        )
        for frequency_index, spatial_frequency in enumerate(spatial_frequencies):
            stimulus = FullfieldDriftingSinusoidalGrating(
                frame_duration=FRAME_DURATION_MS,
                duration=STIMULUS_DURATION_MS,
                trial=0,
                background_luminance=45.0,
                density=1.0 / retina.visual_space_resolution_deg,
                location_x=0.0,
                location_y=0.0,
                size_x=12.0,
                size_y=12.0,
                orientation=0.0,
                spatial_frequency=float(spatial_frequency),
                temporal_frequency=TEMPORAL_FREQUENCY_HZ,
                contrast=100.0,
            )
            model.input_space.clear()
            model.input_space.add_object(str(stimulus), stimulus)
            model.input_space.set_duration(STIMULUS_DURATION_MS)
            kernel_responses, _ = retina.calculate_kernel_responses(
                model.input_space, STIMULUS_DURATION_MS
            )
            currents = retina._calculate_input_currents(kernel_responses)
            presentation_offset_ms = float(model.sim.get_current_time())
            retina.inject_currents(
                currents,
                STIMULUS_DURATION_MS,
                presentation_offset_ms,
            )
            model.sim.run(STIMULUS_DURATION_MS)
            spike_data = retina.sheets["X_ON"].pop.get_data("spikes", clear=True)
            spike_trains = spike_data.segments[-1].spiketrains
            for cell_index, spike_train in enumerate(spike_trains):
                spike_f1[cell_index, frequency_index] = _spike_f1(
                    spike_train.magnitude,
                    presentation_offset_ms,
                )

        fitted_parameters = [
            _fit_log_quadratic(spatial_frequencies, responses) for responses in spike_f1
        ]
        fitted_optima = np.array(
            [2.0 ** parameters[1] for parameters in fitted_parameters]
        )
        octave_errors = np.abs(np.log2(fitted_optima / theoretical_frequencies))
        _report_spatial_frequency_characterization(
            tmp_path,
            spatial_frequencies,
            spike_f1,
            theoretical_frequencies,
            fitted_parameters,
            fitted_optima,
            octave_errors,
        )
        assert np.all(octave_errors <= SPATIAL_FREQUENCY_TOLERANCE_OCTAVES), (
            "LGN spike-output SF optima exceed the approved tolerance: "
            f"errors={octave_errors.tolist()} octaves; "
            f"tolerance={SPATIAL_FREQUENCY_TOLERANCE_OCTAVES} octaves"
        )
    finally:
        model.sim.end()
