"""Deliberately unfinished full-neuron LGN spatial-frequency validation."""

import copy

import matplotlib.pyplot as plt
import numpy as np
from parameters import ParameterSet

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


STIMULUS_DURATION_MS = 2000.0
TEMPORAL_FREQUENCY_HZ = 1.0
TRANSIENT_DISCARD_MS = 500.0
REPRESENTATIVE_ECCENTRICITIES_DEG = (0.0, 4.0)


def _eccentricity_parameters():
    parameters = copy.deepcopy(
        LEGACY_MODEL_PARAMETERS["sheets"]["retina_lgn"]["params"]
    )
    parameters["number_per_polarity"] = len(
        REPRESENTATIVE_ECCENTRICITIES_DEG
    )
    parameters["minimum_samples_per_center_sigma"] = 8.0
    parameters["topography"] = {
        "cap_eccentricity": None,
        "full_max_eccentricity": 90.0,
        "beta": 1.59,
    }
    del parameters["density"]
    del parameters["size"]
    del parameters["receptive_field"]["spatial_resolution"]
    return ParameterSet(parameters)


def _sampled_f1(values, sample_interval_ms):
    values = np.asarray(values, dtype=float)
    times_ms = sample_interval_ms * np.arange(len(values))
    selected = (times_ms >= TRANSIENT_DISCARD_MS) & (
        times_ms < STIMULUS_DURATION_MS
    )
    values = values[selected]
    times_seconds = times_ms[selected] / 1000.0
    values = values - np.mean(values)
    return float(
        2.0
        / len(values)
        * np.abs(
            np.sum(
                values
                * np.exp(
                    -2j
                    * np.pi
                    * TEMPORAL_FREQUENCY_HZ
                    * times_seconds
                )
            )
        )
    )


def _spike_f1(spike_times_ms, presentation_offset_ms):
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    selected = (
        spike_times_ms >= presentation_offset_ms + TRANSIENT_DISCARD_MS
    ) & (
        spike_times_ms < presentation_offset_ms + STIMULUS_DURATION_MS
    )
    spike_times_seconds = spike_times_ms[selected] / 1000.0
    analysis_duration_seconds = (
        STIMULUS_DURATION_MS - TRANSIENT_DISCARD_MS
    ) / 1000.0
    return float(
        2.0
        / analysis_duration_seconds
        * np.abs(
            np.sum(
                np.exp(
                    -2j
                    * np.pi
                    * TEMPORAL_FREQUENCY_HZ
                    * spike_times_seconds
                )
            )
        )
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
            * np.exp(
                -2.0
                * np.pi**2
                * parameters.sigma_c**2
                * frequencies**2
            )
            - parameters.As
            * parameters.sigma_s**2
            * np.exp(
                -2.0
                * np.pi**2
                * parameters.sigma_s**2
                * frequencies**2
            )
        )
    )
    numerical = frequencies[np.argmax(response)]
    assert abs(numerical - analytical) <= frequencies[1] - frequencies[0]


def test_full_neuron_lgn_spatial_frequency_characterization(
    tmp_path, monkeypatch
):
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
    model_parameters["sheets"]["retina_lgn"]["params"] = (
        _eccentricity_parameters().as_dict()
    )
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
        for sheet in retina.sheets.values():
            sheet.pop.record("spikes")

        representative_cells = retina.input_cells["X_ON"]
        theoretical_frequencies = np.array(
            [_theoretical_frequency(cell) for cell in representative_cells]
        )
        for cell in representative_cells:
            _verify_continuous_dog_optimum(cell)

        # These protocol choices are intentionally provisional. The design
        # leaves the mean luminance, trial count, transient rejection,
        # frequency grid/refinement, peak estimator, and agreement tolerance
        # for explicit scientific review.
        spatial_frequencies = np.geomspace(
            0.5 * np.min(theoretical_frequencies),
            1.5 * np.max(theoretical_frequencies),
            5,
        )
        response_curves = {
            "linear": np.zeros(
                (
                    len(REPRESENTATIVE_ECCENTRICITIES_DEG),
                    len(spatial_frequencies),
                )
            ),
            "current": np.zeros(
                (
                    len(REPRESENTATIVE_ECCENTRICITIES_DEG),
                    len(spatial_frequencies),
                )
            ),
            "spikes": np.zeros(
                (
                    len(REPRESENTATIVE_ECCENTRICITIES_DEG),
                    len(spatial_frequencies),
                )
            ),
        }

        for frequency_index, spatial_frequency in enumerate(
            spatial_frequencies
        ):
            stimulus = FullfieldDriftingSinusoidalGrating(
                frame_duration=7.0,
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

            spike_data = retina.sheets["X_ON"].pop.get_data(
                "spikes", clear=True
            )
            spike_trains = spike_data.segments[-1].spiketrains
            for cell_index, (kernel_response, current, spike_train) in enumerate(
                zip(
                    kernel_responses["X_ON"],
                    currents["X_ON"],
                    spike_trains,
                )
            ):
                linear_response = (
                    kernel_response.contrast + kernel_response.luminance
                )
                response_curves["linear"][
                    cell_index, frequency_index
                ] = _sampled_f1(linear_response, 7.0)
                response_curves["current"][
                    cell_index, frequency_index
                ] = _sampled_f1(current["amplitudes"], 7.0)
                response_curves["spikes"][
                    cell_index, frequency_index
                ] = _spike_f1(
                    spike_train.magnitude,
                    presentation_offset_ms,
                )

        figure, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
        for axis, stage in zip(
            axes,
            ("linear", "current", "spikes"),
        ):
            for cell_index, eccentricity in enumerate(
                REPRESENTATIVE_ECCENTRICITIES_DEG
            ):
                label = (
                    f"E={eccentricity:g} deg; "
                    f"theory={theoretical_frequencies[cell_index]:.4g} c/deg"
                )
                axis.plot(
                    spatial_frequencies,
                    response_curves[stage][cell_index],
                    marker="o",
                    label=label,
                )
                axis.axvline(
                    theoretical_frequencies[cell_index],
                    linestyle="--",
                    alpha=0.5,
                )
            axis.set_xscale("log")
            axis.set_ylabel(f"{stage} F1")
            axis.grid(True, alpha=0.25)
            axis.legend()
        axes[-1].set_xlabel("Spatial frequency (cycles/degree)")
        figure.suptitle(
            "Eccentricity LGN theoretical and full-neuron SF characterization"
        )
        figure.tight_layout()
        plot_path = (
            tmp_path / "eccentricity_lgn_spatial_frequency_characterization.png"
        )
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)

        absolute_plot_path = plot_path.resolve()
        print(f"LGN spatial-frequency characterization plot: {absolute_plot_path}")
        for cell_index, eccentricity in enumerate(
            REPRESENTATIVE_ECCENTRICITIES_DEG
        ):
            measured_optima = {
                stage: float(
                    spatial_frequencies[
                        np.argmax(response_curves[stage][cell_index])
                    ]
                )
                for stage in ("linear", "current", "spikes")
            }
            print(
                "LGN SF characterization: "
                f"E={eccentricity:g} deg, "
                f"theory={theoretical_frequencies[cell_index]:.6g} c/deg, "
                f"grid optima={measured_optima}, "
                "processing-stage differences remain provisional"
            )
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

        # TODO Finish: define and approve the spike-output protocol, peak
        # estimator, and agreement tolerance before this can pass.
        assert False, (
            "TODO Finish: define and approve LGN spike-output SF "
            "acceptance criteria"
        )
    finally:
        model.sim.end()
