import pathlib

import numpy as np
import pytest
import quantities as qt
from parameters import ParameterSet

import mozaik
from mozaik.experiments import NoStimulation
from mozaik.experiments.closed_loop import ClosedLoopOptogeneticStimulation
from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus
from mozaik.models import Model
from mozaik.sheets import direct_stimulator
from mozaik.sheets.vision import VisualCorticalUniformSheet3D
from mozaik.tools.distribution_parametrization import (
    load_parameters,
    MozaikExtendedParameterSet,
)

CLOSED_LOOP_STATE_UPDATE_INTERVAL = 6.0


class _FixedClosedLoopSignal:
    def __init__(self, signal):
        self.signal = signal

    def __call__(self, stimulator):
        samples = int(
            stimulator.parameters.state_update_interval
            / stimulator.parameters.update_interval
        )
        if hasattr(stimulator, "next_actuation_time"):
            start_time = stimulator.next_actuation_time()
        else:
            start_time = stimulator.current_time()
        start = int(round(start_time / stimulator.parameters.update_interval))
        stop = start + samples
        if stop <= self.signal.shape[2]:
            return self.signal[:, :, start:stop]
        segment = np.zeros(self.signal.shape[:2] + (samples,))
        if start < self.signal.shape[2]:
            available = self.signal[:, :, start:]
            segment[:, :, : available.shape[2]] = available
        return segment


def _noop_closed_loop_update(stimulator):
    pass


class _SpikeCountController:
    def __init__(self, signal_shape, samples, use_direct_nest_spike_retrieval):
        self.signal_shape = signal_shape
        self.samples = samples
        self.use_direct_nest_spike_retrieval = use_direct_nest_spike_retrieval
        self.spikes_in_latest_interval = 0
        self.cumulative_spike_count = 0
        self.spike_count_history = []

    def update_state(self, stimulator):
        if self.use_direct_nest_spike_retrieval:
            self.spikes_in_latest_interval = int(stimulator.last_spike_counts.sum())
            self.cumulative_spike_count += self.spikes_in_latest_interval
        else:
            # Neo supplies cumulative spike trains, so derive the interval count.
            recording = stimulator.feedback_data["exc_sheet"]
            new_cumulative_count = sum(len(train) for train in recording.spiketrains)
            self.spikes_in_latest_interval = (
                new_cumulative_count - self.cumulative_spike_count
            )
            self.cumulative_spike_count = new_cumulative_count
        self.spike_count_history.append(self.spikes_in_latest_interval)

    def calculate_input(self, stimulator):
        intensity = 1.0 if self.spikes_in_latest_interval == 0 else 0.5
        return np.full(self.signal_shape + (self.samples,), intensity)


def _make_opto_model(
    recording_variables,
    test_dir,
    closed_loop,
    integrated=False,
    actuation_delay=None,
):
    from pyNN import nest

    model_params = load_parameters(test_dir + "/sheets/model_params")
    model_params.reset = True
    mozaik.setup_seeds(
        model_seed=model_params["model_seed"],
        simulation_seed=model_params["simulation_seed"],
        experiment_seed=model_params["experiment_seed"],
    )
    mozaik.setup_mpi()

    sheet_params = load_parameters(test_dir + "/sheets/exc_sheet_params")
    sheet_params.min_depth = 100
    sheet_params.max_depth = 400
    sheet_params.cell = ParameterSet(
        {
            "model": "EIF_cond_exp_isfa_ista",
            "native_nest": False,
            "params": {
                "v_rest": -80.0,
                "v_reset": -60.0,
                "tau_refrac": 2.0,
                "tau_m": 8.0,
                "cm": 0.032,
                "e_rev_E": 0.0,
                "e_rev_I": -80.0,
                "tau_syn_E": 1.5,
                "tau_syn_I": 4.2,
                "a": 0.0,
                "b": 0.0,
                "delta_T": 0.8,
                "tau_w": 1.0,
                "v_thresh": -56.0,
            },
            "receptors": None,
            "initial_values": {"v": -60.0},
        }
    )
    backend_recording_variables = recording_variables
    if integrated:
        backend_recording_variables = tuple(
            "V_m" if variable == "v" else variable for variable in recording_variables
        )
    sheet_params.recorders = _recorders_for_variables(backend_recording_variables)
    if integrated:
        import nest as nest_api

        if "aeif_cond_exp_sc" not in nest_api.node_models:
            nest_api.Install("stepcurrentmodule")
        cell = sheet_params.cell
        sheet_params.cell = {
            "model": "aeif_cond_exp_sc",
            "native_nest": True,
            "params": {
                "E_L": cell.params.v_rest,
                "V_reset": cell.params.v_reset,
                "t_ref": cell.params.tau_refrac,
                "C_m": cell.params.cm * 1000,
                "g_L": cell.params.cm / cell.params.tau_m * 1000,
                "E_ex": cell.params.e_rev_E,
                "E_in": cell.params.e_rev_I,
                "tau_syn_ex": cell.params.tau_syn_E,
                "tau_syn_in": cell.params.tau_syn_I,
                "a": cell.params.a,
                "b": cell.params.b * 1000,
                "Delta_T": 0.8,
                "tau_w": 1.0,
                "V_th": cell.params.v_thresh,
                "V_peak": -40.0,
                "I_e": 0.0,
            },
            "receptors": cell.receptors,
            "initial_values": {"V_m": cell.initial_values.v},
        }

    opt_array_params = load_parameters(test_dir + "/sheets/opt_array_params")
    opt_array_params["transfection_proportion"] = 1.0
    component = "mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR"
    if closed_loop:
        component = "mozaik.sheets.direct_stimulator.ClosedLoopOpticalStimulatorArray"
        opt_array_params["state_update_interval"] = CLOSED_LOOP_STATE_UPDATE_INTERVAL
        opt_array_params["actuation_delay"] = actuation_delay
        opt_array_params["feedback_sheet_names"] = [sheet_params.name]
    sheet_params.artificial_stimulators = {
        "stimulator": {
            "component": component,
            "params": opt_array_params,
        }
    }

    model = Model(nest, 1, model_params)
    VisualCorticalUniformSheet3D(model, ParameterSet(sheet_params))
    return model


def _recorders_for_variables(recording_variables):
    if recording_variables == ():
        return {}
    return {
        "1": {
            "component": "mozaik.sheets.population_selector.RCGrid",
            "variables": recording_variables,
            "params": {
                "size": 400.0,
                "spacing": 20.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
            },
        },
    }


def _single_video_stimulation(model, video_path, duration):
    return SingleOptogeneticArrayStimulus(
        model,
        MozaikExtendedParameterSet(
            {
                "num_trials": 1,
                "stimulator_array_list": [
                    {
                        "sheet": "exc_sheet",
                        "name": "stimulator",
                        "intensity_scaler": 1.0,
                    }
                ],
                "stimulating_signal_function": (
                    "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
                ),
                "stimulating_signal_function_parameters": ParameterSet(
                    {
                        "shape": "video",
                        "video_path": str(video_path),
                        "intensity": 0.1,
                        "duration": duration,
                        "onset_time": 0.0,
                        "offset_time": duration,
                    }
                ),
            }
        ),
    )


def _video_signal(kind, duration):
    video = np.zeros((3, 3, int(duration)))

    if kind == "constant":
        video[1, 1, :] = 1.0
        video[0, 2, :] = 0.5
    elif kind == "sinusoid":
        phase = 2 * np.pi * np.arange(int(duration)) / duration
        video[1, 1, :] = 0.5 + 0.5 * np.sin(phase)
        video[0, 2, :] = 0.5 + 0.5 * np.cos(phase)
    elif kind == "zero":
        pass
    else:
        raise ValueError(kind)

    return video


def _closed_loop_stimulation(model, signal, duration):
    return ClosedLoopOptogeneticStimulation(
        model,
        MozaikExtendedParameterSet(
            {
                "num_trials": 1,
                "duration": int(duration),
                "stimulator_array_list": [
                    {
                        "sheet": "exc_sheet",
                        "name": "stimulator",
                        "input_calculation_function": _FixedClosedLoopSignal(signal),
                        "state_update_function": _noop_closed_loop_update,
                    }
                ],
            }
        ),
    )


def _run_first_stimulus(model, experiment):
    if experiment.direct_stimulation is None:
        direct_stimulation = {}
    else:
        direct_stimulation = experiment.direct_stimulation[0]
    segments, _, _, _, exploded = model.present_stimulus_and_record(
        experiment.stimuli[0], direct_stimulation
    )
    assert not exploded
    return segments


def _recorded_output(segments):
    if not segments:
        return {"spikes": (), "analog": {}}

    assert len(segments) == 1
    segment = segments[0]
    return {
        "spikes": tuple(np.array(st.rescale(qt.ms)) for st in segment.spiketrains),
        "analog": {
            "v" if signal.name == "V_m" else signal.name: np.array(signal)
            for signal in segment.analogsignals
        },
    }


def _assert_same_recorded_output(
    actual, expected, exact_analog=False, analog_rtol=1e-7
):
    assert actual["analog"].keys() == expected["analog"].keys()
    for name in expected["analog"]:
        if exact_analog:
            np.testing.assert_array_equal(
                actual["analog"][name], expected["analog"][name]
            )
        else:
            np.testing.assert_allclose(
                actual["analog"][name],
                expected["analog"][name],
                rtol=analog_rtol,
            )

    assert len(actual["spikes"]) == len(expected["spikes"])
    for actual_spikes, expected_spikes in zip(actual["spikes"], expected["spikes"]):
        np.testing.assert_array_equal(actual_spikes, expected_spikes)


@pytest.mark.parametrize(
    "recording_variables",
    [
        pytest.param((), id="no_recorders"),
        pytest.param(("spikes",), id="spikes_only"),
        pytest.param(("v",), id="analog_only"),
        pytest.param(("spikes", "v"), id="spikes_and_analog"),
    ],
)
@pytest.mark.parametrize(
    "input_signal",
    [
        pytest.param("constant", id="constant_input"),
        pytest.param("sinusoid", id="time_varying_input"),
        pytest.param("zero", id="no_input"),
    ],
)
def test_closed_loop_matches_open_loop_and_spontaneous_activity(
    recording_variables, input_signal, tmp_path
):
    """
    Verify that closed-loop optogenetic stimulation can replay the same direct
    stimulation signal as open-loop video stimulation, independently of the sheet
    recording configuration.

    The comparison covers constant, time-varying sinusoidal, and zero inputs.
    It runs for five closed-loop update intervals so that repeated online
    reprogramming of the stimulator is checked against the open-loop schedule.
    The external ``StepCurrentSource`` and integrated neuron-current paths use
    the same explicit actuation delay. Each closed-loop response is compared
    with its matching open-loop response, and the two current-delivery paths are
    compared with each other. For zero input, each output is also compared to
    NoStimulation to check that a closed-loop stimulator with no input preserves
    spontaneous activity.
    """
    duration = 5 * CLOSED_LOOP_STATE_UPDATE_INTERVAL
    test_dir = str(pathlib.Path(__file__).parent.parent)
    model_parameters = load_parameters(test_dir + "/sheets/model_params")
    common_actuation_delay = model_parameters.min_delay + 2 * model_parameters.time_step
    video = _video_signal(input_signal, duration)
    video_path = tmp_path / "fixed_opto_video.npy"
    np.save(str(video_path), video)
    closed_loop_outputs = {}

    for integrated in (False, True):
        backend = "integrated" if integrated else "external"
        open_loop_model = _make_opto_model(
            recording_variables, test_dir, closed_loop=False, integrated=integrated
        )
        open_loop_exp = _single_video_stimulation(open_loop_model, video_path, duration)
        open_loop_signal = open_loop_exp.stimuli[0].direct_stimulation_signals[0]
        open_loop_output = _recorded_output(
            _run_first_stimulus(open_loop_model, open_loop_exp)
        )

        closed_loop_model = _make_opto_model(
            recording_variables,
            test_dir,
            closed_loop=True,
            integrated=integrated,
            actuation_delay=common_actuation_delay,
        )
        closed_loop_stimulator = closed_loop_model.sheets[
            "exc_sheet"
        ].artificial_stimulators["stimulator"]
        assert closed_loop_stimulator.actuation_delay() == common_actuation_delay
        closed_loop_exp = _closed_loop_stimulation(
            closed_loop_model, open_loop_signal, duration
        )
        closed_loop_output = _recorded_output(
            _run_first_stimulus(closed_loop_model, closed_loop_exp)
        )
        closed_loop_outputs[backend] = closed_loop_output

        # Restarting adaptive ChR integration per segment prevents bitwise open/closed equality.
        _assert_same_recorded_output(
            closed_loop_output, open_loop_output, analog_rtol=2e-7
        )

        if input_signal == "zero":
            spontaneous_model = _make_opto_model(
                recording_variables, test_dir, closed_loop=True, integrated=integrated
            )
            spontaneous_exp = NoStimulation(
                spontaneous_model, ParameterSet({"duration": duration})
            )
            spontaneous_output = _recorded_output(
                _run_first_stimulus(spontaneous_model, spontaneous_exp)
            )
            _assert_same_recorded_output(closed_loop_output, spontaneous_output)

    _assert_same_recorded_output(
        closed_loop_outputs["integrated"],
        closed_loop_outputs["external"],
        exact_analog=True,
    )


def test_closed_loop_spike_retrieval_paths_match(monkeypatch):
    """Compare the temporary direct-NEST path with normal Mozaik retrieval.

    Each run uses the same spiking model and feedback controller. The controller
    obtains interval spike counts either from ``last_spike_counts`` or by
    subtracting consecutive cumulative Neo recordings. Both paths must observe
    the same counts, choose the same stimulation, and produce the same response.
    """
    duration = 3 * CLOSED_LOOP_STATE_UPDATE_INTERVAL
    test_dir = str(pathlib.Path(__file__).parent.parent)
    results_by_retrieval_path = {}

    for use_direct_nest_spike_retrieval in (True, False):
        monkeypatch.setattr(
            direct_stimulator,
            "USE_DIRECT_NEST_SPIKE_RETRIEVAL",
            use_direct_nest_spike_retrieval,
        )
        model = _make_opto_model(
            ("spikes",), test_dir, closed_loop=True, integrated=True
        )
        model.sheets["exc_sheet"].pop.set(I_e=10000.0)
        stimulator = model.sheets["exc_sheet"].artificial_stimulators["stimulator"]
        controller = _SpikeCountController(
            stimulator.stimulator_coords_x.shape,
            int(
                stimulator.parameters.state_update_interval
                / stimulator.parameters.update_interval
            ),
            use_direct_nest_spike_retrieval,
        )
        experiment = ClosedLoopOptogeneticStimulation(
            model,
            MozaikExtendedParameterSet(
                {
                    "num_trials": 1,
                    "duration": int(duration),
                    "stimulator_array_list": [
                        {
                            "sheet": "exc_sheet",
                            "name": "stimulator",
                            "input_calculation_function": controller.calculate_input,
                            "state_update_function": controller.update_state,
                        }
                    ],
                }
            ),
        )
        output = _recorded_output(_run_first_stimulus(model, experiment))
        retrieval_path = (
            "direct_nest" if use_direct_nest_spike_retrieval else "standard_mozaik"
        )
        results_by_retrieval_path[retrieval_path] = {
            "recorded_output": output,
            "cumulative_spike_count": controller.cumulative_spike_count,
            "interval_spike_counts": controller.spike_count_history,
        }

    direct_result = results_by_retrieval_path["direct_nest"]
    mozaik_result = results_by_retrieval_path["standard_mozaik"]
    _assert_same_recorded_output(
        mozaik_result["recorded_output"], direct_result["recorded_output"]
    )
    assert (
        mozaik_result["cumulative_spike_count"]
        == direct_result["cumulative_spike_count"]
    )
    assert (
        mozaik_result["interval_spike_counts"] == direct_result["interval_spike_counts"]
    )
    if mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        # Guard against a vacuous comparison in which neither path sees spikes.
        assert direct_result["cumulative_spike_count"] > 0
        assert all(count > 0 for count in direct_result["interval_spike_counts"])
