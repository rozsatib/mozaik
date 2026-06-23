import pathlib

import numpy as np
import pytest
import quantities as qt
from parameters import ParameterSet

from mozaik.experiments import NoStimulation
from mozaik.experiments.closed_loop import ClosedLoopOptogeneticStimulation
from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus
from mozaik.models import Model
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


def _make_opto_model(recording_variables, test_dir, closed_loop):
    from pyNN import nest
    import mozaik

    mozaik.setup_mpi(mozaik_seed=1024, pynn_seed=1024)
    model_params = load_parameters(test_dir + "/sheets/model_params")
    model_params.reset = True

    sheet_params = load_parameters(test_dir + "/sheets/exc_sheet_params")
    sheet_params.min_depth = 100
    sheet_params.max_depth = 400
    sheet_params.recorders = _recorders_for_variables(recording_variables)

    opt_array_params = load_parameters(test_dir + "/sheets/opt_array_params")
    opt_array_params["transfection_proportion"] = 1.0
    component = "mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR"
    if closed_loop:
        component = "mozaik.sheets.direct_stimulator.ClosedLoopOpticalStimulatorArray"
        opt_array_params["state_update_interval"] = CLOSED_LOOP_STATE_UPDATE_INTERVAL
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
        "analog": {signal.name: np.array(signal) for signal in segment.analogsignals},
    }


def _assert_same_recorded_output(actual, expected):
    assert actual["analog"].keys() == expected["analog"].keys()
    for name in expected["analog"]:
        np.testing.assert_allclose(actual["analog"][name], expected["analog"][name])

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
    For zero input, the recorded output is also compared to NoStimulation to
    check that a closed-loop stimulator with no input preserves spontaneous
    activity.
    """
    duration = 5 * CLOSED_LOOP_STATE_UPDATE_INTERVAL
    test_dir = str(pathlib.Path(__file__).parent.parent)
    video = _video_signal(input_signal, duration)
    video_path = tmp_path / "fixed_opto_video.npy"
    np.save(str(video_path), video)

    open_loop_model = _make_opto_model(recording_variables, test_dir, closed_loop=False)
    open_loop_exp = _single_video_stimulation(open_loop_model, video_path, duration)
    open_loop_signal = open_loop_exp.stimuli[0].direct_stimulation_signals[0]
    open_loop_output = _recorded_output(
        _run_first_stimulus(open_loop_model, open_loop_exp)
    )

    closed_loop_model = _make_opto_model(
        recording_variables, test_dir, closed_loop=True
    )
    closed_loop_exp = _closed_loop_stimulation(
        closed_loop_model, open_loop_signal, duration
    )
    closed_loop_output = _recorded_output(
        _run_first_stimulus(closed_loop_model, closed_loop_exp)
    )

    _assert_same_recorded_output(closed_loop_output, open_loop_output)

    if input_signal == "zero":
        spontaneous_model = _make_opto_model(
            recording_variables, test_dir, closed_loop=True
        )
        spontaneous_exp = NoStimulation(
            spontaneous_model, ParameterSet({"duration": duration})
        )
        spontaneous_output = _recorded_output(
            _run_first_stimulus(spontaneous_model, spontaneous_exp)
        )
        _assert_same_recorded_output(closed_loop_output, spontaneous_output)
