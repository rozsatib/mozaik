import numpy as np
from mozaik.experiments import Experiment
from parameters import ParameterSet
from mozaik.stimuli import InternalStimulus
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from collections import OrderedDict


class ClosedLoopOptogeneticStimulation(Experiment):
    r"""
    Closed-loop optogenetic stimulation of cortical sheets using optical
    stimulator arrays.

    In this experiment, stimulation is generated online based on neural activity.
    Each stimulator is assigned:
    - an input calculation function, computing the stimulation signal,
    - a state update function, maintaining controller state.

    The input calculation function must return a 3D numpy array of shape:

        (Nx, Ny, T_step)

    where Nx and Ny match the stimulator grid, and T_step corresponds to
    state_update_interval / update_interval.

    Values represent a dimensionless light intensity in [0, 1], which is
    converted by the stimulator into physical light units and injected current.

    Each stimulus lasts for *duration* and is repeated *num_trials* times.

    Parameters
    ----------

    model : Model
        The model on which to execute the experiment.

    Other parameters
    ----------------

    stimulator_array_list : list(dict)
        List defining which stimulator arrays are used. Each entry must contain:

        sheet : str
            Name of the sheet containing the stimulator.

        name : str
            Name of the stimulator within the sheet.

        input_calculation_function : callable
            Function computing the stimulation signal. Must return an array of
            shape (Nx, Ny, T_step) with values in [0, 1].

        state_update_function : callable
            Function updating the internal controller state.

    duration : int
        Duration of the stimulation (ms).

    num_trials : int
        Number of times the stimulus is presented.
    """

    required_parameters = ParameterSet(
        {
            "stimulator_array_list": list,
            "duration": int,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        stimulator_array_keys = {
            "sheet",
            "name",
            "input_calculation_function",
            "state_update_function",
        }
        for stimulator_param in self.parameters.stimulator_array_list:
            assert (
                stimulator_param.keys() == stimulator_array_keys
            ), "Stimulator array keys must be: %s. Supplied: %s. Difference: %s" % (
                stimulator_array_keys,
                stimulator_param.keys(),
                set(stimulator_array_keys) ^ set(stimulator_param.keys()),
            )

        stimulators, signals = OrderedDict(), []
        self.direct_stimulation = []
        for i in range(len(self.parameters.stimulator_array_list)):
            p = self.parameters.stimulator_array_list[i]
            stimulator = model.sheets[p["sheet"]].artificial_stimulators[p["name"]]
            stimulator.calculate_input_function = p["input_calculation_function"]
            stimulator.update_state_function = p["state_update_function"]
            stimulators[p["sheet"]] = [
                stimulator
            ]  # In this experiment we only have a single stimulator per sheet
            x, dt = (
                stimulator.stimulator_coords_x,
                stimulator.parameters.update_interval,
            )
            signals.append(
                np.ones(
                    (np.shape(x)[0], np.shape(x)[1], int(self.parameters.duration / dt))
                )
            )

        for trial in range(self.parameters.num_trials):
            self.direct_stimulation.append(stimulators)
            stimulus = InternalStimulus(
                frame_duration=self.parameters.duration,
                duration=self.parameters.duration,
                trial=trial,
                direct_stimulation_name=type(next(iter(stimulators.values()))).__name__,
                direct_stimulation_parameters=MozaikExtendedParameterSet(
                    {"signal_function_parameters": None}
                ),
            )
            stimulus.direct_stimulation_signals = signals
            self.stimuli.append(stimulus)
