#!/usr/local/bin/ipython -i
from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus
from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet


def create_experiments(model):
    radii = [50, 150, 300]
    experiments = []
    for i in range(len(radii)):
        experiments.append(
            SingleOptogeneticArrayStimulus(
                model,
                MozaikExtendedParameterSet(
                    {
                        "stimulator_array_list": [
                            {
                                "sheet": "V1_Exc_L2/3",
                                "name": "stimulator_array",
                                "intensity_scaler": 1.0,
                            }
                        ],
                        "num_trials": 1,
                        "stimulating_signal_function": "mozaik.sheets.direct_stimulator.stimulating_pattern_flash",
                        "stimulating_signal_function_parameters": ParameterSet(
                            {
                                "shape": "circle",
                                "coords": [[0, 0]],
                                "radius": radii[i],
                                "intensity": 10,
                                "duration": 1000,
                                "onset_time": 250,
                                "offset_time": 500,
                            }
                        ),
                    }
                ),
            )
        )
    return experiments
