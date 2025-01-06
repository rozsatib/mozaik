#!/usr/local/bin/ipython -i
from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus
from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet

def create_experiments(model):
    radii = [50,150,300]
    experiments = []
    for i in range(len(radii)):
        experiments.append(
            SingleOptogeneticArrayStimulus(
                model,
                MozaikExtendedParameterSet(
                    {
                        "sheet_list": ["V1_Exc_L2/3"],
                        'sheet_intensity_scaler': [1.0],
                        'sheet_transfection_proportion': [1.0],
                        "num_trials": 1,
                        "stimulator_array_parameters": MozaikExtendedParameterSet(
                            {
                                "size": 1000,
                                "spacing": 10,
                                "depth_sampling_step": 10,
                                "light_source_light_propagation_data": "light_scattering_radial_profiles_lsd10.pickle",
                                "update_interval": 1,
                            }
                        ),
                        "stimulating_signal": "mozaik.sheets.direct_stimulator.stimulating_pattern_flash",
                        "stimulating_signal_parameters": ParameterSet(
                            {
                                "shape": "circle",
                                "coords": [[0,0]],
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