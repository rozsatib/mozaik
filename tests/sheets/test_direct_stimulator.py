import pickle
import pytest
import numpy as np
from pyNN import nest
import quantities as qt
from copy import deepcopy
from mozaik.models import Model
from parameters import ParameterSet
from mozaik.sheets.vision import VisualCorticalUniformSheet3D
from mozaik.sheets.direct_stimulator import *
from mozaik.experiments.optogenetic import *
from mozaik.tools.distribution_parametrization import PyNNDistribution
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet

model_params = {
    "input_space_type": "",
    "input_space": None,
    "sheets": None,
    "results_dir": "",
    "name": "M",
    "reset": True,
    "null_stimulus_period": 150.0,
    "store_stimuli": False,
    "min_delay": 0.1,
    "max_delay": 100,
    "time_step": 0.1,
    "pynn_seed": 936395,
    "mpi_seed": 1023,
}


layer_params = {
    "name": "exc_sheet",
    "sx": 400,
    "sy": 400,
    "density": 1000,
    "min_depth": 100,
    "max_depth": 400,
    "mpi_safe": False,
    "magnification_factor": 1000,
    "cell": {
        "model": "IF_cond_exp",
        "params": {
            "v_thresh": -50.0,
            "v_rest": -60.0,
            "v_reset": -60.0,
            "tau_refrac": 5.0,
            "tau_m": 20.0,
            "cm": 0.2,
            "e_rev_E": 0.0,
            "e_rev_I": -80.0,
            "tau_syn_E": 5.0,
            "tau_syn_I": 10.0,
        },
        "initial_values": {
            "v": PyNNDistribution(name="uniform", low=-60, high=-60),
        },
    },
    "artificial_stimulators": {},
    "recording_interval": 1.0,
    "recorders": {
        "2": {
            "component": "mozaik.sheets.population_selector.RCGrid",
            "variables": ("v"),
            "params": {
                "size": 400.0,
                "spacing": 20.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
            },
        },
    },
}


class TestDirectStimulator:
    pass


class TestBackgroundActivityBombardment:
    pass


class TestKick:
    pass


class TestDepolarization:
    pass


class TestOpticalStimulatorArray:
    pass


class TestOpticalStimulatorArrayChR:

    stim_array_parameters = MozaikExtendedParameterSet(
        {
            "size": 400,
            "spacing": 10,
            "depth_sampling_step": 10,
            "light_source_light_propagation_data": "tests/sheets/unity_radprof.pickle",
            "update_interval": 1,
            "stimulating_signal": "mozaik.sheets.direct_stimulator.stimulating_pattern_flash",
            "stimulating_signal_parameters": ParameterSet(
                {
                    "shape": "circle",
                    "coords": [[0, 0]],
                    "radius": 125,
                    "intensity": [0.05],
                    "duration": 1000,
                    "onset_time": 250,
                    "offset_time": 750,
                }
            ),
        }
    )

    model = None
    sheet = None

    def create_unity_radprof(self, h=1000, w=1000):
        radprof = np.zeros((h, w))
        radprof[:, 0] = 1
        f = open("tests/sheets/unity_radprof.pickle", "wb")
        pickle.dump(radprof, f)
        f.close()

    @classmethod
    def setup_class(cls):
        cls.model = Model(nest, 8, ParameterSet(model_params))
        cls.sheet = VisualCorticalUniformSheet3D(cls.model, ParameterSet(layer_params))
        cls.sheet.record()
        cls.duration = cls.stim_array_parameters.stimulating_signal_parameters.duration

    def record_and_retrieve_data(self, ds, duration):
        self.model.reset()
        self.sheet.prepare_artificial_stimulation(duration, 0, [ds])
        self.model.run(duration)
        ds.inactivate(duration)
        return (
            self.sheet.get_data(duration).analogsignals[0]
            - layer_params["cell"]["params"]["v_rest"] * qt.mV
        )

    def test_scs_sharing(self):
        radii = np.arange(50, 200.1, 50)
        shared_scs = {}

        for radius in radii:
            sap = MozaikExtendedParameterSet(deepcopy(self.stim_array_parameters))
            sap.stimulating_signal_parameters.radius = radius

            ds = OpticalStimulatorArrayChR(self.sheet, sap, shared_scs)
            shared_scs.update(
                {ds.stimulated_cells[i]: ds.scs[i] for i in range(len(ds.scs))}
            )
            d_share = self.record_and_retrieve_data(ds, self.duration)

            ds = OpticalStimulatorArrayChR(self.sheet, sap)
            d_no_share = self.record_and_retrieve_data(ds, self.duration)

            assert np.all(d_share == d_no_share)

    def test_scs_optimization(self):
        shared_scs = None
        shared_scs_optimized = {}
        radii = np.arange(50, 200.1, 50)

        for radius in radii:
            sap = MozaikExtendedParameterSet(deepcopy(self.stim_array_parameters))
            sap.stimulating_signal_parameters.radius = radius

            ds = OpticalStimulatorArrayChR(
                self.sheet, self.stim_array_parameters, shared_scs_optimized
            )

            shared_scs_optimized.update(
                {ds.stimulated_cells[i]: ds.scs[i] for i in range(len(ds.scs))}
            )
            d1 = self.record_and_retrieve_data(ds, self.duration)

            ds = OpticalStimulatorArrayChR(
                self.sheet,
                self.stim_array_parameters,
                shared_scs=shared_scs,
                optimized_scs=False,
            )
            shared_scs = ds.scs
            d2 = self.record_and_retrieve_data(ds, self.duration)

            assert np.array_equal(d1,d2)

    def plot_max_response(self, d1, d2):
        import matplotlib.pyplot as plt

        idx = np.argmax(d2.sum(axis=0))
        plt.plot(d1[:, idx])
        plt.plot(d2[:, idx])
        plt.show()
