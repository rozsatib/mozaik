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
from mozaik.tools.distribution_parametrization import (
    load_parameters,
    PyNNDistribution,
    MozaikExtendedParameterSet,
)


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

    model = None
    sheet = None

    def create_unity_radprof(self, h=20, w=100):
        radprof = np.zeros((h, w))
        radprof[:, 0] = 1
        f = open("tests/sheets/unity_radprof.pickle", "wb")
        pickle.dump(radprof, f)
        f.close()

    @classmethod
    def setup_class(cls):
        model_params = load_parameters("tests/sheets/model_params")
        cls.sheet_params = load_parameters("tests/sheets/exc_sheet_params")
        cls.opt_array_params = load_parameters("tests/sheets/opt_array_params")
        cls.opt_array_params["transfection_proportion"] = 1.0
        cls.opt_array_params[
            "stimulating_signal"
        ] = "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        cls.opt_array_params["stimulating_signal_parameters"] = ParameterSet(
            {
                "shape": "circle",
                "coords": [[0, 0]],
                "radius": 125,
                "intensity": [0.05],
                "duration": 150,
                "onset_time": 0,
                "offset_time": 75,
            }
        )
        cls.model = Model(nest, 8, model_params)
        cls.sheet = VisualCorticalUniformSheet3D(
            cls.model, ParameterSet(cls.sheet_params)
        )
        cls.sheet.record()
        cls.duration = cls.opt_array_params.stimulating_signal_parameters.duration

    def record_and_retrieve_data(self, ds, duration):
        self.model.reset()
        self.sheet.prepare_artificial_stimulation(duration, 0, [ds])
        self.model.run(duration)
        ds.inactivate(duration)
        return (
            self.sheet.get_data(duration).analogsignals[0]
            - self.sheet_params["cell"]["params"]["v_rest"] * qt.mV
        )

    @pytest.mark.parametrize("proportion", [0.25, 0.5, 0.75, 1.0])
    def test_transfection_proportion(self, proportion):
        sap = MozaikExtendedParameterSet(deepcopy(self.opt_array_params))
        sap.transfection_proportion = 1.0
        ds = OpticalStimulatorArrayChR(self.sheet, sap)
        stim_1 = ds.stimulated_cells
        sap.transfection_proportion = proportion
        ds = OpticalStimulatorArrayChR(self.sheet, sap)
        stim_p = ds.stimulated_cells
        assert set(stim_p).issubset(set(stim_1))
        assert np.isclose(len(stim_p) / len(stim_1), proportion, atol=0.02)

    def test_stimulated_cells(self):
        sap = MozaikExtendedParameterSet(deepcopy(self.opt_array_params))
        ds = OpticalStimulatorArrayChR(self.sheet, sap)
        d = self.record_and_retrieve_data(ds, self.duration)
        d = d.sum(axis=0)
        recorded_cells = ds.sheet.pop.all_cells[self.sheet.to_record["v"]]
        stim_ids = set(ds.stimulated_cells)
        for i in range(len(recorded_cells)):
            if recorded_cells[i] in stim_ids:
                assert d[i] != 0, "Zero input to neuron in stimulated_cells!"
            else:
                assert d[i] == 0, "Nonzero input to neuron not in stimulated_cells!"

    @pytest.mark.parametrize("onset_time", np.random.randint(0, 250, 4))
    @pytest.mark.parametrize("stim_duration", np.random.randint(0, 50, 4))
    @pytest.mark.parametrize("time_after_offset", np.random.randint(0, 250, 4))
    def test_duration_independence(self, onset_time, stim_duration, time_after_offset):
        # Ensure that the odeint solver works irrespective of stimulation duration
        sap = MozaikExtendedParameterSet(deepcopy(self.opt_array_params))
        sap.stimulating_signal_parameters.onset_time = onset_time
        sap.stimulating_signal_parameters.offset_time = onset_time + stim_duration
        sap.stimulating_signal_parameters.duration = (
            onset_time + stim_duration + time_after_offset
        )
        ds = OpticalStimulatorArrayChR(self.sheet, sap)
        assert ds.mixed_signals_current.sum() != 0

    @pytest.mark.skip
    # TODO: Fix the fact that we are unable to use reset in NEST 3!!!!!
    def test_scs_sharing(self):
        radii = np.arange(50, 200.1, 50)
        shared_scs = {}

        for radius in radii:
            sap = MozaikExtendedParameterSet(deepcopy(self.opt_array_params))
            sap.stimulating_signal_parameters.radius = radius

            ds = OpticalStimulatorArrayChR(self.sheet, sap, shared_scs)
            shared_scs.update(
                {ds.stimulated_cells[i]: ds.scs[i] for i in range(len(ds.scs))}
            )
            d_share = self.record_and_retrieve_data(ds, self.duration)

            ds = OpticalStimulatorArrayChR(self.sheet, sap)
            d_no_share = self.record_and_retrieve_data(ds, self.duration)

            assert np.all(d_share == d_no_share)

    @pytest.mark.skip
    def test_scs_optimization(self):
        shared_scs = None
        shared_scs_optimized = {}
        radii = np.arange(50, 200.1, 50)

        for radius in radii:
            sap = MozaikExtendedParameterSet(deepcopy(self.opt_array_params))
            sap.stimulating_signal_parameters.radius = radius

            ds = OpticalStimulatorArrayChR(
                self.sheet, self.opt_array_params, shared_scs_optimized
            )

            shared_scs_optimized.update(
                {ds.stimulated_cells[i]: ds.scs[i] for i in range(len(ds.scs))}
            )
            d1 = self.record_and_retrieve_data(ds, self.duration)

            ds = OpticalStimulatorArrayChR(
                self.sheet,
                self.opt_array_params,
                shared_scs=shared_scs,
                optimized_scs=False,
            )
            shared_scs = ds.scs
            d2 = self.record_and_retrieve_data(ds, self.duration)

            assert np.array_equal(d1, d2)

    def plot_max_response(self, d1, d2):
        import matplotlib.pyplot as plt

        idx = np.argmax(d2.sum(axis=0))
        plt.plot(d1[:, idx])
        plt.plot(d2[:, idx])
        plt.show()
