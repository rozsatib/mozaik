import pickle
import pytest
import numpy as np
import quantities as qt
from mozaik.models import Model
from parameters import ParameterSet
from mozaik.sheets.vision import VisualCorticalUniformSheet3D
import mozaik
from mozaik.stimuli import InternalStimulus
import copy
from mozaik.tools.distribution_parametrization import (
    load_parameters,
    MozaikExtendedParameterSet,
)
import pathlib


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


@pytest.fixture(scope="class")
def test_env(request):
    from pyNN import nest

    test_dir = str(pathlib.Path(__file__).parent.parent)

    model_params = load_parameters(test_dir + "/sheets/model_params")
    model_params.null_stimulus_period = 200

    sheet_params = load_parameters(test_dir + "/sheets/exc_sheet_params")
    sheet_params.min_depth = 100
    sheet_params.max_depth = 400

    opt_array_params = load_parameters(test_dir + "/sheets/opt_array_params")
    opt_array_params["transfection_proportion"] = 1.0

    sheet_params.artificial_stimulators = {
        "stimulator": {
            "component": "mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR",
            "params": opt_array_params,
        }
    }

    model = Model(nest, 8, model_params)
    sheet = VisualCorticalUniformSheet3D(model, ParameterSet(sheet_params))
    sheet.record()

    ds = sheet.artificial_stimulators["stimulator"]

    request.cls.model = model
    request.cls.sheet = sheet
    request.cls.ds = ds
    request.cls.sheet_params = sheet_params
    request.cls.duration = 150
    request.cls.test_dir = test_dir

    """
    The first recording in a simulation lasts one min_delay longer for some
    reason, which messes up recording comparisons, so we do a short dummy
    recording in the beginning.
    TODO: Remove once this Github issue is resolved:
    https://github.com/NeuralEnsemble/PyNN/issues/759
    """
    request.cls.record_and_retrieve_data(ds, 1)

    yield
    model.sheets.clear()


@pytest.mark.usefixtures("test_env")
class TestOpticalStimulatorArrayChR:

    def create_unity_radprof(self, h=4, w=424):
        radprof = np.zeros((h, w))
        # radprof[:, :5] = 1
        radprof[:, :5] = 1
        f = open(self.test_dir + "/sheets/unity_radprof.pickle", "wb")
        pickle.dump(radprof, f)
        f.close()

    @classmethod
    def generate_stimulus(cls, model, duration, onset=None, offset=None):
        signal_function = mozaik.sheets.direct_stimulator.stimulating_pattern_flash
        p = {
            "shape": "circle",
            "coords": [[0, 0]],
            "radius": 125,
            "intensity": [1],
            "onset_time": 0,
        }
        p["duration"] = duration
        if onset:
            p["onset_time"] = onset
        if offset:
            p["offset_time"] = offset
        else:
            p["offset_time"] = int(np.floor(duration / 2))
        # next(iter(...)) selects the "first" element of a dict. Here there's only 1
        sheet = next(iter(model.sheets.values()))
        stimulator = next(iter(sheet.artificial_stimulators.values()))
        x, y, dt = (
            stimulator.stimulator_coords_x,
            stimulator.stimulator_coords_y,
            stimulator.parameters.update_interval,
        )
        signal = signal_function(sheet, x, y, dt, ParameterSet(p))
        stimulus = InternalStimulus(
            frame_duration=p["duration"],
            duration=p["duration"],
            trial=0,
            direct_stimulation_name=type(stimulator).__name__,
            direct_stimulation_parameters=MozaikExtendedParameterSet(
                {"signal_function_parameters": p}
            ),
        )
        stimulus.direct_stimulation_signals = [signal]
        return stimulus

    @classmethod
    def record_and_retrieve_data(cls, ds, duration):
        cls.model.reset()
        stim = cls.generate_stimulus(cls.model, duration)
        cls.sheet.prepare_artificial_stimulation(
            stim, duration, cls.model.simulator_time, [ds]
        )
        cls.model.run(duration)
        ds.inactivate(cls.model.simulator_time)
        return np.array(
            cls.sheet.get_data(duration).analogsignals[0]
            - cls.sheet_params["cell"]["params"]["v_rest"] * qt.mV
        )

    @pytest.mark.parametrize("A", [np.random.rand(50, 50, 10) for i in range(5)])
    def test_compress_decompress(self, A):
        self.create_unity_radprof()
        A_compressed = self.ds.compress_array(A)
        A_decompressed = self.ds.decompress_array(A_compressed)
        assert np.all(A == A_decompressed)

    @pytest.mark.parametrize("proportion", [0.25, 0.5, 0.75, 1.0])
    def test_transfection_proportion(self, proportion):
        sheet_params = copy.deepcopy(self.sheet_params)
        sheet_params["name"] = sheet_params["name"] + f"_transfection_{proportion}"
        sheet_params["artificial_stimulators"]["stimulator"]["params"][
            "transfection_proportion"
        ] = proportion
        sheet = VisualCorticalUniformSheet3D(self.model, ParameterSet(sheet_params))
        ds = sheet.artificial_stimulators["stimulator"]
        assert np.isclose(len(ds.active_cells) / sheet.pop.size, proportion, atol=0.02)
        del self.model.sheets[sheet_params["name"]]  # de-register sheet from model

    def test_stimulated_cells(self):
        d = self.record_and_retrieve_data(self.ds, self.duration).sum(axis=0)
        msp = self.ds.mixed_signals_photo.mean(axis=1)
        for i, dj in zip(self.sheet.to_record["v"], d):
            if msp[i] > 0:
                assert dj != 0, "Zero input to neuron in stimulated_cells!"
            else:
                assert dj < 1e-11, "Nonzero input to neuron not in stimulated_cells!"

    @pytest.mark.parametrize("onset_time", np.random.randint(0, 250, 4))
    @pytest.mark.parametrize("stim_duration", np.random.randint(0, 50, 4))
    @pytest.mark.parametrize("time_after_offset", np.random.randint(0, 250, 4))
    def test_duration_independence(self, onset_time, stim_duration, time_after_offset):
        offset_time = onset_time + stim_duration
        duration = offset_time + time_after_offset
        # Ensure that the odeint solver works irrespective of stimulation duration
        stim = self.generate_stimulus(self.model, duration)
        self.sheet.prepare_artificial_stimulation(
            stim, duration, self.model.simulator_time, [self.ds]
        )
        assert self.ds.mixed_signals_current.sum() != 0

    def plot_max_response(self, d1, d2):
        import matplotlib.pyplot as plt

        idx = np.argmax(d2.sum(axis=0))
        plt.plot(d1[:, idx])
        plt.plot(d2[:, idx])
        plt.show()
