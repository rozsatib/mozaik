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
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

from neo.core import AnalogSignal
from mozaik.tools.distribution_parametrization import (
    load_parameters,
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

    stimulator_component = getattr(
        request.cls,
        "stimulator_component",
        "mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR",
    )
    if stimulator_component.endswith("ClosedLoopOpticalStimulatorArray"):
        opt_array_params["state_update_interval"] = 6.0

    sheet_params.artificial_stimulators = {
        "stimulator": {
            "component": stimulator_component,
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


@pytest.mark.usefixtures("test_env")
class TestClosedLoopOpticalStimulatorArray:
    stimulator_component = (
        "mozaik.sheets.direct_stimulator.ClosedLoopOpticalStimulatorArray"
    )

    @pytest.mark.parametrize(
        ("center", "radius"),
        [([0.0, 0.0], 50.0), ([80.0, -60.0], 100.0)],
    )
    def test_recorded_neuron_positions(self, center, radius):
        """Check that reported neuron positions match where the optogenetic
        stimulation was applied."""
        signal = mozaik.sheets.direct_stimulator.stimulating_pattern_flash(
            self.sheet,
            self.ds.stimulator_coords_x,
            self.ds.stimulator_coords_y,
            self.ds.parameters.update_interval,
            ParameterSet(
                {
                    "shape": "circle",
                    "coords": [center],
                    "radius": radius,
                    "intensity": [1.0],
                    "onset_time": 0.0,
                    "offset_time": 2.0,
                    "duration": 2.0,
                }
            ),
        )
        photo = self.ds.calculate_photo(signal)
        indices = self.ds.recorded_neuron_indices("v")
        positions = self.ds.recorded_neuron_positions("v")

        assert positions.shape == (3, len(indices))
        np.testing.assert_allclose(positions[2], self.sheet.pop[indices].positions[2])

        stimulated = np.any(photo[indices] > 0, axis=1)
        assert np.any(stimulated)
        distances = np.sqrt(
            (positions[0, stimulated] - center[0]) ** 2
            + (positions[1, stimulated] - center[1]) ** 2
        )
        assert np.all(distances <= radius + self.ds.parameters.spacing)

    @staticmethod
    def _fake_sheet(name, to_record, data=None):
        return SimpleNamespace(
            name=name,
            to_record=to_record,
            last_recording=data,
            vf_2_cs=lambda x, y: (x, y),
            get_data=Mock(return_value=data),
        )

    @staticmethod
    def _analog_recording(value):
        return SimpleNamespace(
            analogsignals=[
                AnalogSignal(
                    np.full((3, 1), value),
                    units=qt.mV,
                    sampling_period=1 * qt.ms,
                    name="v",
                )
            ],
            spiketrains=[],
        )

    def _set_sheets(self, monkeypatch, *sheets):
        monkeypatch.setattr(
            self.model, "sheets", {sheet.name: sheet for sheet in sheets}
        )

    @pytest.fixture(autouse=True)
    def _clear_sheet_data_cache(self):
        self.ds._sheet_data_cache.clear()

    def test_get_data_all_sheets_selects_and_retrieves_cortical_sheets(
        self, monkeypatch
    ):
        first_data, second_data = object(), object()
        first = self._fake_sheet("cortex_1", {"spikes": [0]}, first_data)
        unrecorded = self._fake_sheet("unrecorded", {})
        second = self._fake_sheet("cortex_2", {"v": [0]}, second_data)
        lgn = SimpleNamespace(name="X_ON", to_record={"spikes": [0]}, get_data=Mock())
        self._set_sheets(monkeypatch, first, lgn, unrecorded, second)

        data = self.ds.get_data_all_sheets()

        assert list(data) == [first.name, unrecorded.name, second.name]
        assert data[first.name] is first_data
        assert data[unrecorded.name] == []
        assert data[second.name] is second_data
        first.get_data.assert_called_once_with(clear=False)
        second.get_data.assert_called_once_with(clear=False)
        unrecorded.get_data.assert_not_called()
        lgn.get_data.assert_not_called()

    def test_get_data_all_sheets_caches_by_simulator_time(self, monkeypatch):
        current_time = [10.0]
        sheet = self._fake_sheet("recorded", {"spikes": [0]})
        sheet.get_data.side_effect = [object(), object()]
        self._set_sheets(monkeypatch, sheet)
        monkeypatch.setattr(self.sheet.sim, "get_current_time", lambda: current_time[0])

        first = self.ds.get_data_all_sheets()[sheet.name]
        second = self.ds.get_data_all_sheets()[sheet.name]

        assert first is second
        sheet.get_data.assert_called_once_with(clear=False)

        current_time[0] = 11.0
        third = self.ds.get_data_all_sheets()[sheet.name]

        assert third is not first
        assert sheet.get_data.call_count == 2

    @pytest.mark.parametrize(
        ("retrieve_all", "expected_other_calls"), [(True, 1), (False, 0)]
    )
    def test_update_state_retrieval_paths(
        self, monkeypatch, retrieve_all, expected_other_calls
    ):
        attached_data = object()
        attached_get_data = Mock(return_value=attached_data)
        other = self._fake_sheet("other_cortex", {"v": [0]}, object())
        fast_get_data = Mock()
        monkeypatch.setattr(self.sheet, "get_data", attached_get_data)
        monkeypatch.setattr(self.ds, "get_data", fast_get_data)
        monkeypatch.setattr(self.ds, "set_input_segment", Mock())
        monkeypatch.setattr(self.ds, "active_cells", np.array([], dtype=int))
        monkeypatch.setattr(self.ds, "times", np.array([0.0]), raising=False)
        monkeypatch.setattr(
            self.ds,
            "_cortical_sheets",
            lambda: {self.sheet.name: self.sheet, other.name: other},
        )
        retrieved = {}

        def update_controller(stimulator):
            if retrieve_all:
                retrieved.update(stimulator.get_data_all_sheets())

        monkeypatch.setattr(self.ds, "update_state_function", update_controller)

        self.ds.update_state()

        fast_get_data.assert_called_once_with()
        attached_get_data.assert_called_once_with(clear=False)
        assert other.get_data.call_count == expected_other_calls
        if retrieve_all:
            assert retrieved[self.sheet.name] is attached_data

    def test_get_recording_selects_sheet_and_handles_missing_data(self, monkeypatch):
        first = self._fake_sheet("cortex_1", {"v": [0]}, self._analog_recording(1.0))
        second = self._fake_sheet("cortex_2", {"v": [0]}, self._analog_recording(2.0))
        unrecorded = self._fake_sheet("unrecorded", {}, None)
        self._set_sheets(monkeypatch, first, second, unrecorded)

        for sheet, value in [(first, 1.0), (second, 2.0)]:
            np.testing.assert_array_equal(
                self.ds.get_recording("v", sheet_name=sheet.name),
                np.full((3, 1), value),
            )
        assert self.ds.get_recording("spikes", sheet_name=first.name) == []
        assert self.ds.get_recording("v", sheet_name=unrecorded.name) == []
