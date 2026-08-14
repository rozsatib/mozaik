import pickle
import pytest
import numpy as np
import quantities as qt
from copy import deepcopy
from mozaik.models import Model
from mozaik.sheets.direct_stimulator import (
    ClosedLoopOpticalStimulatorArray,
    OpticalStimulatorArrayChR,
)
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
    PyNNDistribution,
    MozaikExtendedParameterSet,
)

test_dir = None
PARAM_RNG = np.random.RandomState(1729)


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
    mozaik.setup_seeds(
        model_seed=model_params["model_seed"],
        simulation_seed=model_params["simulation_seed"],
        experiment_seed=model_params["experiment_seed"],
    )
    mozaik.setup_mpi()

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
        opt_array_params["actuation_delay"] = None
        opt_array_params["feedback_sheet_names"] = [sheet_params.name]

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

    def create_unity_radprof(self, h=20, w=100):
        radprof = np.zeros((h, w))
        radprof[:, 0] = 1
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

    @pytest.mark.parametrize("A", [PARAM_RNG.rand(50, 50, 10) for i in range(5)])
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

    @pytest.mark.parametrize("onset_time", PARAM_RNG.randint(0, 250, 4))
    @pytest.mark.parametrize("stim_duration", PARAM_RNG.randint(0, 50, 4))
    @pytest.mark.parametrize("time_after_offset", PARAM_RNG.randint(0, 250, 4))
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


class TestOpticalStimulatorArrayChRIntegratedBackend:
    """Test ``OpticalStimulatorArrayChR`` with integrated current schedules.

    These tests cover MPI-local optical calculations and schedule programming,
    rank-independent transfection, update timing, and equivalence to the legacy
    external PyNN current-source backend.
    """

    STIMULATION_START = 5.0
    STIMULATION_DURATION = 30.0
    STIMULATION_END = STIMULATION_START + STIMULATION_DURATION
    POST_STIMULATION_DURATION = 5.0

    # Minimal PyNN population API for testing arbitrary MPI ownership masks
    # without launching multiple MPI processes.
    class _FakeDistributedPopulation:
        class _FakeScheduleCell:
            def __init__(self, population_index, local):
                self.population_index = population_index
                self.local = local
                self.parameters = []

            def set_parameters(self, **parameters):
                self.parameters.append(parameters)

        def __init__(self, positions, local_mask):
            self.positions = positions
            self.size = positions.shape[1]
            self.celltype = SimpleNamespace(
                has_parameter=lambda name: name
                in ("amplitude_times", "amplitude_values")
            )
            self._mask_local = np.asarray(local_mask, dtype=bool)
            self.all_cells = np.array(
                [
                    self._FakeScheduleCell(index, self._mask_local[index])
                    for index in range(self.size)
                ],
                dtype=object,
            )

        @property
        def local_cells(self):
            return self.all_cells[self._mask_local]

        @staticmethod
        def id_to_index(cell):
            return cell.population_index

    @staticmethod
    def _setup_test_seeds():
        mozaik.setup_seeds(
            model_seed=1024,
            simulation_seed=513,
            experiment_seed=0,
        )
        mozaik.setup_mpi()

    @classmethod
    def _integrated_optical_sheet(cls, local_mask):
        positions = np.array(
            [
                [-0.1, 0.1, -0.05, 0.05],
                [-0.1, -0.1, 0.1, 0.1],
                [100.0, 200.0, 300.0, 400.0],
            ]
        )
        return SimpleNamespace(
            name="integrated",
            parameters=ParameterSet(
                {
                    "min_depth": 100.0,
                    "max_depth": 400.0,
                    "cell": {
                        "model": "aeif_cond_exp_sc",
                        "native_nest": True,
                    },
                }
            ),
            pop=cls._FakeDistributedPopulation(positions, local_mask),
            vf_2_cs=lambda x, y: (x, y),
            sim=SimpleNamespace(get_current_time=lambda: 0.0),
            dt=0.1,
        )

    @staticmethod
    def _parameters(transfection_proportion):
        return ParameterSet(
            {
                "size": 400.0,
                "spacing": 20.0,
                "update_interval": 1.0,
                "depth_sampling_step": 10.0,
                "light_source_light_propagation_data": (
                    "tests/sheets/unity_radprof.pickle"
                ),
                "transfection_proportion": transfection_proportion,
            }
        )

    def test_calculates_and_programs_only_mpi_local_neurons(self):
        self._setup_test_seeds()
        sheet = self._integrated_optical_sheet([True, False, True, False])
        ds = OpticalStimulatorArrayChR(sheet, self._parameters(1.0))

        assert np.array_equal(ds.optical_calc_population_indices, [0, 2])
        assert ds.mixed_signals_photo.shape[0] == 2

        signal = np.ones(ds.stimulator_coords_x.shape + (3,))
        ds.set_input(signal)
        ds.prepare_stimulation(duration=3.0, offset=0.0)

        programmed = [
            index for index, cell in enumerate(sheet.pop.all_cells) if cell.parameters
        ]
        assert programmed == [0, 2]
        assert all(
            np.array_equal(cell.parameters[0]["amplitude_times"], [0.0, 1.0, 2.0])
            for cell in sheet.pop.local_cells
        )

    def test_transfection_is_independent_of_mpi_ownership(self):
        parameters = self._parameters(0.5)

        self._setup_test_seeds()
        first = OpticalStimulatorArrayChR(
            self._integrated_optical_sheet([True, False, True, False]), parameters
        )

        self._setup_test_seeds()
        second = OpticalStimulatorArrayChR(
            self._integrated_optical_sheet([False, True, False, True]), parameters
        )

        assert np.array_equal(first.transfection_mask, second.transfection_mask)

    def test_later_schedule_moves_only_current_time_to_next_step(self):
        self._setup_test_seeds()
        sheet = self._integrated_optical_sheet([True, False, True, False])
        sheet.sim.get_current_time = lambda: 5.0
        ds = OpticalStimulatorArrayChR(sheet, self._parameters(1.0))

        assert np.array_equal(
            ds._integrated_cs_neuron_amplitude_times([5.0, 6.0, 9.0]),
            [5.1, 6.0, 9.0],
        )

    def test_closed_loop_programs_integrated_local_current_schedules(self):
        # Check that closed-loop updates program only rank-local integrated cells.
        self._setup_test_seeds()
        sheet = self._integrated_optical_sheet([True, False, True, False])
        parameters = self._parameters(1.0)
        parameters["state_update_interval"] = 3.0
        parameters["actuation_delay"] = None
        parameters["feedback_sheet_names"] = [sheet.name]
        ds = ClosedLoopOpticalStimulatorArray(sheet, parameters)
        ds.calculate_input_function = lambda stimulator: np.ones(
            stimulator.stimulator_coords_x.shape + (3,)
        )

        ds.set_input_segment()
        ds.stimulation_duration = parameters.state_update_interval
        ds.prepare_stimulation(duration=3.0, offset=0.0)

        programmed = [
            index for index, cell in enumerate(sheet.pop.all_cells) if cell.parameters
        ]
        assert programmed == [0, 2]
        assert ds.integrated_cs
        assert ds.scs == []
        assert ds.actuation_delay() == 2 * sheet.dt

    @pytest.mark.parametrize(
        ("requested_delay", "expected_delay"),
        [(None, 0.2), (0.4, 0.4)],
    )
    def test_closed_loop_actuation_delay_selection(
        self, requested_delay, expected_delay
    ):
        # Verify None selects the backend minimum while an explicit delay is kept.
        self._setup_test_seeds()
        sheet = self._integrated_optical_sheet([True, False, True, False])
        parameters = self._parameters(1.0)
        parameters["state_update_interval"] = 3.0
        parameters["actuation_delay"] = requested_delay
        parameters["feedback_sheet_names"] = [sheet.name]

        ds = ClosedLoopOpticalStimulatorArray(sheet, parameters)

        assert ds.actuation_delay() == expected_delay

    @pytest.mark.parametrize("actuation_delay", [-1.0, 0.1])
    def test_closed_loop_rejects_actuation_delay_below_minimum(self, actuation_delay):
        # Ensure negative and positive delays below the backend minimum are rejected.
        self._setup_test_seeds()
        sheet = self._integrated_optical_sheet([True, False, True, False])
        parameters = self._parameters(1.0)
        parameters["state_update_interval"] = 3.0
        parameters["actuation_delay"] = actuation_delay
        parameters["feedback_sheet_names"] = [sheet.name]

        with pytest.raises(AssertionError, match="shorter than the .* minimum"):
            ClosedLoopOpticalStimulatorArray(sheet, parameters)

    def test_closed_loop_rejects_empty_feedback_sheet_names(self):
        # Ensure every controller explicitly names at least one feedback sheet.
        self._setup_test_seeds()
        sheet = self._integrated_optical_sheet([True, False, True, False])
        parameters = self._parameters(1.0)
        parameters["state_update_interval"] = 3.0
        parameters["actuation_delay"] = None
        parameters["feedback_sheet_names"] = []

        with pytest.raises(AssertionError, match="must not be empty"):
            ClosedLoopOpticalStimulatorArray(sheet, parameters)

    @staticmethod
    def _response_sheet_parameters(integrated):
        if integrated:
            cell = {
                "model": "aeif_cond_exp_sc",
                "native_nest": True,
                "params": {
                    "E_L": -80.0,
                    "V_reset": -60.0,
                    "t_ref": 2.0,
                    "C_m": 32.0,
                    "g_L": 4.0,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "tau_syn_ex": 1.5,
                    "tau_syn_in": 4.2,
                    "a": -0.8,
                    "b": 80.0,
                    "Delta_T": 0.8,
                    "tau_w": 1.0,
                    "V_th": -56.0,
                    "V_peak": -40.0,
                    "I_e": 0.0,
                },
                "receptors": None,
                "initial_values": {"V_m": -60.0},
            }
        else:
            cell = {
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
                    "a": -0.8,
                    "b": 0.08,
                    "delta_T": 0.8,
                    "tau_w": 1.0,
                    "v_thresh": -56.0,
                },
                "receptors": None,
                "initial_values": {"v": -60.0},
            }

        return ParameterSet(
            {
                "name": "integrated" if integrated else "external",
                "sx": 100.0,
                "sy": 100.0,
                "min_depth": 200.0,
                "max_depth": 200.0,
                "density": 100.0,
                "mpi_safe": False,
                "magnification_factor": 1000.0,
                "cell": cell,
                "artificial_stimulators": {
                    "stimulator": {
                        "component": (
                            "mozaik.sheets.direct_stimulator."
                            "OpticalStimulatorArrayChR"
                        ),
                        "params": {
                            "size": 400.0,
                            "spacing": 20.0,
                            "update_interval": 1.0,
                            "depth_sampling_step": 10.0,
                            "light_source_light_propagation_data": (
                                "tests/sheets/unity_radprof.pickle"
                            ),
                            "transfection_proportion": 1.0,
                        },
                    }
                },
                "recording_interval": 0.1,
                "recorders": {},
            }
        )

    @classmethod
    def _simulate_response(cls, integrated):
        import nest
        from pyNN import nest as sim

        if "aeif_cond_exp_sc" not in nest.node_models:
            nest.Install("stepcurrentmodule")

        test_dir = str(pathlib.Path(__file__).parent.parent)
        model_parameters = load_parameters(test_dir + "/sheets/model_params")
        mozaik.setup_seeds(
            model_seed=model_parameters.model_seed,
            simulation_seed=model_parameters.simulation_seed,
            experiment_seed=model_parameters.experiment_seed,
        )
        mozaik.setup_mpi()
        model = Model(sim, 2, model_parameters)
        sheet = VisualCorticalUniformSheet3D(
            model, cls._response_sheet_parameters(integrated)
        )
        excitatory_input = sim.Population(
            1,
            sim.SpikeSourceArray(spike_times=[3.0, 7.0, 11.0, 17.0, 23.0]),
        )
        inhibitory_input = sim.Population(
            1,
            sim.SpikeSourceArray(spike_times=[5.0, 13.0, 19.0, 25.0]),
        )
        sim.Projection(
            excitatory_input,
            sheet.pop,
            sim.OneToOneConnector(),
            synapse_type=sim.StaticSynapse(weight=0.002, delay=0.1),
            receptor_type="excitatory",
        )
        sim.Projection(
            inhibitory_input,
            sheet.pop,
            sim.OneToOneConnector(),
            synapse_type=sim.StaticSynapse(weight=0.003, delay=0.1),
            receptor_type="inhibitory",
        )
        stimulator = sheet.artificial_stimulators["stimulator"]
        voltage_variable = "V_m" if integrated else "v"
        sheet.pop.record("spikes")
        sheet.pop.record(voltage_variable, sampling_interval=0.1)

        signal = np.zeros(
            stimulator.stimulator_coords_x.shape + (int(cls.STIMULATION_DURATION),)
        )
        signal[:, :, 2:20] = 1.0
        model.run(cls.STIMULATION_START)
        stimulator.set_input(signal)
        stimulator.prepare_stimulation(
            cls.STIMULATION_DURATION, offset=cls.STIMULATION_START
        )
        current = stimulator.mixed_signals_current.copy()
        positions = sheet.pop.positions.copy()

        model.run(cls.STIMULATION_DURATION)
        stimulator.inactivate(offset=cls.STIMULATION_END)
        model.run(cls.POST_STIMULATION_DURATION)
        segment = sheet.pop.get_data(["spikes", voltage_variable]).segments[-1]
        voltage = np.asarray(segment.analogsignals[0]).copy()
        spikes = np.asarray(segment.spiketrains[0]).copy()
        sim.end()
        return positions, current, voltage, spikes

    def test_integrated_and_external_cs_produce_identical_responses(self):
        # PyNN current source updates reach neurons after 3*dt (0.3 ms); integrated
        # current source updates take dt (0.1 ms). Activation schedules which were set
        # before presenting stimuli are aligned, but runtime inactivation differs by 2*dt.
        # Thus, for voltages after the offset of stimulation we use a relative bound.
        # For positions, calculated currents, spikes, and pre-offset voltages we use
        # exact comparison.
        external = self._simulate_response(integrated=False)
        integrated = self._simulate_response(integrated=True)

        assert np.ptp(external[2]) > 0
        assert external[3].size > 0
        np.testing.assert_equal(integrated[0], external[0])
        np.testing.assert_equal(integrated[1], external[1])
        np.testing.assert_equal(integrated[3], external[3])

        # Split at deactivation: voltage before it must match exactly; afterwards,
        # allow for the 2*dt (0.2 ms) difference in zero-current delivery.
        times = np.arange(external[2].shape[0]) * 0.1
        before_offset = times < self.STIMULATION_END
        from_offset = times >= self.STIMULATION_END
        np.testing.assert_equal(
            integrated[2][before_offset],
            external[2][before_offset],
        )
        post_offset_relative_error = np.abs(
            integrated[2][from_offset] - external[2][from_offset]
        ) / np.maximum(np.abs(external[2][from_offset]), np.finfo(float).eps)
        assert np.max(post_offset_relative_error) <= 0.01


@pytest.mark.usefixtures("test_env")
class TestClosedLoopOpticalStimulatorArray:
    stimulator_component = (
        "mozaik.sheets.direct_stimulator.ClosedLoopOpticalStimulatorArray"
    )

    def test_external_actuation_delay_uses_backend_minimum(self):
        # Confirm the minimum sentinel includes NEST's configured source delay.
        assert self.ds.actuation_delay() == self.ds.scs[0].min_delay + 2 * self.sheet.dt

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

    def test_get_data_all_sheets_returns_configured_cache_without_retrieval(
        self, monkeypatch
    ):
        # Ensure the public accessor reads the configured cache without MPI retrieval.
        first_data, second_data = object(), object()
        first = self._fake_sheet("cortex_1", {"spikes": [0]}, first_data)
        unrecorded = self._fake_sheet("unrecorded", {})
        second = self._fake_sheet("cortex_2", {"v": [0]}, second_data)
        lgn = SimpleNamespace(name="X_ON", to_record={"spikes": [0]}, get_data=Mock())
        self._set_sheets(monkeypatch, first, lgn, unrecorded, second)
        self.ds.parameters.feedback_sheet_names = [
            first.name,
            unrecorded.name,
            second.name,
        ]

        self.ds._collect_sheet_data()
        first.get_data.reset_mock()
        second.get_data.reset_mock()

        data = self.ds.get_data_all_sheets()

        assert list(data) == [first.name, unrecorded.name, second.name]
        assert data[first.name] is first_data
        assert data[unrecorded.name] == []
        assert data[second.name] is second_data
        first.get_data.assert_not_called()
        second.get_data.assert_not_called()
        unrecorded.get_data.assert_not_called()
        lgn.get_data.assert_not_called()

    def test_get_data_all_sheets_caches_by_simulator_time(self, monkeypatch):
        # Check that automatic collection retrieves each sheet at most once per time.
        current_time = [10.0]
        sheet = self._fake_sheet("recorded", {"spikes": [0]})
        sheet.get_data.side_effect = [object(), object()]
        self._set_sheets(monkeypatch, sheet)
        self.ds.parameters.feedback_sheet_names = [sheet.name]
        monkeypatch.setattr(self.sheet.sim, "get_current_time", lambda: current_time[0])

        self.ds._collect_sheet_data()
        first = self.ds.get_data_all_sheets()[sheet.name]
        self.ds._collect_sheet_data()
        second = self.ds.get_data_all_sheets()[sheet.name]

        assert first is second
        sheet.get_data.assert_called_once_with(clear=False)

        current_time[0] = 11.0
        self.ds._collect_sheet_data()
        third = self.ds.get_data_all_sheets()[sheet.name]

        assert third is not first
        assert sheet.get_data.call_count == 2

    def test_collect_sheet_data_respects_feedback_sheet_names(self, monkeypatch):
        # Verify automatic collection is limited to explicitly configured sheets.
        first = self._fake_sheet("first", {"v": [0]}, object())
        second = self._fake_sheet("second", {"v": [0]}, object())
        self._set_sheets(monkeypatch, first, second)
        self.ds.parameters.feedback_sheet_names = [second.name]

        self.ds._collect_sheet_data()

        first.get_data.assert_not_called()
        second.get_data.assert_called_once_with(clear=False)

    def test_direct_spike_only_feedback_skips_mozaik_retrieval(self, monkeypatch):
        # Keep final spike recording enabled without paying for Neo data every update.
        sheet_get_data = Mock()
        monkeypatch.setattr(self.ds, "use_direct_nest_spike_retrieval", True)
        monkeypatch.setattr(self.sheet, "to_record", {"spikes": [0, 2]})
        monkeypatch.setattr(self.sheet, "get_data", sheet_get_data)
        self.ds.parameters.feedback_sheet_names = [self.sheet.name]

        self.ds._collect_sheet_data()

        sheet_get_data.assert_not_called()
        assert self.ds.feedback_data == {self.sheet.name: []}

    @pytest.mark.parametrize("retrieve_all", [True, False])
    def test_update_state_retrieval_paths(self, monkeypatch, retrieve_all):
        # Ensure feedback is prefetched whether or not the callback reads the cache.
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
        self.ds.parameters.feedback_sheet_names = [self.sheet.name, other.name]
        retrieved = {}

        def update_controller(stimulator):
            if retrieve_all:
                retrieved.update(stimulator.get_data_all_sheets())

        monkeypatch.setattr(self.ds, "update_state_function", update_controller)

        self.ds.update_state()

        fast_get_data.assert_called_once_with()
        attached_get_data.assert_called_once_with(clear=False)
        other.get_data.assert_called_once_with(clear=False)
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

    def test_non_controller_rank_uses_broadcast_input_without_running_callback(
        self, monkeypatch
    ):
        # Check that MPI worker ranks consume rank 0's signal without running user code.
        signal = np.ones(self.ds.stimulator_coords_x.shape + (6,))
        calculate_input = Mock()
        communicator = SimpleNamespace(
            rank=1,
            size=2,
            bcast=Mock(return_value=signal),
        )
        monkeypatch.setattr(mozaik, "mpi_comm", communicator)
        monkeypatch.setattr(self.ds, "calculate_input_function", calculate_input)

        received = self.ds.calculate_input_signal()

        calculate_input.assert_not_called()
        communicator.bcast.assert_called_once_with(None, root=mozaik.MPI_ROOT)
        assert received is signal

    def test_mozaik_spike_path_does_not_create_direct_recorder(self, monkeypatch):
        # TODO: Remove this test if the temporary direct last-spikes path is removed.
        # Verify the standard Mozaik path does not allocate a redundant NEST recorder.
        create_recorder = Mock()
        monkeypatch.setattr(self.ds, "use_direct_nest_spike_retrieval", False)
        monkeypatch.setattr(self.sheet, "to_record", {"v": [0, 2]})
        monkeypatch.setattr(
            mozaik.sheets.direct_stimulator,
            "nest",
            SimpleNamespace(Create=create_recorder),
        )

        self.ds.set_data_recording()

        create_recorder.assert_not_called()
        assert self.ds.spike_recorder is None

    def test_non_controller_rank_does_not_run_state_callback(self, monkeypatch):
        # Ensure MPI workers program the broadcast update but never invoke controller state.
        update_controller = Mock()
        communicator = SimpleNamespace(rank=1, size=2)
        monkeypatch.setattr(mozaik, "mpi_comm", communicator)
        monkeypatch.setattr(self.ds, "update_state_function", update_controller)
        monkeypatch.setattr(self.ds, "_collect_sheet_data", Mock())
        monkeypatch.setattr(self.ds, "get_data", Mock())
        monkeypatch.setattr(self.ds, "set_input_segment", Mock())
        monkeypatch.setattr(self.ds, "_set_stimulation_current_schedule", Mock())
        monkeypatch.setattr(self.ds, "times", np.array([0.0]), raising=False)

        self.ds.update_state()

        update_controller.assert_not_called()
        self.ds.set_input_segment.assert_called_once_with()
        self.ds._set_stimulation_current_schedule.assert_called_once()
        np.testing.assert_array_equal(
            self.ds._set_stimulation_current_schedule.call_args.args[0], [6.0]
        )
