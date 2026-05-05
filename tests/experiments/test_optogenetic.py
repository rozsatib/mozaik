import pickle
import pytest
import numpy as np
import quantities as qt
from copy import deepcopy
from mozaik.models import Model
from parameters import ParameterSet
from mozaik.sheets.vision import VisualCorticalUniformSheet3D
from mozaik.tools.distribution_parametrization import (
    load_parameters,
    MozaikExtendedParameterSet,
)
from mozaik.connectors.vision import MapDependentModularConnectorFunction
from mozaik.tools.circ_stat import circular_dist
import scipy.stats
import pathlib
from mozaik.experiments.optogenetic import *


@pytest.fixture(scope="class")
def test_env(request):
    from pyNN import nest
    import mozaik

    mozaik.setup_mpi(mozaik_seed=1024, pynn_seed=1024)
    test_dir = str(pathlib.Path(__file__).parent.parent)

    model_params = load_parameters(test_dir + "/sheets/model_params")

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

    size = 400
    sheet_params["sx"] = size
    sheet_params["sy"] = size
    sheet_params["recorders"]["1"]["params"]["size"] = size
    opt_array_params["size"] = size

    model = Model(nest, 8, model_params)
    sheet = VisualCorticalUniformSheet3D(model, ParameterSet(sheet_params))
    sheet.record()

    request.cls.model = model
    request.cls.sheet = sheet
    request.cls.sheet_params = sheet_params
    request.cls.opt_array_params = opt_array_params
    request.cls.test_dir = test_dir
    request.cls.stim_array_list = [
        {
            "sheet": "exc_sheet",
            "name": "stimulator",
            "intensity_scaler": 1.0,
        }
    ]

    yield

    model.sheets.clear()


@pytest.mark.usefixtures("test_env")
class TestCorticalStimulationWithOptogeneticArray:

    def plot(self, ds):
        import matplotlib.pyplot as plt

        pos = self.sheet.pop.positions[0:2, :]
        plt.scatter(pos[0, :], pos[1, :])
        print(pos[:, np.argmin(np.sum(pos**2, axis=0))])
        plt.axis("equal")
        plt.xlim(-0.01, 0.01)
        plt.ylim(-0.01, 0.01)
        plt.show()

    def get_coords(self, neuron_ids):
        ac = np.array(list(self.sheet.pop.all()))
        pos = self.sheet.pop.positions[0:2, :]
        return pos[:, [np.where(ac == i)[0][0] for i in neuron_ids]] * 1000  # in µms

    def get_experiment(self, **params):
        raise NotImplementedError  # subclasses define this

    def get_stimulated_cells(self, exp, stimulus):
        ds = exp.direct_stimulation[0]["exc_sheet"][0]
        self.sheet.prepare_artificial_stimulation(
            stimulus, stimulus.duration, self.model.simulator_time, [ds]
        )
        return ds.sheet.pop.all_cells[ds.mixed_signals_photo.mean(axis=1) > 0]

    def stimulated_neuron_in_radius(self, exp, stimulus, invert=False):
        signal_parameters = stimulus.direct_stimulation_parameters[
            "signal_function_parameters"
        ]
        spacing = exp.direct_stimulation[0]["exc_sheet"][0].parameters.spacing
        coords = self.get_coords(self.get_stimulated_cells(exp, stimulus))
        d = np.sqrt(
            ((coords - np.array(signal_parameters["coords"]).T) ** 2).sum(axis=0)
        )
        if invert:
            return np.all(d >= signal_parameters["radius"] - spacing)
        else:
            return np.all(d <= signal_parameters["radius"] + spacing)

    @pytest.mark.skip("To be added later")
    def test_intensity_scaler(self, intensity_scaler):
        pass  # TODO: add intensity scaler output test

    def test_initial_assert(self):
        p = MozaikExtendedParameterSet(
            {
                "stimulator_array_list": self.stim_array_list,
                "num_trials": 1,
            }
        )
        with pytest.raises(AssertionError):
            p["stimulator_array_list"][0]["intensity_scaler"] = -1
            CorticalStimulationWithOptogeneticArray(self.model, p)
        p["stimulator_array_list"][0]["intensity_scaler"] = 1


class TestSingleOptogeneticArrayStimulus(TestCorticalStimulationWithOptogeneticArray):
    def get_experiment(self, x, y):
        return SingleOptogeneticArrayStimulus(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "num_trials": 1,
                    "stimulator_array_list": self.stim_array_list,
                    "stimulating_signal_function": "mozaik.sheets.direct_stimulator.single_pixel",
                    "stimulating_signal_function_parameters": ParameterSet(
                        {"x": x, "y": y, "intensity": 1, "duration": 1}
                    ),
                }
            ),
        )

    # Cortical to stimulator array coordinates)
    def c2a(self, c):
        return int((c + self.opt_array_params.size / 2) / self.opt_array_params.spacing)

    @pytest.mark.parametrize("x", np.random.randint(-10, 10, 7) * 20)
    @pytest.mark.parametrize("y", np.random.randint(-10, 10, 7) * 20)
    def test_random_pixels(self, x, y):
        size, spacing = self.opt_array_params.size, self.opt_array_params.spacing
        assert size == 400 and spacing == 20
        exp = self.get_experiment(x=x, y=y)
        coords = self.get_coords(self.get_stimulated_cells(exp, exp.stimuli[0]))
        assert np.all(np.isclose(coords[0, :], x, atol=spacing // 2))
        assert np.all(np.isclose(coords[1, :], y, atol=spacing // 2))


class TestOptogeneticArrayStimulusCircles(TestCorticalStimulationWithOptogeneticArray):
    def get_experiment(self, center, inverted):
        return OptogeneticArrayStimulusCircles(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "stimulator_array_list": self.stim_array_list,
                    "num_trials": 1,
                    "x_center": center[0],
                    "y_center": center[1],
                    "radii": [25, 50, 100, 150],
                    "intensities": [0.5, 1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "inverted": inverted,
                }
            ),
        )

    @pytest.mark.parametrize("center", [[0, 0], [0, 1], [1, 0], [1, 1]])
    @pytest.mark.parametrize("inverted", [False, True])
    def test_stimulated_neurons_in_radius(self, center, inverted):
        exp = self.get_experiment(center=center, inverted=inverted)
        for stim in exp.stimuli:
            assert self.stimulated_neuron_in_radius(exp, stim, inverted)


class TestOptogeneticArrayStimulusHexagonalTiling(
    TestCorticalStimulationWithOptogeneticArray
):
    def get_experiment(self, center, radius):
        return OptogeneticArrayStimulusHexagonalTiling(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "stimulator_array_list": self.stim_array_list,
                    "num_trials": 1,
                    "x_center": center[0],
                    "y_center": center[1],
                    "radius": radius,
                    "intensities": [0.5],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "angle": 0,
                    "shuffle": False,
                }
            ),
        )

    @pytest.mark.parametrize("center", [[0, 0], [0, 1], [1, 0], [1, 1]])
    @pytest.mark.parametrize("radius", [25, 50])
    def test_stimulated_neurons_in_radius(self, center, radius):
        exp = self.get_experiment(center=center, radius=radius)
        for stim in exp.stimuli:
            assert self.stimulated_neuron_in_radius(exp, stim)

    @pytest.mark.parametrize("radius", [25, 50, 75])
    def test_hexagon_centers(self, radius):
        # Check that all hexagon centers are at least 2*sqrt(3)/2*r distance
        # and that there is at least one hexagon at precisely that distance
        stimuli = self.get_experiment(center=[0, 0], radius=radius).stimuli
        centers = np.array(
            [
                stim.direct_stimulation_parameters["signal_function_parameters"][
                    "coords"
                ]
                for stim in stimuli
            ]
        ).squeeze()
        for i in range(centers.shape[0]):
            d = np.sqrt(((centers[i] - centers) ** 2).sum(axis=1))
            d[i] = np.infty
            a = np.isclose(d, radius * np.sqrt(3))
            assert np.any(a)
            d[a] = np.infty
            assert np.all(d >= radius * np.sqrt(3))


class TestOptogeneticArrayImageStimulus(TestCorticalStimulationWithOptogeneticArray):
    def get_experiment(self, im_path):
        return OptogeneticArrayImageStimulus(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "num_trials": 1,
                    "stimulator_array_list": self.stim_array_list,
                    "intensities": [1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "images_path": im_path,
                }
            ),
        )

    def test_or_map_activation(self):
        MapDependentModularConnectorFunction(
            self.sheet,
            self.sheet,
            ParameterSet(
                {
                    "map_location": self.test_dir + "/sheets/or_map",
                    "map_stretch": 1,
                    "sigma": 0,
                    "periodic": True,
                }
            ),
        )

        f = open(self.test_dir + "/sheets/or_map", "rb")
        or_map = pickle.load(f, encoding="latin1")
        f.close()
        np.save(self.test_dir + "/sheets/or_map.npy", circular_dist(0, or_map, 1))

        anns = self.model.neuron_annotations()["exc_sheet"]
        ors = [circular_dist(0, ann["LGNAfferentOrientation"], np.pi) for ann in anns]

        exp = self.get_experiment(im_path=self.test_dir + "/sheets/or_map.npy")
        ds = exp.direct_stimulation[0]["exc_sheet"][0]
        self.sheet.prepare_artificial_stimulation(
            exp.stimuli[0], exp.stimuli[0].duration, self.model.simulator_time, [ds]
        )
        msp = exp.direct_stimulation[0]["exc_sheet"][0].mixed_signals_photo[:, 0]
        assert len(msp) == len(ors)
        corr, _ = scipy.stats.pearsonr(msp, ors)
        assert corr > 0.825


class TestOptogeneticArrayStimulusOrientationTuningProtocol(
    TestCorticalStimulationWithOptogeneticArray
):
    def get_experiment(self, n_orientations):
        return OptogeneticArrayStimulusOrientationTuningProtocol(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "num_trials": 1,
                    "stimulator_array_list": self.stim_array_list,
                    "num_orientations": n_orientations,
                    "sharpness": 1,
                    "intensities": [1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                }
            ),
        )

    @pytest.mark.parametrize("n_orientations", range(1, 7))
    def test_or_map_activation(self, n_orientations):
        MapDependentModularConnectorFunction(
            self.sheet,
            self.sheet,
            ParameterSet(
                {
                    "map_location": "tests/sheets/or_map",
                    "map_stretch": 1,
                    "sigma": 0,
                    "periodic": True,
                }
            ),
        )

        exp = self.get_experiment(n_orientations=n_orientations)
        ds = exp.direct_stimulation[0]["exc_sheet"][0]

        orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
        for i in range(len(orientations)):
            anns = self.model.neuron_annotations()["exc_sheet"]
            dist = [
                circular_dist(orientations[i], a["LGNAfferentOrientation"], np.pi)
                for a in anns
            ]
            inv_dist = 1 - np.array(dist) / np.pi

            self.sheet.prepare_artificial_stimulation(
                exp.stimuli[i], exp.stimuli[i].duration, self.model.simulator_time, [ds]
            )
            msp = exp.direct_stimulation[0]["exc_sheet"][0].mixed_signals_photo[:, 0]

            assert len(msp) == len(inv_dist)
            corr, _ = scipy.stats.pearsonr(msp, inv_dist)
            assert corr > 0.89


class TestOptogeneticArrayStimulusContrastBasedOrientationTuningProtocol:
    # TODO: Enforce small activation difference between this and fullfield gratings
    pass


@pytest.mark.usefixtures("test_env")
class TestOptogeneticArrayStimulusCircleWithFullfieldSquareGrating(
    TestCorticalStimulationWithOptogeneticArray
):
    # TODO: Modify the test model sheets so that they have an input layer
    # and thus do not crash on this experiment
    # Then test the circle activation for 0 contrast
    pass
    """
    def get_experiment(self, center, inverted):
        return OptogeneticArrayStimulusCircleWithFullfieldSquareGrating(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "stimulator_array_list": self.stim_array_list,
                    "num_trials": 1,
                    "x_center": center[0],
                    "y_center": center[1],
                    "radii": [25, 50, 100, 150],
                    "intensities": [0.5, 1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "inverted": inverted,
                    "orientations": [0.0],
                    "spatial_frequency": 1.0,
                    "temporal_frequency": 1.0,
                    "contrasts": [0.0],  # Makes test equivalent to just circles
                    "shuffle_stimuli": False,
                }
            ),
        )

    @pytest.mark.parametrize("center", [[0, 0], [0, 1], [1, 0], [1, 1]])
    @pytest.mark.parametrize("inverted", [False, True])
    def test_stimulated_neurons_in_radius(self, center, inverted):
        exp = self.get_experiment(center=center, inverted=inverted)

        for stim in exp.stimuli:
            assert self.stimulated_neuron_in_radius(exp, stim, inverted)
    """
