import pickle
import pytest
import numpy as np
from pyNN import nest
import quantities as qt
from copy import deepcopy
from mozaik.models import Model
from parameters import ParameterSet
from mozaik.sheets.vision import VisualCorticalUniformSheet3D
from mozaik.experiments.optogenetic import *
from mozaik.tools.distribution_parametrization import (
    load_parameters,
    PyNNDistribution,
    MozaikExtendedParameterSet,
)
from mozaik.connectors.vision import MapDependentModularConnectorFunction
from mozaik.tools.circ_stat import circular_dist
import scipy.stats


class TestCorticalStimulationWithOptogeneticArray:

    model = None
    sheet = None

    @classmethod
    def setup_class(cls):
        model_params = load_parameters("tests/sheets/model_params")
        cls.sheet_params = load_parameters("tests/sheets/exc_sheet_params")
        cls.opt_array_params = load_parameters("tests/sheets/opt_array_params")
        cls.set_sheet_size(cls, 400)
        cls.model = Model(nest, 8, model_params)
        cls.sheet = VisualCorticalUniformSheet3D(
            cls.model, ParameterSet(cls.sheet_params)
        )
        cls.sheet.record()

    def set_sheet_size(self, size):
        self.sheet_params["sx"] = size
        self.sheet_params["sy"] = size
        self.sheet_params["recorders"]["1"]["params"]["size"] = size
        self.opt_array_params["size"] = size

    def get_coords(self, neuron_ids):
        ac = np.array(list(self.sheet.pop.all()))
        pos = self.sheet.pop.positions[0:2, :]
        return pos[:, [np.where(ac == i)[0][0] for i in neuron_ids]]

    def get_experiment(self, **params):
        pass

    def get_experiment_direct_stimulators(self, **params):
        dss = self.get_experiment(**params).direct_stimulation
        return [ds["exc_sheet"][0] for ds in dss]

    def stimulated_neuron_in_radius(self, ds, invert=False):
        ssp = ds.parameters.stimulating_signal_parameters
        center = np.array(ssp.coords).T
        coords = self.get_coords(ds.stimulated_cells) * 1000  # to µms
        d = np.sqrt(((coords - center) ** 2).sum(axis=0))
        if invert:
            return np.all(d > ssp.radius - ds.parameters.spacing / 2)
        else:
            return np.all(d < ssp.radius + ds.parameters.spacing / 2)


class TestSingleOptogeneticArrayStimulus:
    pass


class TestOptogeneticArrayStimulusCircles(TestCorticalStimulationWithOptogeneticArray):
    def get_experiment(self, center, inverted):
        return OptogeneticArrayStimulusCircles(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
                    "x_center": center[0],
                    "y_center": center[1],
                    "radii": [25, 50, 100, 150, 200],
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
    @pytest.mark.skip
    def test_stimulated_neurons_in_radius(self, center, inverted):
        dss = self.get_experiment_direct_stimulators(center=center, inverted=inverted)
        for ds in dss:
            assert self.stimulated_neuron_in_radius(ds, inverted)


class TestOptogeneticArrayStimulusHexagonalTiling(
    TestCorticalStimulationWithOptogeneticArray
):
    def get_experiment(self, center, radius):
        return OptogeneticArrayStimulusHexagonalTiling(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
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
    @pytest.mark.skip
    def test_stimulated_neurons_in_radius(self, center, radius):
        dss = self.get_experiment_direct_stimulators(center=center, radius=radius)
        for ds in dss:
            assert self.stimulated_neuron_in_radius(ds)

    @pytest.mark.parametrize("radius", [25, 50, 75])
    @pytest.mark.skip
    def test_hexagon_centers(self, radius):
        # Check that all hexagon centers are at least 2*sqrt(3)/2*r distance
        # and that there is at least one hexagon at precisely that distance
        dss = self.get_experiment_direct_stimulators(center=[0, 0], radius=radius)
        centers = np.array(
            [ds.parameters.stimulating_signal_parameters.coords for ds in dss]
        ).squeeze()
        for i in range(centers.shape[0]):
            d = np.sqrt(((centers[i] - centers) ** 2).sum(axis=1))
            d[i] = np.infty
            a = np.isclose(d, radius * np.sqrt(3))
            assert np.any(a)
            d[a] = np.infty
            assert np.all(d >= radius * np.sqrt(3))


class TestOptogeneticArrayImageStimulus(TestCorticalStimulationWithOptogeneticArray):
    """"""

    def get_experiment(self, im_path):
        return OptogeneticArrayImageStimulus(
            self.model,
            MozaikExtendedParameterSet(
                {
                    "sheet_list": ["exc_sheet"],
                    "num_trials": 1,
                    "stimulator_array_parameters": deepcopy(self.opt_array_params),
                    "intensities": [1.0],
                    "duration": 150,
                    "onset_time": 0,
                    "offset_time": 75,
                    "images_path": im_path,
                }
            ),
        )

    # Test if or_map stimulation checks out with or assignment
    @pytest.mark.skip
    def test_or_map_activation(self):
        MapDependentModularConnectorFunction(
            self.sheet,
            self.sheet,
            ParameterSet(
                {"map_location": "tests/sheets/or_map", "sigma": 0, "periodic": True}
            ),
        )
        dss = self.get_experiment_direct_stimulators(im_path="tests/sheets/or_map.npy")
        anns = self.model.neuron_annotations()["exc_sheet"]
        ids = self.model.neuron_ids()["exc_sheet"]
        ors = [circular_dist(0, ann["LGNAfferentOrientation"], np.pi) for ann in anns]
        assert len(dss) == 1
        msp = dss[0].mixed_signals_photo[:, 0]
        assert len(msp) == len(ors)
        corr, _ = scipy.stats.pearsonr(msp, ors)
        assert corr > 0.9


class TestOptogeneticArrayStimulusOrientationTuningProtocol:
    # TODO: test activation correlation with orientation map
    pass


class TestOptogeneticArrayStimulusContrastBasedOrientationTuningProtocol:
    # TODO: Enforce small activation difference between this and fullfield gratings
    pass
