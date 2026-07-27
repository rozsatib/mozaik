"""Phase 1 Stage 2/3/4 tests for eccentricity-dependent LGN input."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from parameters import ParameterSet
import pytest
from scipy.integrate import quad
from scipy.stats import chisquare, kstest

from mozaik.models.vision import cai97
from mozaik.models.vision.spatiotemporalfilter import (
    CellWithReceptiveField,
    EccentricityDependentCellWithReceptiveField,
    EccentricityDependentSpatioTemporalFilterRetinaLGN,
    KernelResponse,
    SpatioTemporalReceptiveField,
    SpatioTemporalFilterRetinaLGN,
    _round_down_to_two_significant_digits,
    _sample_lgn_positions,
)
from mozaik.models.vision.topography import RadiallySymmetricLGNTopography
from mozaik.space import VisualSpace
from mozaik.sheets.vision import ExplicitPositions
from devtools.dummy_model import DummyModel
from mozaik.experiments.vision import VisualExperiment
from tests.models.vision.test_legacy_lgn_regression import LEGACY_MODEL_PARAMETERS


RF_TYPES = ("X_ON", "X_OFF")
PROBE_ARGUMENT = "--eccentricity-stage-4-probe"
SIGNATURE_PREFIX = "ECCENTRICITY_STAGE_4_SIGNATURE="


def visual_field(size_x=16.0, size_y=20.0):
    return SimpleNamespace(
        location_x=0.0,
        location_y=0.0,
        size_x=size_x,
        size_y=size_y,
    )


def topography_parameters(cap_eccentricity=None):
    return ParameterSet(
        {
            "cap_eccentricity": cap_eccentricity,
            "full_max_eccentricity": 90.0,
            "beta": 1.59,
        }
    )


def eccentricity_parameters(**overrides):
    parameters = copy.deepcopy(LEGACY_MODEL_PARAMETERS["sheets"]["retina_lgn"]["params"])
    parameters["number_per_polarity"] = 32
    parameters["minimum_samples_per_center_sigma"] = 4.0
    parameters["topography"] = {
        "cap_eccentricity": None,
        "full_max_eccentricity": 90.0,
        "beta": 1.59,
    }
    del parameters["density"]
    del parameters["size"]
    del parameters["receptive_field"]["spatial_resolution"]
    parameters.update(overrides)
    return ParameterSet(parameters)


def provider(cap_eccentricity=None):
    return RadiallySymmetricLGNTopography(
        visual_field(), topography_parameters(cap_eccentricity)
    )


class TestExplicitPositions:
    def test_generate_positions_requires_exact_count_and_returns_a_copy(self):
        positions = np.array(
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=float
        )
        structure = ExplicitPositions(positions)
        positions[0, 0] = 99.0

        first = structure.generate_positions(2)
        second = structure.generate_positions(2)
        assert first[0, 0] == 1.0
        first[0, 0] = -1.0
        assert second[0, 0] == 1.0

        with pytest.raises(ValueError, match="PyNN requested 1"):
            structure.generate_positions(1)

    @pytest.mark.parametrize(
        "positions,error",
        [
            (np.zeros((2, 3)), r"shape \(3, N\)"),
            (np.zeros((3, 2, 1)), r"shape \(3, N\)"),
            (np.array([[0.0], [0.0], [1.0]]), "third row"),
            (np.array([[np.nan], [0.0], [0.0]]), "finite"),
        ],
    )
    def test_invalid_position_arrays_are_rejected(self, positions, error):
        with pytest.raises(ValueError, match=error):
            ExplicitPositions(positions)


class TestLGNPositionSampling:
    def test_count_domain_seed_reproducibility_and_independence(self):
        topography = provider()
        first = _sample_lgn_positions(
            topography, 500, np.random.RandomState(seed=1234)
        )
        repeated = _sample_lgn_positions(
            topography, 500, np.random.RandomState(seed=1234)
        )
        different = _sample_lgn_positions(
            topography, 500, np.random.RandomState(seed=1235)
        )

        assert first.shape == (2, 500)
        assert np.array_equal(first, repeated)
        assert not np.array_equal(first, different)
        assert np.all(np.hypot(first[0], first[1]) < topography.max_eccentricity_deg)

    @pytest.mark.parametrize("cap_eccentricity", [None, 2.0])
    def test_radial_and_two_dimensional_distribution(self, cap_eccentricity):
        topography = provider(cap_eccentricity)
        sample_count = 50_000
        positions = _sample_lgn_positions(
            topography, sample_count, np.random.RandomState(seed=932_104)
        )
        radii = np.hypot(positions[0], positions[1])
        angles = np.mod(np.arctan2(positions[1], positions[0]) + np.pi, 2 * np.pi)
        maximum_eccentricity = topography.max_eccentricity_deg

        def radial_mass(radius):
            return radius * topography.relative_density_at_eccentricity(radius)

        normalization = quad(
            radial_mass,
            0.0,
            maximum_eccentricity,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
        cdf_grid = np.linspace(0.0, maximum_eccentricity, 4097)
        cdf_values = np.zeros_like(cdf_grid)
        for index in range(1, len(cdf_grid)):
            cdf_values[index] = cdf_values[index - 1] + quad(
                radial_mass,
                cdf_grid[index - 1],
                cdf_grid[index],
                epsabs=1e-12,
                epsrel=1e-12,
            )[0]
        cdf_values /= normalization

        radial_result = kstest(
            radii,
            lambda values: np.interp(values, cdf_grid, cdf_values),
        )
        angular_uniform = angles / (2.0 * np.pi)
        angular_result = kstest(angular_uniform, "uniform")
        assert radial_result.pvalue >= 1e-3
        assert angular_result.pvalue >= 1e-3

        quadrant_counts = np.histogram(angular_uniform, bins=np.linspace(0, 1, 5))[0]
        quadrant_fractions = quadrant_counts / sample_count
        assert np.all(np.abs(quadrant_fractions - 0.25) <= 0.015)

        radial_edges = np.linspace(0.0, maximum_eccentricity, 9)
        angular_edges = np.linspace(0.0, 2.0 * np.pi, 13)
        observed = np.histogram2d(
            radii, angles, bins=(radial_edges, angular_edges)
        )[0]
        radial_probabilities = np.array(
            [
                quad(
                    radial_mass,
                    radial_edges[index],
                    radial_edges[index + 1],
                    epsabs=1e-12,
                    epsrel=1e-12,
                )[0]
                / normalization
                for index in range(8)
            ]
        )
        expected = (
            sample_count * radial_probabilities[:, np.newaxis] / 12.0
        )
        expected = np.broadcast_to(expected, observed.shape)
        assert np.all(expected >= 5.0)
        assert chisquare(observed.ravel(), expected.ravel()).pvalue >= 1e-3


class TestEccentricityComponentConfiguration:
    def test_schema_replaces_all_legacy_only_keys(self):
        component = object.__new__(
            EccentricityDependentSpatioTemporalFilterRetinaLGN
        )
        component.check_parameters(eccentricity_parameters())

        with_spatial_resolution = eccentricity_parameters()
        with_spatial_resolution.receptive_field.spatial_resolution = 0.1
        with pytest.raises(KeyError, match="spatial_resolution"):
            component.check_parameters(with_spatial_resolution)

        with_density = eccentricity_parameters(density=10)
        with pytest.raises(KeyError, match="density"):
            component.check_parameters(with_density)

        with_size = eccentricity_parameters(size=(8.0, 8.0))
        with pytest.raises(KeyError, match="size"):
            component.check_parameters(with_size)

    @pytest.mark.parametrize("number", [0, -1, 3.0, True])
    def test_number_per_polarity_requires_a_positive_integer(self, number):
        component = object.__new__(
            EccentricityDependentSpatioTemporalFilterRetinaLGN
        )
        component.parameters = eccentricity_parameters(number_per_polarity=number)
        with pytest.raises(ValueError, match="integer greater than zero"):
            component._validate_stage_2_parameters()

    @pytest.mark.parametrize(
        "path,value",
        [
            ("minimum_samples_per_center_sigma", 0.0),
            ("minimum_samples_per_center_sigma", np.inf),
            ("minimum_samples_per_center_sigma", np.nan),
            ("receptive_field.width", 0.0),
            ("receptive_field.height", -1.0),
            ("receptive_field.temporal_resolution", np.inf),
            ("receptive_field.duration", np.nan),
        ],
    )
    def test_positive_finite_dimensions_are_required(self, path, value):
        parameters = eccentricity_parameters()
        target = parameters
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

        component = object.__new__(
            EccentricityDependentSpatioTemporalFilterRetinaLGN
        )
        component.parameters = parameters
        with pytest.raises(ValueError, match="positive and finite"):
            component._validate_stage_2_parameters()

    def test_original_2024_mode_is_rejected_before_position_seeding(self, monkeypatch):
        parameters = eccentricity_parameters(original_2024_lgn_mode=True)
        seed_calls = []
        monkeypatch.setattr(
            "mozaik.models.vision.spatiotemporalfilter.mozaik.get_seeds",
            lambda size=None: seed_calls.append(size),
        )
        with pytest.raises(ValueError, match="incompatible"):
            EccentricityDependentSpatioTemporalFilterRetinaLGN(
                SimpleNamespace(), parameters
            )
        assert seed_calls == []

    def test_visual_field_must_precede_the_input_component(self):
        with pytest.raises(ValueError, match="model.visual_field must exist"):
            EccentricityDependentSpatioTemporalFilterRetinaLGN(
                SimpleNamespace(), eccentricity_parameters()
            )

    def test_position_seeds_are_requested_once_before_noise_seeds(
        self, monkeypatch
    ):
        seed_calls = []
        sampler_draws = []

        def fake_get_seeds(size=None):
            seed_calls.append(size)
            if size == 2:
                return np.array([101, 202])
            return np.arange(size[0], dtype=np.uint32)

        def fake_sampler(topography, number, rng):
            sampler_draws.append(rng.randint(2**31))
            return np.zeros((2, number))

        class FakeCell:
            def inject(self, source):
                pass

        class FakeSheet:
            def __init__(self, model, parameters, positions_deg, topography):
                self.topography = topography
                self.pop = SimpleNamespace(
                    size=positions_deg.shape[1],
                    all_cells=[FakeCell() for _ in range(positions_deg.shape[1])],
                    _mask_local=np.ones(positions_deg.shape[1], dtype=bool),
                )

        class FakeSim:
            @staticmethod
            def StepCurrentSource(**parameters):
                return SimpleNamespace(**parameters)

            @staticmethod
            def NoisyCurrentSource(**parameters):
                return SimpleNamespace(**parameters)

        monkeypatch.setattr(
            "mozaik.models.vision.spatiotemporalfilter.mozaik.get_seeds",
            fake_get_seeds,
        )
        monkeypatch.setattr(
            "mozaik.models.vision.spatiotemporalfilter._sample_lgn_positions",
            fake_sampler,
        )
        monkeypatch.setattr(
            "mozaik.models.vision.spatiotemporalfilter."
            "RetinalInhomogeneousDiskSheet",
            FakeSheet,
        )
        monkeypatch.setattr(
            EccentricityDependentSpatioTemporalFilterRetinaLGN,
            "_initialize_receptive_fields",
            lambda self: None,
        )
        model = SimpleNamespace(sim=FakeSim(), visual_field=visual_field())
        retina = EccentricityDependentSpatioTemporalFilterRetinaLGN(
            model, eccentricity_parameters(number_per_polarity=3)
        )

        assert seed_calls == [2, (3,), (3,)]
        assert sampler_draws == [
            np.random.RandomState(101).randint(2**31),
            np.random.RandomState(202).randint(2**31),
        ]
        assert retina.sheets["X_ON"].topography is retina.topography
        assert retina.sheets["X_OFF"].topography is retina.topography


def _rf_only_component(
    positions_by_type,
    cap_eccentricity=None,
    minimum_samples=4.0,
    width=6.0,
    height=4.0,
):
    component = object.__new__(
        EccentricityDependentSpatioTemporalFilterRetinaLGN
    )
    parameters = eccentricity_parameters(
        number_per_polarity=positions_by_type["X_ON"].shape[1],
        minimum_samples_per_center_sigma=minimum_samples,
    )
    parameters.topography.cap_eccentricity = cap_eccentricity
    parameters.receptive_field.width = width
    parameters.receptive_field.height = height
    component.parameters = parameters
    component.topography = provider(cap_eccentricity)
    component.rf_types = RF_TYPES
    component.sheets = {}
    for rf_type in RF_TYPES:
        positions = np.asarray(positions_by_type[rf_type], dtype=float)
        component.sheets[rf_type] = SimpleNamespace(
            canonical_positions_deg=positions,
            pop=SimpleNamespace(
                size=positions.shape[1],
                _mask_local=np.ones(positions.shape[1], dtype=bool),
            ),
        )
    component.model = SimpleNamespace(
        input_space=VisualSpace(
            ParameterSet(
                {
                    "update_interval": 7.0,
                    "background_luminance": 45.0,
                }
            )
        )
    )
    component._validate_stage_2_parameters()
    component._validate_receptive_field_parameters()
    component._initialize_receptive_fields()
    return component


class TestReceptiveFieldConfiguration:
    @pytest.mark.parametrize(
        "func",
        [
            "cai97.F_2d",
            "not_a_receptive_field",
        ],
    )
    def test_only_cai97_spatiotemporal_rf_is_supported(self, func):
        component = object.__new__(
            EccentricityDependentSpatioTemporalFilterRetinaLGN
        )
        component.parameters = eccentricity_parameters()
        component.parameters.receptive_field.func = func

        with pytest.raises(ValueError, match="cai97.stRF_2d"):
            component._validate_receptive_field_parameters()

    def test_subtract_mean_is_rejected(self):
        component = object.__new__(
            EccentricityDependentSpatioTemporalFilterRetinaLGN
        )
        component.parameters = eccentricity_parameters()
        component.parameters.receptive_field.func_params.subtract_mean = True

        with pytest.raises(ValueError, match="subtract_mean=False"):
            component._validate_receptive_field_parameters()

    @pytest.mark.parametrize(
        "name,value,error",
        [
            ("Ac", np.nan, "Ac"),
            ("K1", np.inf, "K1"),
            ("sigma_c", 0.0, "sigma_c"),
            ("sigma_s", -1.0, "sigma_s"),
        ],
    )
    def test_cai97_parameters_must_be_finite_and_sigmas_positive(
        self, name, value, error
    ):
        component = object.__new__(
            EccentricityDependentSpatioTemporalFilterRetinaLGN
        )
        component.parameters = eccentricity_parameters()
        component.parameters.receptive_field.func_params[name] = value

        with pytest.raises(ValueError, match=error):
            component._validate_receptive_field_parameters()

    def test_missing_cai97_parameter_is_rejected(self):
        component = object.__new__(
            EccentricityDependentSpatioTemporalFilterRetinaLGN
        )
        component.parameters = eccentricity_parameters()
        del component.parameters.receptive_field.func_params["td"]

        with pytest.raises(ValueError, match="missing.*td"):
            component._validate_receptive_field_parameters()


class TestResolutionQuantization:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.151218, 0.15),
            (0.1, 0.1),
            (np.nextafter(0.1, 0.0), 0.099),
            (np.nextafter(0.1, np.inf), 0.1),
            (1.0, 1.0),
            (np.nextafter(1.0, 0.0), 0.99),
            (np.nextafter(1.0, np.inf), 1.0),
            (0.001, 0.001),
        ],
    )
    def test_rounds_down_at_decimal_boundaries(self, value, expected):
        assert _round_down_to_two_significant_digits(value) == expected

    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
    def test_rounding_rejects_invalid_values(self, value):
        with pytest.raises(ValueError, match="positive and finite"):
            _round_down_to_two_significant_digits(value)

    def test_smallest_actual_sigma_controls_resolution(self):
        positions = {
            "X_ON": np.array([[2.0, 4.0], [0.0, 0.0]]),
            "X_OFF": np.array([[3.0, 5.0], [0.0, 0.0]]),
        }
        component = _rf_only_component(positions, minimum_samples=4.0)
        expected_raw = component.topography.center_sigma_deg(2.0) / 4.0

        assert component.visual_space_resolution_deg == (
            _round_down_to_two_significant_digits(expected_raw)
        )
        for rf_type in RF_TYPES:
            assert np.all(
                component._rf_parameters[rf_type]["center_sigmas_deg"]
                / component.visual_space_resolution_deg
                >= 4.0 - 1e-12
            )

    def test_higher_sampling_requirement_reduces_pixel_size(self):
        positions = {
            "X_ON": np.array([[1.0], [0.0]]),
            "X_OFF": np.array([[1.0], [0.0]]),
        }
        coarse = _rf_only_component(positions, minimum_samples=4.0)
        fine = _rf_only_component(positions, minimum_samples=8.0)

        assert fine.visual_space_resolution_deg < coarse.visual_space_resolution_deg
        assert (
            fine.input_cells["X_ON"][0].receptive_field.kernel.shape[0]
            > coarse.input_cells["X_ON"][0].receptive_field.kernel.shape[0]
        )

    def test_higher_sampling_requirements_converge_complete_kernel_response(self):
        positions = {
            "X_ON": np.array([[1.0], [0.0]]),
            "X_OFF": np.array([[1.0], [0.0]]),
        }
        complete_kernel_responses = []
        for minimum_samples in (4.0, 8.0, 16.0):
            component = _rf_only_component(
                positions, minimum_samples=minimum_samples
            )
            complete_kernel_responses.append(
                component.input_cells["X_ON"][0].receptive_field.kernel.sum()
            )

        coarse_change = abs(
            complete_kernel_responses[1] - complete_kernel_responses[0]
        )
        fine_change = abs(
            complete_kernel_responses[2] - complete_kernel_responses[1]
        )
        assert fine_change < coarse_change


class TestPerCellReceptiveFields:
    def test_sigmas_support_and_complete_kernels_follow_position(self, caplog):
        caplog.set_level("INFO", logger="Mozaik")
        positions = {
            "X_ON": np.array([[0.0, 3.0], [0.0, 0.0]]),
            "X_OFF": np.array([[0.0, 3.0], [0.0, 0.0]]),
        }
        component = _rf_only_component(positions)
        reference = component.parameters.receptive_field.func_params

        for rf_type in RF_TYPES:
            cells = component.input_cells[rf_type]
            assert len(cells) == 2
            assert all(
                isinstance(
                    cell, EccentricityDependentCellWithReceptiveField
                )
                for cell in cells
            )
            for cell in cells:
                rf = cell.receptive_field
                eccentricity = np.hypot(cell.x, cell.y)
                expected_center = component.topography.center_sigma_deg(
                    eccentricity
                )
                expected_scale = expected_center / reference.sigma_c

                np.testing.assert_allclose(
                    rf.func_params.sigma_c,
                    expected_center,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    rf.func_params.sigma_s,
                    expected_scale * reference.sigma_s,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    rf.width,
                    expected_scale * component.parameters.receptive_field.width,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    rf.height,
                    expected_scale * component.parameters.receptive_field.height,
                    rtol=1e-12,
                    atol=1e-12,
                )
                assert rf.kernel.shape == (
                    int(np.ceil(rf.height / component.visual_space_resolution_deg)),
                    int(np.ceil(rf.width / component.visual_space_resolution_deg)),
                    int(
                        np.ceil(
                            component.parameters.receptive_field.duration
                            / component.parameters.receptive_field.temporal_resolution
                        )
                    ),
                )
                assert rf.func_params.Ac == reference.Ac
                assert rf.func_params.As == reference.As
                assert rf.func_params.K1 == reference.K1
                assert rf.func_params.td == reference.td
                assert rf.temporal_resolution == (
                    component.parameters.receptive_field.temporal_resolution
                )
                assert cell.visual_region.size_x == rf.width
                assert cell.visual_region.size_y == rf.height

        for on_cell, off_cell in zip(
            component.input_cells["X_ON"], component.input_cells["X_OFF"]
        ):
            assert on_cell.receptive_field is not off_cell.receptive_field
            assert np.array_equal(
                off_cell.receptive_field.kernel,
                -on_cell.receptive_field.kernel,
            )
        assert "rf" not in component.__dict__
        assert any(
            "estimated local kernel/contrast memory" in record.getMessage()
            for record in caplog.records
        )

    def test_explicit_cap_plateaus_center_surround_and_support(self):
        positions = {
            "X_ON": np.array([[0.0, 1.0, 3.0], [0.0, 0.0, 0.0]]),
            "X_OFF": np.array([[0.0, 1.0, 3.0], [0.0, 0.0, 0.0]]),
        }
        component = _rf_only_component(positions, cap_eccentricity=2.0)

        for rf_type in RF_TYPES:
            parameters = component._rf_parameters[rf_type]
            assert parameters["center_sigmas_deg"][0] == (
                parameters["center_sigmas_deg"][1]
            )
            assert parameters["surround_sigmas_deg"][0] == (
                parameters["surround_sigmas_deg"][1]
            )
            assert parameters["widths_deg"][0] == parameters["widths_deg"][1]
            assert parameters["heights_deg"][0] == parameters["heights_deg"][1]
            assert parameters["center_sigmas_deg"][2] > (
                parameters["center_sigmas_deg"][1]
            )

    def test_uncapped_fixation_and_periphery_have_different_scales(self):
        positions = {
            "X_ON": np.array([[0.0, 4.0], [0.0, 0.0]]),
            "X_OFF": np.array([[0.0, 4.0], [0.0, 0.0]]),
        }
        component = _rf_only_component(positions)
        parameters = component._rf_parameters["X_ON"]

        assert parameters["center_sigmas_deg"][0] < parameters["center_sigmas_deg"][1]
        assert parameters["surround_sigmas_deg"][0] < (
            parameters["surround_sigmas_deg"][1]
        )
        np.testing.assert_allclose(
            parameters["surround_sigmas_deg"]
            / parameters["center_sigmas_deg"],
            component.parameters.receptive_field.func_params.sigma_s
            / component.parameters.receptive_field.func_params.sigma_c,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            parameters["widths_deg"] / parameters["center_sigmas_deg"],
            np.full(
                2,
                component.parameters.receptive_field.width
                / component.parameters.receptive_field.func_params.sigma_c,
            ),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rf_near_disk_rim_keeps_rectangular_background_semantics(self):
        positions = {
            "X_ON": np.array([[7.99], [0.0]]),
            "X_OFF": np.array([[7.99], [0.0]]),
        }
        component = _rf_only_component(positions)
        cell = component.input_cells["X_ON"][0]
        cell.initialize(7.0)
        cell.view()

        assert cell.va.shape == cell.receptive_field.kernel.shape[:2]
        assert np.all(cell.va == component.model.input_space.background_luminance)
        assert (
            cell.visual_region.location_x + cell.visual_region.size_x / 2.0
            > component.topography.max_eccentricity_deg
        )


class _StaticVisualSpace:
    def __init__(self, renderer=None, background_luminance=45.0):
        self.update_interval = 7.0
        self.background_luminance = background_luminance
        self._renderer = renderer

    def view(self, visual_region, pixel_size):
        horizontal_samples = int(np.ceil(visual_region.size_x / pixel_size))
        vertical_samples = int(np.ceil(visual_region.size_y / pixel_size))
        width = horizontal_samples * pixel_size
        height = vertical_samples * pixel_size
        x = (
            np.linspace(0.0, width - pixel_size, horizontal_samples)
            + pixel_size / 2.0
            - width / 2.0
            + visual_region.location_x
        )
        y = (
            np.linspace(0.0, height - pixel_size, vertical_samples)
            + pixel_size / 2.0
            - height / 2.0
            + visual_region.location_y
        )
        x_grid, y_grid = np.meshgrid(x, y)
        if self._renderer is None:
            return np.full(
                (vertical_samples, horizontal_samples),
                self.background_luminance,
            )
        return self._renderer(x_grid, y_grid)


def _scaled_eccentricity_cell(
    scale,
    rf_type,
    pixel_size=0.00625,
    support=0.6,
    visual_space=None,
):
    parameters = eccentricity_parameters()
    function_parameters = copy.deepcopy(
        parameters.receptive_field.func_params
    )
    function_parameters.Ac = 1.0
    function_parameters.As = 0.2
    function_parameters.sigma_c = 0.1 * scale
    function_parameters.sigma_s = 0.2 * scale
    function_parameters.subtract_mean = False
    rf_function = cai97.stRF_2d
    if rf_type == "X_OFF":
        rf_function = lambda x, y, t, p: -cai97.stRF_2d(x, y, t, p)
    receptive_field = SpatioTemporalReceptiveField(
        rf_function,
        function_parameters,
        support * scale,
        support * scale,
        70.0,
    )
    receptive_field.quantize(pixel_size, pixel_size, 7.0)
    return EccentricityDependentCellWithReceptiveField(
        0.0,
        0.0,
        receptive_field,
        parameters.gain_control,
        visual_space or _StaticVisualSpace(),
        False,
    )


def _scaled_stimulus_metrics(cell, scale, stimulus_name):
    rf = cell.receptive_field
    vertical_samples, horizontal_samples = rf.kernel.shape[:2]
    pixel_size = rf.spatial_resolution
    width = horizontal_samples * pixel_size
    height = vertical_samples * pixel_size
    x = (
        np.linspace(0.0, width - pixel_size, horizontal_samples)
        + pixel_size / 2.0
        - width / 2.0
    )
    y = (
        np.linspace(0.0, height - pixel_size, vertical_samples)
        + pixel_size / 2.0
        - height / 2.0
    )
    x_grid, y_grid = np.meshgrid(x, y)
    background = cell.background_luminance
    if stimulus_name == "uniform":
        image = np.full_like(x_grid, 1.5 * background)
    elif stimulus_name == "gaussian":
        image = background * (
            1.0
            + np.exp(
                -(x_grid**2 + y_grid**2)
                / (2.0 * (0.12 * scale) ** 2)
            )
        )
    elif stimulus_name == "grating":
        image = background * (
            1.0 + np.cos(2.0 * np.pi * x_grid / (0.4 * scale))
        )
    else:
        raise ValueError(stimulus_name)

    direct = np.sum(rf.kernel * image[:, :, np.newaxis], axis=(0, 1))
    contrast = np.dot(
        rf.kernel_contrast_component,
        image.reshape(-1) / background,
    )
    luminance = rf.kernel_luminance_component * np.mean(image)
    combined = contrast + luminance
    gain = cell.gain_control.non_linear_gain
    current = cell.gain_function(
        contrast, gain.contrast_gain, gain.contrast_scaler
    ) + cell.gain_function(
        luminance, gain.luminance_gain, gain.luminance_scaler
    )
    return {
        "direct": float(np.max(np.abs(direct))),
        "contrast": float(np.max(np.abs(contrast))),
        "luminance": float(np.max(np.abs(luminance))),
        "combined": float(np.max(np.abs(combined))),
        "current": float(np.max(np.abs(current))),
    }


class TestEccentricityLuminanceCorrection:
    def test_kernel_components_and_all_luminance_states_use_spatial_sum(self):
        positions = {
            "X_ON": np.array([[0.0], [0.0]]),
            "X_OFF": np.array([[0.0], [0.0]]),
        }
        component = _rf_only_component(positions)

        for rf_type in RF_TYPES:
            cell = component.input_cells[rf_type][0]
            rf = cell.receptive_field
            spatial_sum = rf.kernel.sum(axis=(0, 1))
            np.testing.assert_allclose(
                rf.kernel_luminance_component,
                spatial_sum,
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                rf.kernel_contrast_component.sum(axis=1),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                cell.null_response.luminance,
                spatial_sum.sum() * cell.background_luminance,
                rtol=1e-12,
                atol=1e-12,
            )
            kernel_cumsum = np.cumsum(spatial_sum)
            np.testing.assert_allclose(
                cell.starting_luminance_kernel_state,
                cell.background_luminance
                * (kernel_cumsum[-1] - kernel_cumsum[:-1]),
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                cell.luminance_step_response,
                np.concatenate(
                    [[0.0], cell.background_luminance * kernel_cumsum]
                ),
                rtol=1e-12,
                atol=1e-12,
            )

            cell.initialize(14.0)
            before_view = cell.kernel_response.luminance.copy()
            cell.view()
            np.testing.assert_allclose(
                cell.kernel_response.luminance[: rf.kernel_duration]
                - before_view[: rf.kernel_duration],
                spatial_sum * cell.background_luminance,
                rtol=1e-12,
                atol=1e-12,
            )
            cell.response_current(cell.kernel_response)
            carried_state = copy.deepcopy(cell.filter_state)
            cell.initialize(14.0)
            assert np.array_equal(
                cell.kernel_response.contrast[: rf.kernel_duration],
                carried_state.contrast,
            )
            assert np.array_equal(
                cell.kernel_response.luminance[: rf.kernel_duration],
                carried_state.luminance,
            )

    def test_blank_input_uses_corrected_luminance_step(self):
        positions = {
            "X_ON": np.array([[0.0], [0.0]]),
            "X_OFF": np.array([[0.0], [0.0]]),
        }
        component = _rf_only_component(positions)
        captured = {}
        for rf_type in RF_TYPES:
            cell = component.input_cells[rf_type][0]
            original_response_current = cell.response_current

            def capture_response(
                response,
                rf_type=rf_type,
                original=original_response_current,
            ):
                captured[rf_type] = KernelResponse(
                    contrast=response.contrast.copy(),
                    luminance=response.luminance.copy(),
                )
                return original(response)

            cell.response_current = capture_response

        component.calculate_null_input(14.0)
        for rf_type in RF_TYPES:
            cell = component.input_cells[rf_type][0]
            kernel_duration = cell.receptive_field.kernel_duration
            num_frames = 2
            response_length = num_frames + kernel_duration
            initial = np.pad(
                cell.starting_luminance_kernel_state,
                (0, response_length - len(cell.starting_luminance_kernel_state)),
            )
            step_on = np.concatenate(
                [
                    cell.luminance_step_response[1:],
                    np.full(num_frames, cell.luminance_steady_state),
                ]
            )
            step_off = np.concatenate(
                [
                    np.zeros(num_frames),
                    cell.luminance_step_response[1:kernel_duration],
                ]
            )
            added = step_on[: response_length - 1] - step_off[: response_length - 1]
            expected = initial + np.pad(added, (0, 1))
            np.testing.assert_allclose(
                captured[rf_type].luminance,
                expected,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_legacy_cell_keeps_bitwise_mean_luminance_kernel(self):
        parameters = eccentricity_parameters()
        function_parameters = copy.deepcopy(
            parameters.receptive_field.func_params
        )
        receptive_field = SpatioTemporalReceptiveField(
            cai97.stRF_2d,
            function_parameters,
            6.0,
            6.0,
            70.0,
        )
        receptive_field.quantize(0.1, 0.1, 7.0)
        kernel = receptive_field.kernel.copy()
        expected_mean = kernel.mean(axis=(0, 1))
        expected_contrast = kernel - expected_mean
        CellWithReceptiveField(
            0.0,
            0.0,
            receptive_field,
            parameters.gain_control,
            _StaticVisualSpace(),
            False,
        )

        assert np.array_equal(
            receptive_field.kernel_luminance_component,
            expected_mean,
        )
        assert np.array_equal(
            receptive_field.kernel_contrast_component,
            expected_contrast.reshape(-1, expected_contrast.shape[2]).T,
        )

    def test_geometric_rf_rescaling_is_response_invariant(
        self, record_property
    ):
        scales = (0.5, 1.0, 2.0, 4.0)
        stimulus_names = ("uniform", "gaussian", "grating")
        pixel_size = 0.00625
        boundary_free_window_size = 0.6 * max(scales) + 2.0 * pixel_size
        metrics = {}
        for rf_type in RF_TYPES:
            for scale in scales:
                cell = _scaled_eccentricity_cell(
                    scale,
                    rf_type,
                    pixel_size=pixel_size,
                )
                assert (
                    cell.receptive_field.func_params.sigma_c / pixel_size
                    >= 8.0
                )
                assert (
                    cell.visual_region.size_x + 2.0 * pixel_size
                    <= boundary_free_window_size
                )
                assert (
                    cell.visual_region.size_y + 2.0 * pixel_size
                    <= boundary_free_window_size
                )
                for stimulus_name in stimulus_names:
                    metrics[(rf_type, scale, stimulus_name)] = (
                        _scaled_stimulus_metrics(
                            cell, scale, stimulus_name
                        )
                    )

        for rf_type in RF_TYPES:
            for stimulus_name in stimulus_names:
                reference = metrics[(rf_type, 1.0, stimulus_name)]
                for scale in scales:
                    actual = metrics[(rf_type, scale, stimulus_name)]
                    for metric_name, reference_value in reference.items():
                        relative_error = abs(
                            actual[metric_name] - reference_value
                        ) / max(abs(reference_value), 1e-12)
                        record_property(
                            (
                                f"{rf_type}_{stimulus_name}_{metric_name}_"
                                f"scale_{scale:g}_relative_error"
                            ),
                            relative_error,
                        )
                        assert relative_error <= 0.01

    def test_corrected_current_change_is_reported_without_gain_compensation(
        self, record_property
    ):
        corrected_cell = _scaled_eccentricity_cell(1.0, "X_ON")
        corrected_rf = corrected_cell.receptive_field
        legacy_rf = SpatioTemporalReceptiveField(
            cai97.stRF_2d,
            copy.deepcopy(corrected_rf.func_params),
            corrected_rf.width,
            corrected_rf.height,
            corrected_rf.duration,
        )
        legacy_rf.quantize(
            corrected_rf.spatial_resolution,
            corrected_rf.spatial_resolution,
            corrected_rf.temporal_resolution,
        )
        legacy_cell = CellWithReceptiveField(
            0.0,
            0.0,
            legacy_rf,
            corrected_cell.gain_control,
            _StaticVisualSpace(),
            False,
        )

        corrected_current = _scaled_stimulus_metrics(
            corrected_cell, 1.0, "uniform"
        )["current"]
        legacy_current = _scaled_stimulus_metrics(
            legacy_cell, 1.0, "uniform"
        )["current"]
        relative_change = abs(corrected_current - legacy_current) / max(
            abs(legacy_current), 1e-12
        )
        record_property(
            "corrected_vs_legacy_uniform_current_relative_change",
            relative_change,
        )
        assert corrected_cell.gain_control is legacy_cell.gain_control
        assert relative_change > 0.0

    def test_support_truncation_is_reported_separately(self, record_property):
        responses = {}
        for support in (0.5, 0.6, 0.8):
            cell = _scaled_eccentricity_cell(
                1.0,
                "X_ON",
                support=support,
            )
            responses[support] = _scaled_stimulus_metrics(
                cell, 1.0, "uniform"
            )["direct"]
        relative_deviation = abs(responses[0.5] - responses[0.8]) / abs(
            responses[0.8]
        )
        record_property(
            "support_truncation_relative_deviation",
            relative_deviation,
        )
        assert relative_deviation > 0.0

    def test_stimulus_boundary_effect_is_reported_separately(
        self, record_property
    ):
        background = 45.0

        def renderer(x_grid, y_grid):
            inside_stimulus = (np.abs(x_grid) <= 0.35) & (
                np.abs(y_grid) <= 0.35
            )
            pattern = background * (
                1.0
                + np.exp(
                    -(x_grid**2 + y_grid**2)
                    / (2.0 * 0.12**2)
                )
            )
            return np.where(inside_stimulus, pattern, background)

        visual_space = _StaticVisualSpace(renderer=renderer)
        center_cell = _scaled_eccentricity_cell(
            1.0, "X_ON", visual_space=visual_space
        )
        edge_cell = _scaled_eccentricity_cell(
            1.0, "X_ON", visual_space=visual_space
        )
        edge_cell.x = 0.35
        edge_cell.visual_region.location_x = 0.35

        responses = []
        for cell in (center_cell, edge_cell):
            cell.initialize(7.0)
            cell.view()
            combined = (
                cell.kernel_response.contrast
                + cell.kernel_response.luminance
            )
            responses.append(float(np.max(np.abs(combined))))
        relative_deviation = abs(responses[1] - responses[0]) / abs(
            responses[0]
        )
        record_property(
            "stimulus_boundary_relative_deviation",
            relative_deviation,
        )
        assert relative_deviation > 0.0


class TestCommonVisualResolution:
    def test_legacy_component_returns_configured_resolution(self):
        component = object.__new__(SpatioTemporalFilterRetinaLGN)
        component.parameters = ParameterSet(
            copy.deepcopy(LEGACY_MODEL_PARAMETERS["sheets"]["retina_lgn"]["params"])
        )
        assert component.visual_space_resolution_deg == 0.1

    def test_dummy_model_exposes_common_property(self):
        model = DummyModel(
            density=12.5,
            background_luminance=45.0,
            frame_duration=7.0,
            size_x=8.0,
            size_y=8.0,
        )
        assert model.input_layer.visual_space_resolution_deg == 0.08
        assert model.input_layer.parameters.receptive_field.spatial_resolution == 0.08

    def test_visual_experiment_reads_common_property(self):
        class EmptyVisualExperiment(VisualExperiment):
            def generate_stimuli(self):
                self.stimuli = []

        model = DummyModel(
            density=20.0,
            background_luminance=45.0,
            frame_duration=7.0,
            size_x=8.0,
            size_y=8.0,
        )
        del model.input_layer.parameters.receptive_field.spatial_resolution
        experiment = EmptyVisualExperiment(
            model, ParameterSet({"shuffle_stimuli": False})
        )
        assert experiment.density == 20.0


def _sha256_positions(positions):
    canonical = np.array(positions, dtype=np.dtype("<f8"), order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _collect_stage_4_signature():
    import mozaik
    from mozaik.models import Model
    from mozaik.space import VisualRegion
    from pyNN import nest

    model_parameters = copy.deepcopy(LEGACY_MODEL_PARAMETERS)
    model_parameters["sheets"]["retina_lgn"]["params"] = (
        eccentricity_parameters(number_per_polarity=16).as_dict()
    )
    parameters = ParameterSet(model_parameters)
    mozaik.setup_mpi(parameters.mpi_seed, parameters.pynn_seed)
    model = Model(nest, 1, parameters)
    model.visual_field = VisualRegion(
        location_x=0.0,
        location_y=0.0,
        size_x=12.0,
        size_y=16.0,
    )
    comm = mozaik.mpi_comm

    try:
        retina = EccentricityDependentSpatioTemporalFilterRetinaLGN(
            model, parameters.sheets.retina_lgn.params
        )
        resolution = float(retina.visual_space_resolution_deg)
        assert all(
            rank_resolution == resolution
            for rank_resolution in comm.allgather(resolution)
        )
        signature = {"visual_space_resolution_deg": resolution}
        hashes = {}
        for rf_type in RF_TYPES:
            sheet = retina.sheets[rf_type]
            assert sheet.topography is retina.topography
            assert sheet.pop.size == 16
            assert sheet.pop.positions.shape == (3, 16)
            assert np.array_equal(
                sheet.pop.positions[:2], sheet.canonical_positions_deg
            )
            assert not sheet.canonical_positions_deg.flags.writeable
            assert np.all(
                np.hypot(
                    sheet.canonical_positions_deg[0],
                    sheet.canonical_positions_deg[1],
                )
                < retina.topography.max_eccentricity_deg
            )
            position_hash = _sha256_positions(sheet.canonical_positions_deg)
            assert all(
                rank_hash == position_hash
                for rank_hash in comm.allgather(position_hash)
            )
            assert comm.allreduce(int(np.count_nonzero(sheet.pop._mask_local))) == 16
            local_rf_parameters = []
            local_index = 0
            for global_index, is_local in enumerate(sheet.pop._mask_local):
                if not is_local:
                    continue
                cell = retina.input_cells[rf_type][local_index]
                rf = cell.receptive_field
                local_rf_parameters.append(
                    (
                        global_index,
                        {
                            "center_sigma_deg": float(rf.func_params.sigma_c),
                            "surround_sigma_deg": float(rf.func_params.sigma_s),
                            "width_deg": float(rf.width),
                            "height_deg": float(rf.height),
                            "kernel_shape": list(rf.kernel.shape),
                            "kernel_sha256": _sha256_positions(rf.kernel),
                            "luminance_kernel_sha256": _sha256_positions(
                                rf.kernel_luminance_component
                            ),
                            "contrast_kernel_sha256": _sha256_positions(
                                rf.kernel_contrast_component
                            ),
                        },
                    )
                )
                local_index += 1
            gathered_rf_parameters = comm.gather(local_rf_parameters, root=0)
            hashes[rf_type] = position_hash
            signature[rf_type] = {
                "count": int(sheet.pop.size),
                "positions_sha256": position_hash,
            }
            if comm.rank == 0:
                global_rf_parameters = {}
                for rank_parameters in gathered_rf_parameters:
                    for global_index, rf_parameters in rank_parameters:
                        assert global_index not in global_rf_parameters
                        global_rf_parameters[global_index] = rf_parameters
                assert sorted(global_rf_parameters) == list(range(16))
                signature[rf_type]["rf_parameters"] = [
                    global_rf_parameters[index] for index in range(16)
                ]
        assert hashes["X_ON"] != hashes["X_OFF"]
        return signature if comm.rank == 0 else None
    finally:
        model.sim.end()


def _run_probe(process_count):
    command = [sys.executable, str(Path(__file__).resolve()), PROBE_ARGUMENT]
    if process_count == 2:
        command = ["mpirun", "--oversubscribe", "-np", "2", *command]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mozaik-matplotlib")
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "\n".join(
                [
                    f"Stage 4 LGN probe failed with {process_count} rank(s)",
                    f"Return code: {result.returncode}",
                    "===== stdout =====",
                    result.stdout or "<empty>",
                    "===== stderr =====",
                    result.stderr or "<empty>",
                ]
            )
        )
    lines = [
        line for line in result.stdout.splitlines() if line.startswith(SIGNATURE_PREFIX)
    ]
    assert len(lines) == 1, result.stdout
    return json.loads(lines[0][len(SIGNATURE_PREFIX) :])


def test_eccentricity_lgn_single_process_construction():
    signature = _run_probe(process_count=1)
    assert signature["visual_space_resolution_deg"] > 0.0
    assert signature["X_ON"]["count"] == 16
    assert signature["X_OFF"]["count"] == 16
    assert len(signature["X_ON"]["rf_parameters"]) == 16
    assert len(signature["X_OFF"]["rf_parameters"]) == 16
    assert (
        signature["X_ON"]["positions_sha256"]
        != signature["X_OFF"]["positions_sha256"]
    )


@pytest.mark.mpi
def test_eccentricity_lgn_positions_match_between_one_and_two_ranks():
    assert _run_probe(process_count=2) == _run_probe(process_count=1)


if __name__ == "__main__" and PROBE_ARGUMENT in sys.argv:
    probe_signature = _collect_stage_4_signature()
    if probe_signature is not None:
        print(SIGNATURE_PREFIX + json.dumps(probe_signature, sort_keys=True))
