"""Phase 1 Stage 2 tests for eccentricity-dependent LGN positions."""

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

from mozaik.models.vision.spatiotemporalfilter import (
    EccentricityDependentSpatioTemporalFilterRetinaLGN,
    _sample_lgn_positions,
)
from mozaik.models.vision.topography import RadiallySymmetricLGNTopography
from mozaik.sheets.vision import ExplicitPositions
from tests.models.vision.test_legacy_lgn_regression import LEGACY_MODEL_PARAMETERS


RF_TYPES = ("X_ON", "X_OFF")
PROBE_ARGUMENT = "--eccentricity-stage-2-probe"
SIGNATURE_PREFIX = "ECCENTRICITY_STAGE_2_SIGNATURE="


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


def _sha256_positions(positions):
    canonical = np.array(positions, dtype=np.dtype("<f8"), order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _collect_stage_2_signature():
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
        signature = {}
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
            hashes[rf_type] = position_hash
            signature[rf_type] = {
                "count": int(sheet.pop.size),
                "positions_sha256": position_hash,
            }
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
                    f"Stage 2 LGN probe failed with {process_count} rank(s)",
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
    assert signature["X_ON"]["count"] == 16
    assert signature["X_OFF"]["count"] == 16
    assert (
        signature["X_ON"]["positions_sha256"]
        != signature["X_OFF"]["positions_sha256"]
    )


@pytest.mark.mpi
def test_eccentricity_lgn_positions_match_between_one_and_two_ranks():
    assert _run_probe(process_count=2) == _run_probe(process_count=1)


if __name__ == "__main__" and PROBE_ARGUMENT in sys.argv:
    probe_signature = _collect_stage_2_signature()
    if probe_signature is not None:
        print(SIGNATURE_PREFIX + json.dumps(probe_signature, sort_keys=True))
