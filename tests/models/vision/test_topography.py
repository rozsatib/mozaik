import inspect
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from parameters import ParameterSet

from mozaik.models.vision import cai97
from mozaik.models.vision import topography as topography_module
from mozaik.models.vision.topography import (
    RadiallySymmetricLGNTopography,
    capped_relative_lgn_cell_density,
    capped_relative_lgn_cell_density_at_eccentricity,
    relative_lgn_cell_density,
    relative_lgn_cell_density_at_eccentricity,
    rf_center_sigma,
)
from mozaik.space import VisualRegion

RTOL = 1e-12
ATOL = 1e-12


def topography_parameters(cap_eccentricity=None, **overrides):
    values = {
        "cap_eccentricity": cap_eccentricity,
        "full_max_eccentricity": 90.0,
        "beta": 1.59,
    }
    values.update(overrides)
    return ParameterSet(values)


def visual_field(size_x=10.0, size_y=8.0, location_x=0.0, location_y=0.0):
    return SimpleNamespace(
        location_x=location_x,
        location_y=location_y,
        size_x=size_x,
        size_y=size_y,
    )


class TestLGNDensity:
    def test_reference_values_and_scalar_return(self):
        eccentricities = np.array([0.0, 1.0, 5.0, 10.0, 25.0, 60.0, 89.0])
        expected = np.array(
            [
                256.27959300000003,
                176.31399900666253,
                59.29995286827676,
                32.22851854409236,
                12.465202829090261,
                1.5312702625878194,
                0.2694939500098275,
            ]
        )

        actual = relative_lgn_cell_density_at_eccentricity(eccentricities)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
        fixation_density = relative_lgn_cell_density_at_eccentricity(0.0)
        assert isinstance(fixation_density, float)
        np.testing.assert_allclose(
            fixation_density, 200.54919 + 55.730403, rtol=RTOL, atol=ATOL
        )
        assert np.all(actual > 0.0)
        assert np.all(np.isfinite(actual))

    def test_cartesian_density_is_radial_area_density_without_jacobian(self):
        radial_density = relative_lgn_cell_density_at_eccentricity(5.0)
        cartesian_density = relative_lgn_cell_density(3.0, 4.0)

        assert isinstance(cartesian_density, float)
        np.testing.assert_allclose(
            cartesian_density, radial_density, rtol=RTOL, atol=ATOL
        )
        assert cartesian_density != pytest.approx(5.0 * radial_density)
        np.testing.assert_allclose(
            relative_lgn_cell_density(-3.0, -4.0),
            radial_density,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_cartesian_density_broadcasts(self):
        x = np.array([[0.0], [3.0]])
        y = np.array([[0.0, 4.0, 8.0]])
        actual = relative_lgn_cell_density(x, y)

        assert actual.shape == (2, 3)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                np.testing.assert_allclose(
                    actual[i, j],
                    relative_lgn_cell_density(float(x[i, 0]), float(y[0, j])),
                    rtol=RTOL,
                    atol=ATOL,
                )

    def test_explicit_cap_plateaus_density(self):
        cap = 5.0
        eccentricities = np.array([0.0, 2.0, 5.0, 8.0])
        expected = relative_lgn_cell_density_at_eccentricity(
            np.array([5.0, 5.0, 5.0, 8.0])
        )

        np.testing.assert_allclose(
            capped_relative_lgn_cell_density_at_eccentricity(eccentricities, cap),
            expected,
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            capped_relative_lgn_cell_density(3.0, 4.0, cap),
            expected[2],
            rtol=RTOL,
            atol=ATOL,
        )

    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
    def test_invalid_eccentricity_raises(self, value):
        with pytest.raises(ValueError, match="eccentricity_deg"):
            relative_lgn_cell_density_at_eccentricity(value)

    @pytest.mark.parametrize("cap", [0.0, -1.0, np.nan, np.inf, None])
    def test_invalid_explicit_cap_raises(self, cap):
        with pytest.raises(ValueError, match="cap_eccentricity_deg"):
            capped_relative_lgn_cell_density_at_eccentricity(1.0, cap)

    def test_nonfinite_cartesian_input_and_incompatible_shapes_raise(self):
        with pytest.raises(ValueError, match="x_deg"):
            relative_lgn_cell_density(np.nan, 0.0)
        with pytest.raises(ValueError, match="broadcast-compatible"):
            relative_lgn_cell_density(np.zeros(2), np.zeros(3))


class TestRFCenterSigma:
    def test_reference_values_and_scalar_return(self):
        eccentricities = np.array([0.0, 1.0, 5.0, 10.0, 25.0, 60.0, 89.0])
        expected = np.array(
            [
                0.12344826830757859,
                0.12749260711614505,
                0.14503906859609314,
                0.1704060470642565,
                0.27636624777983226,
                0.8540533532149346,
                2.1751289172894936,
            ]
        )

        actual = rf_center_sigma(eccentricities)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
        assert isinstance(rf_center_sigma(0.0), float)
        assert np.all(actual > 0.0)

    def test_explicit_cap_plateaus_sigma(self):
        cap = 5.0
        actual = rf_center_sigma(np.array([0.0, 2.0, 5.0, 8.0]), cap)
        expected = rf_center_sigma(np.array([5.0, 5.0, 5.0, 8.0]))
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
    def test_invalid_eccentricity_raises(self, value):
        with pytest.raises(ValueError, match="eccentricity_deg"):
            rf_center_sigma(value)


class TestRadiallySymmetricLGNTopography:
    def test_domain_comes_from_smaller_visual_field_edge(self):
        provider = RadiallySymmetricLGNTopography(
            VisualRegion(
                location_x=0.0,
                location_y=0.0,
                size_x=12.0,
                size_y=8.0,
            ),
            topography_parameters(),
        )
        assert provider.max_eccentricity_deg == 4.0
        assert provider.user_cap_eccentricity_deg is None

    @pytest.mark.parametrize(
        "location_x,location_y",
        [(1e-12, -1e-12), (-1e-12, 1e-12)],
    )
    def test_center_tolerance_accepts_parsed_zero(self, location_x, location_y):
        provider = RadiallySymmetricLGNTopography(
            visual_field(location_x=location_x, location_y=location_y),
            topography_parameters(),
        )
        assert provider.max_eccentricity_deg == 4.0

    @pytest.mark.parametrize(
        "location_x,location_y",
        [(1.0001e-12, 0.0), (0.0, -1.0001e-12), (1.0, -2.0)],
    )
    def test_off_center_visual_field_is_rejected(self, location_x, location_y):
        with pytest.raises(ValueError, match="fixation-centred visual rectangle"):
            RadiallySymmetricLGNTopography(
                visual_field(location_x=location_x, location_y=location_y),
                topography_parameters(),
            )

    @pytest.mark.parametrize(
        "field_name,value",
        [
            ("location_x", np.nan),
            ("location_y", np.inf),
            ("size_x", np.nan),
            ("size_y", np.inf),
            ("size_x", 0.0),
            ("size_y", -1.0),
        ],
    )
    def test_invalid_visual_field_value_is_rejected(self, field_name, value):
        field = visual_field()
        setattr(field, field_name, value)
        with pytest.raises(ValueError, match=field_name):
            RadiallySymmetricLGNTopography(field, topography_parameters())

    def test_missing_visual_field_value_is_rejected(self):
        field = visual_field()
        del field.size_y
        with pytest.raises(ValueError, match="visual_field.size_y"):
            RadiallySymmetricLGNTopography(field, topography_parameters())

    @pytest.mark.parametrize("size", [180.0, 181.0])
    def test_maximum_eccentricity_at_or_above_90_is_rejected(self, size):
        with pytest.raises(ValueError, match="E_max >= 90"):
            RadiallySymmetricLGNTopography(
                visual_field(size_x=size, size_y=size), topography_parameters()
            )

    def test_rectangle_corners_beyond_90_are_accepted(self, caplog):
        caplog.set_level(logging.WARNING, logger="Mozaik")
        provider = RadiallySymmetricLGNTopography(
            visual_field(size_x=178.0, size_y=178.0), topography_parameters()
        )
        assert provider.max_eccentricity_deg == 89.0
        assert np.hypot(89.0, 89.0) > 90.0

    def test_none_cap_leaves_retinal_functions_uncapped(self):
        provider = RadiallySymmetricLGNTopography(
            visual_field(), topography_parameters()
        )
        eccentricities = np.array([0.0, 2.0, 4.0])
        np.testing.assert_allclose(
            provider.relative_density_at_eccentricity(eccentricities),
            relative_lgn_cell_density_at_eccentricity(eccentricities),
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            provider.center_sigma_deg(eccentricities),
            rf_center_sigma(eccentricities),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_explicit_cap_is_shared_by_density_and_rf_size(self):
        provider = RadiallySymmetricLGNTopography(
            visual_field(), topography_parameters(cap_eccentricity=2.0)
        )
        assert provider.user_cap_eccentricity_deg == 2.0
        assert provider.relative_density_at_eccentricity(0.0) == (
            provider.relative_density_at_eccentricity(2.0)
        )
        assert provider.relative_density_xy(0.0, 0.0) == (
            provider.relative_density_xy(2.0, 0.0)
        )
        assert provider.center_sigma_deg(0.0) == provider.center_sigma_deg(2.0)

    def test_cap_equal_to_maximum_eccentricity_is_accepted(self):
        provider = RadiallySymmetricLGNTopography(
            visual_field(), topography_parameters(cap_eccentricity=4.0)
        )
        assert provider.user_cap_eccentricity_deg == (provider.max_eccentricity_deg)
        assert provider.relative_density_at_eccentricity(0.0) == (
            provider.relative_density_at_eccentricity(4.0)
        )

    def test_empirical_cap_reference_value_is_the_explicit_lower_bound(self):
        empirical_cap = 1.821096469899997
        provider = RadiallySymmetricLGNTopography(
            visual_field(),
            topography_parameters(cap_eccentricity=empirical_cap),
        )
        assert provider.user_cap_eccentricity_deg == empirical_cap

        with pytest.raises(ValueError, match="empirical mapping cap"):
            RadiallySymmetricLGNTopography(
                visual_field(),
                topography_parameters(
                    cap_eccentricity=np.nextafter(empirical_cap, -np.inf)
                ),
            )

    def test_phase_2_mapping_api_is_not_exposed(self):
        provider = RadiallySymmetricLGNTopography(
            visual_field(), topography_parameters()
        )
        for phase_2_name in (
            "mapping_cap_eccentricity_deg",
            "mapping_axis_extent_mm",
            "visual_to_cortical_mm",
            "cortical_to_visual_deg",
        ):
            assert not hasattr(provider, phase_2_name)

    def test_explicit_cap_below_empirical_minimum_is_rejected(self):
        with pytest.raises(ValueError, match="empirical mapping cap"):
            RadiallySymmetricLGNTopography(
                visual_field(), topography_parameters(cap_eccentricity=1.8)
            )

    def test_explicit_cap_above_maximum_eccentricity_is_rejected(self):
        with pytest.raises(ValueError, match="must not exceed E_max"):
            RadiallySymmetricLGNTopography(
                visual_field(), topography_parameters(cap_eccentricity=4.1)
            )

    @pytest.mark.parametrize(
        "overrides,error",
        [
            ({"full_max_eccentricity": 0.0}, "positive and finite"),
            ({"full_max_eccentricity": np.nan}, "finite real scalar"),
            ({"full_max_eccentricity": np.inf}, "finite real scalar"),
            ({"full_max_eccentricity": 1.0}, "no solution"),
            ({"beta": 0.0}, "0 < beta < 2"),
            ({"beta": 2.0}, "0 < beta < 2"),
            ({"beta": np.nan}, "finite real scalar"),
        ],
    )
    def test_invalid_mapping_calibration_is_rejected(self, overrides, error):
        with pytest.raises(ValueError, match=error):
            RadiallySymmetricLGNTopography(
                visual_field(), topography_parameters(**overrides)
            )

    def test_exact_topography_parameter_keys_are_required(self):
        missing = {
            "cap_eccentricity": None,
            "full_max_eccentricity": 90.0,
        }
        with pytest.raises(KeyError, match="missing=.*beta"):
            RadiallySymmetricLGNTopography(visual_field(), missing)

        unexpected = {
            "cap_eccentricity": None,
            "full_max_eccentricity": 90.0,
            "beta": 1.59,
            "temporary_mapping_api": True,
        }
        with pytest.raises(KeyError, match="unexpected=.*temporary_mapping_api"):
            RadiallySymmetricLGNTopography(visual_field(), unexpected)

    def test_configuration_is_copied_and_provider_is_immutable(self):
        parameters = {
            "cap_eccentricity": None,
            "full_max_eccentricity": 90.0,
            "beta": 1.59,
        }
        provider = RadiallySymmetricLGNTopography(visual_field(), parameters)
        parameters["cap_eccentricity"] = 3.0
        parameters["full_max_eccentricity"] = 45.0
        parameters["beta"] = 1.0

        assert provider.user_cap_eccentricity_deg is None
        with pytest.raises(AttributeError, match="immutable"):
            provider.max_eccentricity_deg = 3.0
        with pytest.raises(AttributeError, match="immutable"):
            provider.new_attribute = "not allowed"

    def test_visual_position_validation_accepts_boundary_and_broadcasts(self):
        provider = RadiallySymmetricLGNTopography(
            visual_field(), topography_parameters()
        )
        assert provider.validate_visual_position(4.0, 0.0) is None
        assert (
            provider.validate_visual_position(
                np.array([[0.0], [3.0]]), np.array([[0.0, 1.0]])
            )
            is None
        )

    @pytest.mark.parametrize(
        "x,y,error",
        [
            (4.0 + 1e-12, 0.0, "outside"),
            (np.nan, 0.0, "x_deg"),
            (0.0, np.inf, "y_deg"),
            (np.zeros(2), np.zeros(3), "broadcast-compatible"),
        ],
    )
    def test_invalid_visual_position_is_rejected(self, x, y, error):
        provider = RadiallySymmetricLGNTopography(
            visual_field(), topography_parameters()
        )
        with pytest.raises(ValueError, match=error):
            provider.validate_visual_position(x, y)

    def test_provider_methods_broadcast_and_return_python_float_for_scalars(self):
        provider = RadiallySymmetricLGNTopography(
            visual_field(), topography_parameters()
        )
        assert isinstance(provider.relative_density_at_eccentricity(1.0), float)
        assert isinstance(provider.relative_density_xy(1.0, 2.0), float)
        assert isinstance(provider.center_sigma_deg(1.0), float)

        x = np.array([[0.0], [1.0]])
        y = np.array([[0.0, 1.0, 2.0]])
        assert provider.relative_density_xy(x, y).shape == (2, 3)
        assert provider.center_sigma_deg(np.zeros((2, 3))).shape == (2, 3)

    def test_extrapolation_warnings_are_emitted_once_at_initialization(self, caplog):
        caplog.set_level(logging.WARNING, logger="Mozaik")
        provider = RadiallySymmetricLGNTopography(
            visual_field(size_x=178.0, size_y=178.0), topography_parameters()
        )

        density_warnings = [
            record
            for record in caplog.records
            if "density" in record.getMessage()
            and "extrapolated" in record.getMessage()
        ]
        rf_warnings = [
            record
            for record in caplog.records
            if "receptive-field sizes" in record.getMessage()
            and "extrapolated" in record.getMessage()
        ]
        assert len(density_warnings) == 1
        assert len(rf_warnings) == 1

        assert np.isfinite(provider.relative_density_at_eccentricity(89.0))
        assert np.isfinite(provider.center_sigma_deg(89.0))
        assert len(caplog.records) == 2

    def test_density_warning_starts_above_60_degrees(self, caplog):
        caplog.set_level(logging.WARNING, logger="Mozaik")
        RadiallySymmetricLGNTopography(
            visual_field(size_x=120.0, size_y=120.0), topography_parameters()
        )
        assert not any("density" in record.getMessage() for record in caplog.records)

    def test_production_module_has_no_external_development_dependency(self):
        assert "log_polar_retina" not in inspect.getsource(topography_module)


class TestDogOptimalSpatialFrequency:
    REFERENCE_PARAMETERS = {
        "Ac": 1.0,
        "As": 0.032473,
        "sigma_c": 0.103253,
        "sigma_s": 0.461558,
    }

    def test_reference_parameters(self):
        frequency = cai97.dog_optimal_spatial_frequency(**self.REFERENCE_PARAMETERS)
        assert isinstance(frequency, float)
        np.testing.assert_allclose(frequency, 0.8008972614767929, rtol=RTOL, atol=ATOL)

    def test_matches_dense_continuous_fourier_evaluation(self):
        parameters = self.REFERENCE_PARAMETERS
        analytical = cai97.dog_optimal_spatial_frequency(**parameters)
        frequencies = np.linspace(1e-6, 2.0, 400_001)
        fourier_response = (
            2.0
            * np.pi
            * (
                parameters["Ac"]
                * parameters["sigma_c"] ** 2
                * np.exp(-2.0 * np.pi**2 * parameters["sigma_c"] ** 2 * frequencies**2)
                - parameters["As"]
                * parameters["sigma_s"] ** 2
                * np.exp(-2.0 * np.pi**2 * parameters["sigma_s"] ** 2 * frequencies**2)
            )
        )
        numerical = frequencies[np.argmax(fourier_response)]
        grid_spacing = frequencies[1] - frequencies[0]

        assert abs(numerical - analytical) <= grid_spacing

    def test_frequency_scales_inversely_with_spatial_unit(self):
        reference = cai97.dog_optimal_spatial_frequency(**self.REFERENCE_PARAMETERS)
        scaled_parameters = {
            **self.REFERENCE_PARAMETERS,
            "sigma_c": 2.0 * self.REFERENCE_PARAMETERS["sigma_c"],
            "sigma_s": 2.0 * self.REFERENCE_PARAMETERS["sigma_s"],
        }
        scaled = cai97.dog_optimal_spatial_frequency(**scaled_parameters)
        np.testing.assert_allclose(scaled, reference / 2.0, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize(
        "overrides,error",
        [
            ({"Ac": 0.0}, "Ac"),
            ({"Ac": np.nan}, "Ac"),
            ({"As": -1.0}, "As"),
            ({"As": np.inf}, "As"),
            ({"sigma_c": 0.0}, "sigma_c"),
            ({"sigma_s": -1.0}, "sigma_s"),
            ({"sigma_s": np.nan}, "sigma_s"),
            ({"sigma_s": 0.103253}, "unequal"),
            ({"As": 1e-10}, "positive real nonzero optimum"),
        ],
    )
    def test_invalid_parameters_raise(self, overrides, error):
        parameters = {**self.REFERENCE_PARAMETERS, **overrides}
        with pytest.raises(ValueError, match=error):
            cai97.dog_optimal_spatial_frequency(**parameters)
