# encoding: utf-8
r"""
Retina/LGN model based on that developed by Jens Kremkow (CNRS-INCM/ALUF)
"""

import pylab
import numpy
import os.path
import pickle
import mozaik
import numbers
from decimal import Decimal, ROUND_FLOOR
from pyNN import space
from mozaik.models.vision import cai97
from mozaik.models.vision.topography import RadiallySymmetricLGNTopography
from mozaik.space import VisualSpace, VisualRegion
from mozaik.core import SensoryInputComponent
from mozaik.sheets.vision import RetinalInhomogeneousDiskSheet, RetinalUniformSheet
from mozaik.sheets.vision import VisualCorticalUniformSheet
from mozaik.tools.mozaik_parametrized import MozaikParametrized
from mozaik.tools.pyNN import *
from parameters import ParameterSet
from builtins import zip
from collections import OrderedDict
from dataclasses import dataclass
import copy
from scipy.special import ndtri

logger = mozaik.getMozaikLogger()


_CAI97_RF_NUMERIC_PARAMETERS = (
    "Ac",
    "As",
    "K1",
    "K2",
    "c1",
    "c2",
    "n1",
    "n2",
    "t1",
    "t2",
    "td",
    "sigma_c",
    "sigma_s",
)


def _round_down_to_two_significant_digits(value):
    """Round a positive finite value downward to two significant digits."""

    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not numpy.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("value must be positive and finite")
    decimal_value = Decimal(str(value))
    quantum = Decimal(1).scaleb(decimal_value.adjusted() - 1)
    return float(decimal_value.quantize(quantum, rounding=ROUND_FLOOR))


def _sample_lgn_positions(topography, number, rng):
    """Sample fixed-count Cartesian RF centres from the radial area density."""

    maximum_eccentricity = topography.max_eccentricity_deg
    maximum_density = topography.relative_density_at_eccentricity(
        topography.user_cap_eccentricity_deg
        if topography.user_cap_eccentricity_deg is not None
        else 0.0
    )
    accepted_x = []
    accepted_y = []
    accepted_count = 0

    while accepted_count < number:
        remaining = number - accepted_count
        proposal_count = max(1024, 2 * remaining)
        uniform_radius = rng.uniform(size=proposal_count)
        uniform_angle = rng.uniform(size=proposal_count)
        uniform_acceptance = rng.uniform(size=proposal_count)

        radius = maximum_eccentricity * numpy.sqrt(uniform_radius)
        angle = 2.0 * numpy.pi * uniform_angle - numpy.pi
        density = topography.relative_density_at_eccentricity(radius)
        accepted = uniform_acceptance < density / maximum_density
        accepted_indices = numpy.flatnonzero(accepted)[:remaining]
        accepted_x.append(radius[accepted_indices] * numpy.cos(angle[accepted_indices]))
        accepted_y.append(radius[accepted_indices] * numpy.sin(angle[accepted_indices]))
        accepted_count += len(accepted_indices)

    positions = numpy.vstack(
        (
            numpy.concatenate(accepted_x),
            numpy.concatenate(accepted_y),
        )
    )
    if positions.shape != (2, number):
        raise AssertionError(
            "LGN position sampling did not produce the requested count"
        )
    if numpy.any(numpy.hypot(positions[0], positions[1]) >= maximum_eccentricity):
        raise AssertionError("LGN position sampling produced a point outside the disk")
    return positions


def _sample_truncated_lognormal_scales(
    number, mu, sigma, lower_quantile, upper_quantile, rng
):
    """Sample a log-normal distribution conditioned on a CDF interval."""

    quantiles = rng.uniform(lower_quantile, upper_quantile, size=number)
    return numpy.exp(mu + sigma * ndtri(quantiles))


def _temporal_scale_rngs(pynn_seed):
    """Return deterministic private ON/OFF streams derived from ``pynn_seed``."""

    # The fixed stream tag isolates temporal-scale sampling from PyNN's shared
    # RNG state while retaining pynn_seed as the sole source of randomness.

    # TODO: once the seed-separation refactor is merged into this branch,
    # set this using model_seed!!! and remove this comment.
    seed_sequence = numpy.random.SeedSequence([int(pynn_seed), 0x4C474E54])
    return tuple(
        numpy.random.RandomState(int(child.generate_state(1, dtype=numpy.uint32)[0]))
        for child in seed_sequence.spawn(2)
    )


def meshgrid3D(x, y, z):
    r"""A slimmed-down version of http://www.scipy.org/scipy/numpy/attachment/ticket/966/meshgrid.py"""
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    z = numpy.asarray(z)
    mult_fact = numpy.ones((len(x), len(y), len(z)))
    nax = numpy.newaxis
    return (
        x[:, nax, nax] * mult_fact,
        y[nax, :, nax] * mult_fact,
        z[nax, nax, :] * mult_fact,
    )


class SpatioTemporalReceptiveField(object):
    r"""
    Implements spatio-temporal receptive field.

    Parameters
    ----------

    func : function
        should be a function of x, y, t, and a ParameterSet object

    func_params : ParameterSet
        ParameterSet that is passed to `func`.

    width : float (degrees)
        x-dimension size

    height : float (degrees)
        y-dimension size

    duration : float (ms)
        length of the temporal axis of the RF

    polarity : float
        Receptive-field sign. Must be either 1 for ON or -1 for OFF.

    Notes
    -----

    Coordinates x = 0 and y = 0 are at the centre of the spatial kernel.
    A factorized RF stores ``K(x, y, t) = sum_i S_i(x, y) T_i(t)``. Each
    spatial factor is either a 2-D array or a pair of 1-D axis factors.

    """

    def __init__(self, func, func_params, width, height, duration, polarity=1.0):
        self.func = func
        self.func_params = func_params
        self.width = float(width)
        self.height = float(height)
        self.duration = float(duration)
        self.polarity = float(polarity)
        if self.polarity not in (-1.0, 1.0):
            raise ValueError("polarity must be either -1 or 1")
        self.kernel = None
        self.factors = None
        self.shape = None
        self.x_coordinates = None
        self.y_coordinates = None
        self.time_points = None
        self.normalization_denominator = None
        self.spatial_resolution = numpy.inf
        self.temporal_resolution = numpy.inf

    def quantize(self, dx, dy, dt, use_factorization=False):
        r"""
        Quantizes the the receptive field.

        Parameters
        ----------

        dx : float
            Difference between pixel positions along the x axis.

        dy : float
            Difference between pixel positions along the y axis.

        dt : float
            Difference between timesteps.

        use_factorization : bool
            If True and ``func`` exposes a callable ``factorize`` attribute,
            store its spatial/temporal factors instead of evaluating a dense
            kernel. Callables without that attribute retain the dense path.

        Notes
        -----

        If `dx` does not
        divide exactly into the width, then the actual width will be slightly
        larger than the nominal width. `dx` and `dy` should be in degrees and `dt` in ms.


        """
        assert dx == dy  # For now, at least
        nx = int(numpy.ceil(self.width / dx))
        ny = int(numpy.ceil(self.height / dy))
        nt = int(numpy.ceil(self.duration / dt))
        width = nx * dx
        height = ny * dy
        duration = nt * dt
        # x and y are the coordinates of the centre of each pixel
        # I use linspace instead of arange as arange sometimes can return inconsistent number of elements
        # (must have something to do with rounding errors)
        # x = numpy.arange(0.0, width, dx)  + dx/2.0 - width/2.0
        # y = numpy.arange(0.0, height, dy) + dy/2.0 - height/2.0

        x = numpy.linspace(0.0, width - dx, nx) + dx / 2.0 - width / 2.0
        y = numpy.linspace(0.0, height - dy, ny) + dx / 2.0 - height / 2.0

        # t is the time at the beginning of each timestep
        t = numpy.arange(0.0, duration, dt)
        self.shape = (ny, nx, nt)
        self.x_coordinates = x
        self.y_coordinates = y
        self.time_points = t
        self.normalization_denominator = nx * ny * nt

        factorizer = getattr(self.func, "factorize", None)
        if use_factorization and callable(factorizer):
            # The general term sequence currently represents the two Cai97
            # centre/surround terms. Supporting more terms keeps the RF
            # representation general without changing its present purpose:
            # reducing receptive-field memory and evaluation work.
            self.factors = self._validate_factors(
                factorizer(y, x, t, self.func_params), self.shape
            )
            self.kernel = None
        else:
            X, Y, T = meshgrid3D(y, x, t)  # x,y are reversed because (x,y) <--> (j,i)
            kernel = self.polarity * self.func(X, Y, T, self.func_params)
            # logger.debug("Created receptive field kernel: width=%gº, height=%gº, duration=%g ms, shape=%s" %
            #                 (width, height, duration, kernel.shape))
            # logger.debug("before normalization: min=%g, max=%g" %
            #                 (kernel.min(), kernel.max()))
            kernel = kernel / (
                self.normalization_denominator
            )  # normalize to make the kernel sum quasi-independent of the quantization
            # logger.debug("  after normalization: min=%g, max=%g, sum=%g" %
            #                 (kernel.min(), kernel.max(), kernel.sum()))
            self.kernel = kernel
            self.factors = None
        self.spatial_resolution = dx
        self.temporal_resolution = dt

    @staticmethod
    def _validate_factors(factors, kernel_shape):
        base_message = (
            "factorize() must return one or more numeric spatial/temporal "
            "factor pairs matching the RF shape"
        )

        def factor_error(number, component):
            return ValueError(
                f"{base_message}; factor {number} has an invalid {component}"
            )

        def validated_array(value, expected_shape, number, component):
            try:
                array = numpy.array(value, dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise factor_error(number, component) from exc
            if array.shape != expected_shape:
                raise factor_error(number, component)
            array.setflags(write=False)
            return array

        try:
            factors = tuple(factors)
        except TypeError as exc:
            raise ValueError(base_message) from exc
        if not factors:
            raise ValueError(base_message)

        validated = []
        for factor_number, factor in enumerate(factors, start=1):
            try:
                spatial, temporal = factor
            except (TypeError, ValueError) as exc:
                raise factor_error(factor_number, "factor structure") from exc

            spatial_is_separable = isinstance(spatial, tuple)
            spatial_values = spatial if spatial_is_separable else (spatial,)
            expected_spatial_shapes = (
                ((kernel_shape[0],), (kernel_shape[1],))
                if spatial_is_separable
                else (kernel_shape[:2],)
            )
            if len(spatial_values) != len(expected_spatial_shapes):
                raise factor_error(factor_number, "spatial component")
            spatial_arrays = tuple(
                validated_array(value, shape, factor_number, "spatial component")
                for value, shape in zip(spatial_values, expected_spatial_shapes)
            )
            temporal = validated_array(
                temporal, kernel_shape[2:], factor_number, "temporal component"
            )
            spatial = spatial_arrays if spatial_is_separable else spatial_arrays[0]
            validated.append((spatial, temporal))
        return tuple(validated)

    @property
    def is_factorized(self):
        return self.factors is not None

    def as_dense(self):
        """Return the quantized receptive field as a dense 3-D array."""

        if self.shape is None:
            raise RuntimeError("receptive field has not been quantized")
        if self.kernel is not None:
            return self.kernel

        kernel = numpy.zeros(self.shape, dtype=float)
        for spatial, temporal in self.factors:
            if isinstance(spatial, tuple):
                spatial = numpy.multiply.outer(*spatial)
            kernel += spatial[:, :, numpy.newaxis] * temporal
        return self.polarity * kernel / self.normalization_denominator

    @property
    def kernel_duration(self):
        r"""
        Returns the temporal duration of the quantized kernel.

        Notes
        -----

        This relies on the kernel having been quantized. If not, accessing this will raise an error.
        """
        return self.shape[2]

    def __str__(self):
        s = "Receptive field: width=%gº, height=%gº, duration=%g ms" % (
            self.width,
            self.height,
            self.duration,
        )
        if self.shape is not None:
            h, w = self.shape[0:2]
            factorization = (
                "%d factors" % len(self.factors) if self.is_factorized else "dense"
            )
            s += (
                ", quantization=%s, representation=%s, actual width=%gº, "
                "actual_height=%gº."
                % (
                    self.shape,
                    factorization,
                    w * self.spatial_resolution,
                    h * self.spatial_resolution,
                )
            )
        else:
            s += ". Not quantized."
        return s


@dataclass
class KernelResponse:
    r"""
    Unscaled receptive-field convolution output split into two components.

    Parameters
    ----------
    contrast : ndarray
        Response of the zero-mean spatial component of the receptive-field
        kernel. This component is normalized by background luminance when
        frames are sampled.

    luminance : ndarray
        Response of the spatially uniform component of the receptive-field
        kernel. This component tracks absolute image luminance and is used to
        model luminance-dependent gain separately from contrast-dependent gain.
    """

    contrast: numpy.ndarray
    luminance: numpy.ndarray


class KernelResponseOperator:
    r"""Compute contrast and luminance responses for a receptive field.

    The luminance response represents the contribution of the spatially
    uniform part of the receptive field. The contrast response represents the
    remaining spatial variation, whose mean is zero.

    The luminance kernel is the spatial mean of the full kernel. The contrast
    kernel is what remains after subtracting that mean from the full kernel.

    The luminance kernel is multiplied by ``luminance_spatial_scale``: legacy cells
    use 1 to preserve the historical kernel-mean x image-mean calculation, while
    eccentricity-dependent cells use the number of spatial samples to convert
    the kernel mean into a spatial sum, preventing luminance responses from
    artificially decreasing as receptive fields grow.

    Factorized receptive fields are evaluated directly from their factors;
    dense receptive fields use the existing dense calculation.
    """

    def __init__(self, receptive_field, luminance_spatial_scale=1.0):
        if receptive_field.shape is None:
            raise ValueError("receptive field must be quantized")

        self.receptive_field = receptive_field
        self.spatial_shape = receptive_field.shape[:2]
        self.kernel_contrast_component = None

        # Precompute the image-independent coefficients without constructing
        # a dense 3-D kernel.
        if receptive_field.is_factorized:
            self._spatial_factors = tuple(
                spatial for spatial, _ in receptive_field.factors
            )
            normalization = (
                receptive_field.polarity / receptive_field.normalization_denominator
            )
            self._temporal_factors = normalization * numpy.column_stack(
                [temporal for _, temporal in receptive_field.factors]
            )
            self._spatial_means = numpy.array(
                [self._spatial_mean(spatial) for spatial in self._spatial_factors]
            )
            spatial_mean = self._temporal_factors @ self._spatial_means
        else:
            # Dense RF callables retain the legacy precomputed contrast matrix.
            self._spatial_factors = None
            self._temporal_factors = None
            self._spatial_means = None
            spatial_mean = receptive_field.kernel.mean(axis=(0, 1))
            contrast_kernel = receptive_field.kernel - spatial_mean
            self.kernel_contrast_component = contrast_kernel.reshape(
                -1, receptive_field.kernel_duration
            ).T

        # Either representation reduces luminance to a single temporal course.
        self.kernel_luminance_component = float(luminance_spatial_scale) * spatial_mean
        # Operators may be shared by many cells, so keep coefficients immutable.
        for array in (
            self._temporal_factors,
            self._spatial_means,
            self.kernel_contrast_component,
            self.kernel_luminance_component,
        ):
            if array is not None:
                array.setflags(write=False)

    @staticmethod
    def _spatial_mean(spatial):
        if isinstance(spatial, tuple):
            return numpy.prod([axis.mean() for axis in spatial])
        return spatial.mean()

    @staticmethod
    def _spatial_projection(spatial, image):
        if isinstance(spatial, tuple):
            return numpy.dot(numpy.dot(spatial[0], image), spatial[1])
        return numpy.dot(spatial.reshape(-1), image.reshape(-1))

    def __call__(self, image, background_luminance):
        """Response to a single rendered image patch."""

        image = numpy.asarray(image)
        if image.shape != self.spatial_shape:
            raise ValueError(
                "image patch shape %s does not match receptive-field shape %s"
                % (image.shape, self.spatial_shape)
            )

        image_sum = image.sum()
        if self._spatial_factors is None:
            contrast = numpy.dot(self.kernel_contrast_component, image.reshape(-1))
        else:
            projections = numpy.array(
                [
                    self._spatial_projection(spatial, image)
                    for spatial in self._spatial_factors
                ]
            )
            contrast = self._temporal_factors @ (
                projections - self._spatial_means * image_sum
            )

        # Normalize contrast by background luminance so it is independent of
        # the overall luminance level.
        # Applying the spatially uniform luminance kernel at every pixel and
        # summing is equivalent to scaling its 1-D course by the image mean.
        return KernelResponse(
            contrast=contrast / background_luminance,
            luminance=self.kernel_luminance_component * (image_sum / image.size),
        )


class CellWithReceptiveField(object):
    r"""
    A model of the input current generated by retinothalamic preprocessing for
    a simulated thalamic relay neuron. It multiplies, in space and time,
    the luminance values impinging on its receptive field by a spatiotemporal
    kernel. Spatial summation over the result of this multiplication at each
    time point gives a current in nA, that may then be injected into a
    simulated thalamic relay neuron.

    initialize() should be called once, before stimulus presentation
    view() should be called in a loop, once for each stimulus frame
    response_current() should be called at the end of stimulus presentation

    Parameters
    ----------

    x , y : float
        x and y coordinates of the center of the RF in visual space.

    receptive_field : SpatioTemporalReceptiveField
        The receptive field object containing the RFs data.

    gain_control : ParameterSet
        The calculated input current values will be multiplied by the gain
        parameter (gain_control.gain), if gain_control.non_linear_gain is None.

        Otherwise, the input current values can be further scaled by a nonlinear
        gain function according to luminance and contrast (parameters set by
        gain_control.non_linear_gain). This nonlinear function mimics the luminance
        and contrast saturation effects of retina interneurons.

    visual_space : VisualSpace
        The object representing the visual space.

    original_2024_lgn_mode : bool
        If True, reproduce the LGN input-current behavior used by the 2024
        LSV1M model version: no filter state is retained between consecutive
        stimuli and the pre-stimulus luminance state is initialized as if the
        filter had previously seen zero luminance. If False, the cell keeps the
        trailing kernel state between presentations and initializes from the
        visual-space background luminance.

    kernel_response_operator : KernelResponseOperator, optional
        Precomputed linear-response operator. Supplying it allows cells that
        share an RF to share its immutable response coefficients as well.

    """

    def __init__(
        self,
        x,
        y,
        receptive_field,
        gain_control,
        visual_space,
        original_2024_lgn_mode=False,
        kernel_response_operator=None,
    ):
        self.x = x  # position in space
        self.y = y  #
        self.visual_space = visual_space
        assert isinstance(receptive_field, SpatioTemporalReceptiveField)
        self.receptive_field = receptive_field
        self.gain_control = (
            gain_control  # (nA.m²/cd) could imagine making this a function of
        )
        # the background luminance
        self.i = 0
        self.visual_region = VisualRegion(
            location_x=self.x,
            location_y=self.y,
            size_x=self.receptive_field.width,
            size_y=self.receptive_field.height,
        )
        # logger.debug("view_array.shape = %s" % str(view_array.shape))
        # logger.debug("receptive_field.shape = %s" % str(self.receptive_field.shape))
        # logger.debug("response.shape = %s" % str(self.response.shape))
        if visual_space.update_interval % self.receptive_field.temporal_resolution != 0:
            errmsg = (
                "The receptive field temporal resolution (%g ms) must be an integer multiple of the visual space update interval (%g ms)"
                % (
                    self.receptive_field.temporal_resolution,
                    visual_space.update_interval,
                )
            )
            raise Exception(errmsg)
        self.update_factor = int(
            visual_space.update_interval / self.receptive_field.temporal_resolution
        )
        self.original_2024_lgn_mode = original_2024_lgn_mode
        self.filter_state = None

        self.background_luminance = visual_space.background_luminance
        # The kernel is separated into a luminance and contrast component,
        # which are then scaled separately by the non-linear gain.
        # This separability is based on doi:10.1038/nn1556, although they
        # do the separation there in a different way - multiplicatively, where
        # the partial kernels themselves change with luminance and contrast
        rf = self.receptive_field
        self._validate_receptive_field_shape(rf.shape)
        self.kernel_response_operator = kernel_response_operator
        if self.kernel_response_operator is None:
            self.kernel_response_operator = KernelResponseOperator(
                rf, self._luminance_spatial_scale(rf.shape)
            )
        elif self.kernel_response_operator.receptive_field is not rf:
            raise ValueError("kernel response operator does not match receptive field")
        luminance_kernel = self.kernel_response_operator.kernel_luminance_component
        self.null_response = KernelResponse(
            contrast=0,
            luminance=luminance_kernel.sum() * self.background_luminance,
        )

        # Unscaled cumulative sum of the luminance kernel (length = kdur)
        kernel_cumsum = numpy.cumsum(luminance_kernel)
        # Padded, scaled version for background-luminance step response
        self.kernel_luminance_cumsum_padded = numpy.concatenate(
            [[0], self.background_luminance * kernel_cumsum]
        )
        self.luminance_step_response = self.kernel_luminance_cumsum_padded
        self.luminance_steady_state = self.luminance_step_response[-1]

        # Set starting luminance (default = background luminance)
        self.starting_luminance = (
            0 if self.original_2024_lgn_mode else self.background_luminance
        )

        # Tail from infinite past of constant starting_luminance
        self.starting_luminance_kernel_state = self.starting_luminance * (
            kernel_cumsum[-1] - kernel_cumsum[:-1]
        )

    def _validate_receptive_field_shape(self, kernel_shape):
        assert (
            kernel_shape[0] == kernel_shape[1]
        ), "With the current implementation, receptive fields must be symmetric!"

    def _luminance_spatial_scale(self, kernel_shape):
        """Return the legacy mean-normalized luminance scale."""

        return 1.0

    def initialize(self, stimulus_duration):
        r"""
        Create the array that will contain the current response, and set the
        initial values on the assumption that the system was looking at a blank
        screen of constant luminance prior to stimulus onset.

        Parameters
        ----------

        stimulus_duration : float (ms)
            The duration  of the visual stimulus.

        Notes
        -----

        The response buffer includes extra samples equal to the temporal kernel
        duration. These samples hold the filter tail and are removed when
        converting the kernel response into an injected current.
        """
        rf = self.receptive_field
        self.response_length = int(
            numpy.ceil(stimulus_duration / rf.temporal_resolution) + rf.kernel_duration
        )

        # We assume the model has been looking at starting luminance prior to
        # stimulus onset; thus the initial response is the accumulation of luminance
        # kernel convolution at starting luminance until the present time point
        if self.filter_state is None:
            self.kernel_response = KernelResponse(
                contrast=numpy.zeros(self.response_length),
                luminance=numpy.pad(
                    self.starting_luminance_kernel_state,
                    (
                        0,
                        self.response_length
                        - len(self.starting_luminance_kernel_state),
                    ),
                ),
            )
        else:
            self.kernel_response = KernelResponse(
                contrast=numpy.pad(
                    self.filter_state.contrast,
                    (0, self.response_length - len(self.filter_state.contrast)),
                ),
                luminance=numpy.pad(
                    self.filter_state.luminance,
                    (0, self.response_length - len(self.filter_state.luminance)),
                ),
            )
        assert rf.kernel_duration <= self.response_length
        self.i = 0

    def view(self):
        r"""
        Look at the visual space and update t
        Where the kernel temporal resolution is the same as the frame duration
        (visual space update interval):
        R_i = Sum[j=0,L-1] K_j.I_i-j
        where L is the kernel length/duration
        Where the kernel temporal resolution = (frame duration)/α (α an integer)
        R_k = Sum[j=0,L-1] K_j.I_i'
        where i' = (k-j)//α  (// indicates integer division, discarding the
        remainder)
        To avoid loading the entire image sequence into memory, we build up the response array one frame at a time.
        """
        view_array = self.visual_space.view(
            self.visual_region, pixel_size=self.receptive_field.spatial_resolution
        )
        linear_response = self.kernel_response_operator(
            view_array, self.background_luminance
        )
        contrast_time_course = linear_response.contrast
        luminance_time_course = linear_response.luminance
        self.va = view_array

        d = self.receptive_field.kernel_duration
        if self.update_factor != 1.0:
            for j in range(self.i, self.i + self.update_factor):
                self.kernel_response.contrast[j : j + d] += contrast_time_course[:d]
                self.kernel_response.luminance[j : j + d] += luminance_time_course[:d]
        else:
            self.kernel_response.contrast[self.i : self.i + d] += contrast_time_course[
                :d
            ]
            self.kernel_response.luminance[
                self.i : self.i + d
            ] += luminance_time_course[:d]

        self.i += (
            self.update_factor
        )  # we assume there is only ever 1 visual space used between initializations

    def gain_function(self, response, gain, scaler):
        r"""
        Scale the response by a symmetric Naka-Rushton function to
        achieve the variable luminance/contrast gain observed in the retina.
        """
        return gain * response / (numpy.abs(response) + scaler)

    def response_current(self, kernel_response):
        r"""
        Multiply the response (units of luminance (cd/m²) if we assume the
        kernel values are dimensionless) by the 'gain', to produce a current in
        nA. Returns a dictionary containing 'times' and 'amplitudes'.

        Parameters
        ----------
        kernel_response : KernelResponse
            Contrast and luminance components of the convolution response for
            one stimulus presentation.

        Returns
        -------
        dict
            Dictionary with ``times`` in ms and ``amplitudes`` in nA. The final
            kernel-duration tail is omitted from the returned arrays. In normal
            mode, that omitted tail is retained internally and used as the
            initial filter state for the next presentation.
        """
        kdur = self.receptive_field.kernel_duration
        # We scale the luminance and contrast components separately,
        # and then add them
        nlg = self.gain_control.non_linear_gain
        contrast_response = self.gain_function(
            kernel_response.contrast, nlg.contrast_gain, nlg.contrast_scaler
        )
        luminance_response = self.gain_function(
            kernel_response.luminance, nlg.luminance_gain, nlg.luminance_scaler
        )
        response = contrast_response + luminance_response

        # Remove extra padding at the end
        response = response[:-kdur]
        time_points = self.receptive_field.temporal_resolution * numpy.arange(
            0, len(response)
        )

        if not self.original_2024_lgn_mode:
            self.filter_state = KernelResponse(
                contrast=kernel_response.contrast[-kdur:].copy(),
                luminance=kernel_response.luminance[-kdur:].copy(),
            )
        return {"times": time_points, "amplitudes": response}


class EccentricityDependentCellWithReceptiveField(CellWithReceptiveField):
    """Per-cell eccentricity RF path, including rectangular scaled support."""

    def _validate_receptive_field_shape(self, kernel_shape):
        pass

    def _luminance_spatial_scale(self, kernel_shape):
        """Scale the spatial mean into the full spatial sum."""

        return kernel_shape[0] * kernel_shape[1]


class SpatioTemporalFilterRetinaLGN(SensoryInputComponent):
    r"""
    Retinothalamic preprocessing with spatiotemporal receptive fields feeding
    simulated thalamic relay neurons.

    Parameters
    ----------

    density : int (1/degree^2)
        Number of neurons to simulate per square degree of visual space.

    size : tuple (degree,degree)
        The x and y size of the visual field.

    linear_scaler : float
        The linear scaler that the RF output is multiplied with.

    mpi_reproducible_noise : bool
        If true the background noise is generated in such a way that is
        reproducible across runs using different numbers of MPI processes.
        Significant slowdown if True.

    recorders : ParameterSet
        Recorder configuration passed to the simulated thalamic relay sheets.

    recording_interval : float
        Sampling interval for recorded variables on the simulated thalamic
        relay sheets.

    receptive_field : ParameterSet
        Parameters describing the receptive-field function, its spatial and
        temporal extent, and the quantization resolution.

    original_2024_lgn_mode : bool
        If True, use the legacy LGN behavior needed to reproduce the 2024 LSV1M
        reference data. This disables filter-state carry-over, starts filters
        from zero luminance, and uses the historical optimized null-stimulus
        current injection path. If False, filter state is retained across
        consecutive presentations and null input is computed through the same
        kernel-response machinery as explicit stimuli.

    cell : ParameterSet
        PyNN cell configuration for the simulated thalamic relay cell
        populations.

    gain_control : ParameterSet
        Gain and nonlinear gain parameters used to transform receptive-field
        responses into injected currents.

    noise : ParameterSet
        Mean and standard deviation of the additive current noise.

    Notes
    -----

    Stimulus responses are cached only in memory for the lifetime of the
    component. The cache is keyed by the stimulus identity with ``trial`` ignored,
    so repeated trials can reuse the same receptive-field convolution while
    still receiving independently generated noise.

    """

    required_parameters = ParameterSet(
        {
            "density": int,  # neurons per degree squared
            "size": tuple,  # degrees of visual field
            "linear_scaler": float,  # linear scaler that the RF output is multiplied with
            "mpi_reproducible_noise": bool,  # if True, noise is precomputed and StepCurrentSource is used which makes it slower
            "recorders": ParameterSet,
            "recording_interval": float,
            "receptive_field": ParameterSet(
                {
                    "func": str,
                    "func_params": ParameterSet,
                    "width": float,
                    "height": float,
                    "spatial_resolution": float,
                    "temporal_resolution": float,
                    "duration": float,
                }
            ),
            "original_2024_lgn_mode": bool,
            "cell": ParameterSet(
                {
                    "model": str,
                    "params": ParameterSet,
                    "receptors": ParameterSet,
                    "native_nest": bool,
                    "initial_values": ParameterSet,
                }
            ),
            "gain_control": {
                "gain": float,
                "non_linear_gain": ParameterSet(
                    {
                        "luminance_gain": float,
                        "luminance_scaler": float,
                        "contrast_gain": float,
                        "contrast_scaler": float,
                    }
                ),
            },
            "noise": ParameterSet(
                {
                    "mean": float,
                    "stdev": float,  # nA
                }
            ),
        }
    )

    def __init__(self, model, parameters):
        SensoryInputComponent.__init__(self, model, parameters)
        self.shape = (self.parameters.density, self.parameters.density)
        self.sheets = OrderedDict()
        self.rf_types = ("X_ON", "X_OFF")
        sim = self.model.sim
        self.pops = OrderedDict()

        if self.parameters.cell.model[-6:] == "_sc_nc":
            self.integrated_cs = True
            import copy

            cell = copy.deepcopy(self.parameters.cell)
            cell.params.update(
                [
                    ("mean", self.parameters.noise.mean * 1000),
                    ("std", self.parameters.noise.stdev * 1000),
                    ("dt", self.model.sim.get_time_step()),
                ]
            )
        else:
            self.integrated_cs = False
            self.scs = OrderedDict()
            self.ncs = OrderedDict()
            cell = self.parameters.cell

        self.ncs_rng = OrderedDict()
        self.internal_stimulus_cache = OrderedDict()
        for rf_type in self.rf_types:

            p = RetinalUniformSheet(
                model,
                ParameterSet(
                    {
                        "sx": self.parameters.size[0],
                        "sy": self.parameters.size[1],
                        "density": self.parameters.density,
                        "cell": cell,
                        "name": rf_type,
                        "artificial_stimulators": OrderedDict(),
                        "recorders": self.parameters.recorders,
                        "recording_interval": self.parameters.recording_interval,
                        "mpi_safe": False,
                    }
                ),
            )
            self.sheets[rf_type] = p

        for rf_type in self.rf_types:
            self.ncs_rng[rf_type] = []
            seeds = mozaik.get_seeds((self.sheets[rf_type].pop.size,))

            if self.integrated_cs:
                for i, lgn_cell in enumerate(self.sheets[rf_type].pop.all_cells):
                    if self.sheets[rf_type].pop._mask_local[i]:
                        self.ncs_rng[rf_type].append(
                            numpy.random.RandomState(seed=seeds[i])
                        )
            else:
                self.scs[rf_type] = []
                self.ncs[rf_type] = []
                for i, lgn_cell in enumerate(self.sheets[rf_type].pop.all_cells):
                    scs = sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])

                    if not self.parameters.mpi_reproducible_noise:
                        ncs = sim.NoisyCurrentSource(**self.parameters.noise)
                    else:
                        ncs = sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])

                    if self.sheets[rf_type].pop._mask_local[i]:
                        self.ncs_rng[rf_type].append(
                            numpy.random.RandomState(seed=seeds[i])
                        )
                        self.scs[rf_type].append(scs)
                        self.ncs[rf_type].append(ncs)
                    lgn_cell.inject(scs)
                    lgn_cell.inject(ncs)

        P_rf = self.parameters.receptive_field
        rf_function = eval(P_rf.func)

        rf_ON = SpatioTemporalReceptiveField(
            rf_function, P_rf.func_params, P_rf.width, P_rf.height, P_rf.duration
        )
        rf_OFF = SpatioTemporalReceptiveField(
            rf_function,
            P_rf.func_params,
            P_rf.width,
            P_rf.height,
            P_rf.duration,
            polarity=-1.0,
        )

        dx = dy = P_rf.spatial_resolution
        dt = P_rf.temporal_resolution
        for rf in rf_ON, rf_OFF:
            rf.quantize(dx, dy, dt, use_factorization=True)

        self.rf = {"X_ON": rf_ON, "X_OFF": rf_OFF}
        self.kernel_response_operators = {
            rf_type: KernelResponseOperator(rf)
            for rf_type, rf in self.rf.items()
        }

        # create population of CellWithReceptiveFields, setting the receptive
        # field centres based on the size/location of self
        logger.debug("Creating population of `CellWithReceptiveField`s")
        self.input_cells = OrderedDict()
        for rf_type in self.rf_types:
            self.input_cells[rf_type] = []
            for i in numpy.nonzero(self.sheets[rf_type].pop._mask_local)[0]:
                cell = CellWithReceptiveField(
                    self.sheets[rf_type].pop.positions[0][i],
                    self.sheets[rf_type].pop.positions[1][i],
                    self.rf[rf_type],
                    self.parameters.gain_control,
                    model.input_space,
                    self.parameters.original_2024_lgn_mode,
                    self.kernel_response_operators[rf_type],
                )
                self.input_cells[rf_type].append(cell)

    @property
    def visual_space_resolution_deg(self):
        """Return the configured legacy visual-space pixel size in degrees."""

        return self.parameters.receptive_field.spatial_resolution

    def _noise_parameters(self, rf_type):
        """Return the noise parameters used by one response polarity."""

        return self.parameters.noise

    def process_input(self, visual_space, stimulus, duration=None, offset=0):
        r"""
        Present a visual stimulus to the model and create currents for the
        simulated thalamic relay neurons.

        Parameters
        ----------

        visual_space : VisualSpace
            The visual space to which the stimuli are presented.

        stimulus : VisualStimulus
            The visual stimulus to be shown.

        duration : int (ms)
            The time for which we will simulate the stimulus

        offset : int(ms)
            The time (in absolute time of the whole simulation) at which the stimulus starts.


        Returns
        -------

        retinal_input : list(ndarray)
            List of 2D arrays containing the frames of luminances that were presented to the retina.


        """
        logger.debug("Presenting visual stimulus from visual space %s" % visual_space)
        visual_space.set_duration(duration)
        self.input = visual_space
        st = MozaikParametrized.idd(stimulus)
        st.trial = None  # to avoid recalculating RFs response to multiple trials of the same stimulus

        # We check if we haven't already presented it during this simulation run.
        # This is mainly to avoid regenerating stimuli for multiple trials.
        if str(st) in self.internal_stimulus_cache:
            kernel_responses, retinal_input = self.internal_stimulus_cache[str(st)]
        else:
            kernel_responses, retinal_input = self.calculate_kernel_responses(
                visual_space, duration
            )
        input_currents = self._calculate_input_currents(kernel_responses)
        self.inject_currents(input_currents, duration, offset)

        # also save into internal cache
        self.internal_stimulus_cache[str(st)] = (
            copy.deepcopy(kernel_responses),
            retinal_input,
        )

        return retinal_input

    def inject_currents(self, input_currents, duration=None, offset=0):
        r"""
        Inject precomputed current traces into the simulated thalamic relay
        neurons.

        Parameters
        ----------
        input_currents : OrderedDict
            Mapping from RF type (``X_ON`` or ``X_OFF``) to a list of current
            dictionaries, one per local simulated thalamic relay neuron. Each
            dictionary contains ``times`` in ms and ``amplitudes`` in nA.

        duration : float (ms)
            Duration of the current injection. Used to generate reproducible
            noise traces when ``mpi_reproducible_noise`` is enabled.

        offset : float (ms)
            Absolute simulator time at which the stimulus starts.

        Notes
        -----
        The integrated current-source cell model expects amplitudes in pA, so
        amplitudes are scaled by 1000 on that path. In legacy 2024 mode, the
        historical PyNN/NEST offset correction is preserved only for the
        integrated current-source path; ordinary StepCurrentSource injection
        keeps the old uncorrected timing.
        """
        ts = self.model.sim.get_time_step()
        if self.parameters.original_2024_lgn_mode and self.integrated_cs:
            offset = convert_time_pyNN_to_nest(self.model.sim, offset) + ts

        for rf_type in self.rf_types:
            assert isinstance(input_currents[rf_type], list)
            if self.integrated_cs:
                for i, (lgn_cell, input_current) in enumerate(
                    zip(self.sheets[rf_type].pop, input_currents[rf_type])
                ):
                    assert isinstance(input_current, dict)
                    t = input_current["times"] + offset
                    a = self.parameters.linear_scaler * input_current["amplitudes"]
                    lgn_cell.set_parameters(
                        amplitude_times=t[1:], amplitude_values=a[1:] * 1000
                    )

            else:
                with self.model.sim.state.freeze_time():
                    for i, (lgn_cell, input_current, scs, ncs) in enumerate(
                        zip(
                            self.sheets[rf_type].pop,
                            input_currents[rf_type],
                            self.scs[rf_type],
                            self.ncs[rf_type],
                        )
                    ):
                        assert isinstance(input_current, dict)
                        t = input_current["times"] + offset
                        a = self.parameters.linear_scaler * input_current["amplitudes"]
                        scs.set_parameters(times=t, amplitudes=a, copy=False)
                        if self.parameters.mpi_reproducible_noise:
                            t = numpy.arange(0, duration, ts) + offset
                            noise = self._noise_parameters(rf_type)
                            amplitudes = noise.mean + noise.stdev * self.ncs_rng[
                                rf_type
                            ][i].randn(len(t))
                            ncs.set_parameters(
                                times=t, amplitudes=amplitudes, copy=False
                            )

    def _provide_legacy_null_input(self, visual_space, duration=None, offset=0):
        r"""
        Inject the optimized null-stimulus current used by legacy 2024 runs.

        This is intentionally separate from ``calculate_null_input`` and
        ``inject_currents`` because the old implementation did not construct a
        full time series. It injected one constant two-point current step per RF
        type and used a historical StepCurrentSource timing workaround
        (``offset + 3 * timestep``) on the non-integrated current-source path.
        """
        ts = self.model.sim.get_time_step()
        if self.integrated_cs:
            new_offset = convert_time_pyNN_to_nest(self.model.sim, offset) + ts
            times = numpy.array(
                [new_offset, duration - visual_space.update_interval + new_offset]
            )
        else:
            times = numpy.array(
                [offset + 3 * ts, duration - visual_space.update_interval + offset]
            )

        for rf_type in self.rf_types:
            cell = self.input_cells[rf_type][0]
            nlg = cell.gain_control.non_linear_gain
            amplitude = cell.null_response.luminance
            amplitude = self.parameters.linear_scaler * cell.gain_function(
                amplitude, nlg.luminance_gain, nlg.luminance_scaler
            )

            if self.integrated_cs:
                for lgn_cell in self.sheets[rf_type].pop:
                    lgn_cell.set_parameters(
                        amplitude_times=times,
                        amplitude_values=numpy.zeros_like(times) + amplitude * 1000,
                    )
            else:
                with self.model.sim.state.freeze_time():
                    for i, (scs, ncs) in enumerate(
                        zip(self.scs[rf_type], self.ncs[rf_type])
                    ):
                        scs.set_parameters(
                            times=times,
                            amplitudes=numpy.zeros_like(times) + amplitude,
                            copy=False,
                        )
                        if self.parameters.mpi_reproducible_noise:
                            t = numpy.arange(0, duration, ts) + offset
                            noise = self._noise_parameters(rf_type)
                            amplitudes = noise.mean + noise.stdev * self.ncs_rng[
                                rf_type
                            ][i].randn(len(t))
                            ncs.set_parameters(
                                times=t, amplitudes=amplitudes, copy=False
                            )

    def calculate_null_input(self, duration=None):
        r"""
        Calculate current traces for a blank screen, retaining the convolution
        tail.

        Parameters
        ----------
        duration : float (ms)
            Duration of the blank presentation.

        Returns
        -------
        OrderedDict
            Mapping from RF type to a list of current dictionaries, one per
            local simulated thalamic relay neuron. Each dictionary contains
            ``times`` in ms and ``amplitudes`` in nA.

        Notes
        -----
        This method is used only when ``original_2024_lgn_mode`` is False.
        Legacy-mode null input is handled by ``_provide_legacy_null_input`` so
        that the old shortcut remains isolated and easy to remove.
        """
        input_currents = OrderedDict()
        for rf_type in self.rf_types:
            input_currents[rf_type] = []
            for cell in self.input_cells[rf_type]:
                num_frames = (
                    int(numpy.ceil(duration / cell.visual_space.update_interval))
                    * cell.update_factor
                )
                cell.initialize(duration)

                kdur = cell.receptive_field.kernel_duration
                # Build background luminance step response extended to the
                # needed length (num_frames + kdur)
                step_on = numpy.concatenate(
                    [
                        cell.luminance_step_response[1:],  # t = 1..kdur
                        numpy.full(
                            num_frames, cell.luminance_steady_state
                        ),  # t = kdur+1 .. kdur+num_frames
                    ]
                )
                # Delayed step (subtract after num_frames)
                step_off = numpy.concatenate(
                    [
                        numpy.zeros(num_frames),  # delay
                        cell.luminance_step_response[
                            1:kdur
                        ],  # same shape as step_on's tail
                    ]
                )
                added_luminance = (
                    step_on[: num_frames + kdur - 1] - step_off[: num_frames + kdur - 1]
                )

                # Pad to full kernel_response length (last sample stays zero)
                if len(added_luminance) < len(cell.kernel_response.luminance):
                    added_luminance = numpy.pad(
                        added_luminance,
                        (0, len(cell.kernel_response.luminance) - len(added_luminance)),
                    )
                cell.kernel_response.luminance += added_luminance
                input_currents[rf_type].append(
                    cell.response_current(cell.kernel_response)
                )

        return input_currents

    def provide_null_input(self, visual_space, duration=None, offset=0):
        r"""
        This function exists for optimization purposes. It is the analog to
        :func:`process_input` for the special case when a blank stimulus is
        shown.

        Parameters
        ----------

        visual_space : VisualSpace
            The visual space to which the blank stimulus are presented.

        duration : int (ms)
            The time for which we will simulate the blank stimulus

        offset : int(ms)
            The time (in absolute time of the whole simulation) at which the stimulus starts.

        Returns
        -------

        None
            The generated currents are injected into the simulated thalamic
            relay neurons directly.

        Notes
        -----
        In normal mode, blank input is represented as a finite background
        luminance step and goes through the same current-injection machinery as
        explicit stimuli. In ``original_2024_lgn_mode``, this dispatches to the
        historical optimized null path to reproduce the 2024 reference data.
        Once ``original_2024_lgn_mode`` is no longer needed, it can be merged
        with ``calculate_null_input``.
        """
        if self.parameters.original_2024_lgn_mode:
            self._provide_legacy_null_input(visual_space, duration, offset)
        else:
            self.inject_currents(self.calculate_null_input(duration), duration, offset)

    def calculate_kernel_responses(self, visual_space, duration):
        r"""
        Convolve all local receptive-field cells with the current visual input.

        Parameters
        ----------
        visual_space : VisualSpace
            Visual space containing the stimulus objects to sample.

        duration : float (ms)
            Duration over which frames should be sampled. If None, the maximum
            duration of objects in ``visual_space`` is used.

        Returns
        -------
        tuple
            ``(kernel_responses, retinal_input)`` where ``kernel_responses`` is
            an ``OrderedDict`` mapping RF type to a list of ``KernelResponse``
            objects, one per local simulated thalamic relay neuron.
            ``retinal_input`` contains stored stimulus frames when
            ``store_stimuli`` is enabled; otherwise it contains ``None``
            entries matching the frame updates.
        """
        assert isinstance(visual_space, VisualSpace)
        if duration is None:
            duration = visual_space.get_maximum_duration()

        logger.debug("Processing frames")

        t = 0
        retinal_input = []

        for rf_type in self.rf_types:
            for cell in self.input_cells[rf_type]:
                cell.initialize(duration)

        while t < duration:
            t = visual_space.update()
            for rf_type in self.rf_types:
                for cell in self.input_cells[rf_type]:
                    cell.view()

            if self.model.parameters.store_stimuli == True:
                visual_region = VisualRegion(
                    location_x=0,
                    location_y=0,
                    size_x=self.model.visual_field.size_x,
                    size_y=self.model.visual_field.size_y,
                )
                im = visual_space.view(
                    visual_region, pixel_size=self.visual_space_resolution_deg
                )
            else:
                im = None
            retinal_input.append(im)

        kernel_responses = OrderedDict()
        for rf_type in self.rf_types:
            kernel_responses[rf_type] = [
                cell.kernel_response for cell in self.input_cells[rf_type]
            ]
        return (kernel_responses, retinal_input)

    def _calculate_input_currents(self, kernel_responses=None):
        r"""
        Calculate the input currents for all cells.

        Parameters
        ----------
        kernel_responses : OrderedDict
            Mapping from RF type to the ``KernelResponse`` objects produced by
            ``calculate_kernel_responses``.

        Returns
        -------
        OrderedDict
            Mapping from RF type to current dictionaries suitable for
            ``inject_currents``.
        """
        input_currents = OrderedDict()
        for rf_type in kernel_responses.keys():
            input_currents[rf_type] = [
                cell.response_current(kernel_response)
                for cell, kernel_response in zip(
                    self.input_cells[rf_type], kernel_responses[rf_type]
                )
            ]
        return input_currents


class EccentricityDependentSpatioTemporalFilterRetinaLGN(SensoryInputComponent):
    """Fixed-count eccentricity-dependent LGN population.

    Global ON/OFF positions are sampled independently, while each locally
    owned relay cell receives a complete Cai97 kernel whose spatial scale is
    derived from its eccentricity and whose temporal scale is sampled from a
    truncated log-normal distribution.
    """

    required_parameters = ParameterSet(
        {
            "number_per_polarity": int,
            "minimum_samples_per_center_sigma": float,
            "topography": ParameterSet(
                {
                    "cap_eccentricity": (float, type(None)),
                    "full_max_eccentricity": float,
                    "beta": float,
                }
            ),
            "temporal_scale_distribution": ParameterSet(
                {
                    "mu": float,
                    "sigma": float,
                    "lower_quantile": float,
                    "upper_quantile": float,
                }
            ),
            "linear_scaler": float,
            "mpi_reproducible_noise": bool,
            "recorders": ParameterSet,
            "recording_interval": float,
            "receptive_field": ParameterSet(
                {
                    "func": str,
                    "func_params": ParameterSet,
                    "width": float,
                    "height": float,
                    "temporal_resolution": float,
                    "duration": float,
                }
            ),
            "original_2024_lgn_mode": bool,
            "cell": ParameterSet(
                {
                    "model": str,
                    "params": ParameterSet,
                    "receptors": ParameterSet,
                    "native_nest": bool,
                    "initial_values": ParameterSet,
                }
            ),
            "gain_control": {
                "gain": float,
                "non_linear_gain": ParameterSet(
                    {
                        "luminance_gain_ON": float,
                        "luminance_scaler_ON": float,
                        "luminance_gain_OFF": float,
                        "luminance_scaler_OFF": float,
                        "contrast_gain": float,
                        "contrast_scaler": float,
                    }
                ),
            },
            "noise": ParameterSet(
                {
                    "X_ON": ParameterSet(
                        {
                            "mean": float,
                            "stdev": float,
                        }
                    ),
                    "X_OFF": ParameterSet(
                        {
                            "mean": float,
                            "stdev": float,
                        }
                    ),
                }
            ),
        }
    )

    def __init__(self, model, parameters):
        SensoryInputComponent.__init__(self, model, parameters)
        self._validate_stage_2_parameters()
        self._validate_receptive_field_parameters()
        if not hasattr(model, "visual_field"):
            raise ValueError(
                "model.visual_field must exist before constructing eccentricity "
                "LGN input; eccentricity mode currently requires a "
                "fixation-centred visual rectangle"
            )

        self.topography = RadiallySymmetricLGNTopography(
            model.visual_field, self.parameters.topography
        )
        self.rf_types = ("X_ON", "X_OFF")
        self.sheets = OrderedDict()
        self.pops = OrderedDict()
        self.ncs_rng = OrderedDict()
        self.internal_stimulus_cache = OrderedDict()

        if self.parameters.cell.model[-6:] == "_sc_nc":
            self.integrated_cs = True
        else:
            self.integrated_cs = False
            self.scs = OrderedDict()
            self.ncs = OrderedDict()

        position_seeds = mozaik.get_seeds(2)
        positions_by_type = OrderedDict()
        for index, rf_type in enumerate(self.rf_types):
            positions_by_type[rf_type] = _sample_lgn_positions(
                self.topography,
                self.parameters.number_per_polarity,
                numpy.random.RandomState(seed=position_seeds[index]),
            )

        for rf_type in self.rf_types:
            cell = copy.deepcopy(self.parameters.cell)
            if self.integrated_cs:
                noise = self._noise_parameters(rf_type)
                cell.params.update(
                    [
                        ("mean", noise.mean * 1000),
                        ("std", noise.stdev * 1000),
                        ("dt", self.model.sim.get_time_step()),
                    ]
                )
            self.sheets[rf_type] = RetinalInhomogeneousDiskSheet(
                model,
                ParameterSet(
                    {
                        "cell": cell,
                        "name": rf_type,
                        "artificial_stimulators": OrderedDict(),
                        "recorders": self.parameters.recorders,
                        "recording_interval": self.parameters.recording_interval,
                        "mpi_safe": False,
                    }
                ),
                positions_by_type[rf_type],
                self.topography,
            )

        self._initialize_noise_sources()
        self._initialize_receptive_fields()

    def _validate_stage_2_parameters(self):
        number = self.parameters.number_per_polarity
        if (
            isinstance(number, bool)
            or not isinstance(number, numbers.Integral)
            or number <= 0
        ):
            raise ValueError("number_per_polarity must be an integer greater than zero")
        if self.parameters.original_2024_lgn_mode:
            raise ValueError(
                "original_2024_lgn_mode=True is incompatible with "
                "eccentricity-dependent LGN input"
            )
        self._validate_noise_parameters()
        self._validate_gain_control_parameters()
        self._validate_temporal_scale_distribution()

        positive_finite_parameters = (
            (
                "minimum_samples_per_center_sigma",
                self.parameters.minimum_samples_per_center_sigma,
            ),
            ("receptive_field.width", self.parameters.receptive_field.width),
            ("receptive_field.height", self.parameters.receptive_field.height),
            (
                "receptive_field.temporal_resolution",
                self.parameters.receptive_field.temporal_resolution,
            ),
            ("receptive_field.duration", self.parameters.receptive_field.duration),
        )
        for name, value in positive_finite_parameters:
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not numpy.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")

    def _validate_temporal_scale_distribution(self):
        distribution = self.parameters.temporal_scale_distribution
        for name in ("mu", "sigma", "lower_quantile", "upper_quantile"):
            value = distribution[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not numpy.isfinite(value)
            ):
                raise ValueError(
                    f"temporal_scale_distribution.{name} must be finite"
                )
        if distribution.sigma <= 0.0:
            raise ValueError(
                "temporal_scale_distribution.sigma must be positive and finite"
            )
        if not (
            0.0
            < distribution.lower_quantile
            < distribution.upper_quantile
            < 1.0
        ):
            raise ValueError(
                "temporal_scale_distribution quantiles must satisfy "
                "0 < lower_quantile < upper_quantile < 1"
            )

    def _sample_temporal_scales(self):
        try:
            pynn_seed = self.model.parameters.pynn_seed
        except AttributeError as exc:
            raise ValueError(
                "model.parameters.pynn_seed is required for temporal-scale sampling"
            ) from exc

        distribution = self.parameters.temporal_scale_distribution
        rngs = _temporal_scale_rngs(pynn_seed)
        return OrderedDict(
            (
                rf_type,
                _sample_truncated_lognormal_scales(
                    self.parameters.number_per_polarity,
                    distribution.mu,
                    distribution.sigma,
                    distribution.lower_quantile,
                    distribution.upper_quantile,
                    rng,
                ),
            )
            for rf_type, rng in zip(self.rf_types, rngs)
        )

    def _validate_noise_parameters(self):
        for rf_type in ("X_ON", "X_OFF"):
            pair = self.parameters.noise[rf_type]
            for parameter_name in ("mean", "stdev"):
                value = pair[parameter_name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, numbers.Real)
                    or not numpy.isfinite(value)
                ):
                    raise ValueError(
                        f"noise.{rf_type}.{parameter_name} must be finite"
                    )
            if pair.stdev < 0.0:
                raise ValueError(f"noise.{rf_type}.stdev must be nonnegative")

    def _validate_gain_control_parameters(self):
        nonlinear = self.parameters.gain_control.non_linear_gain
        for name, value in nonlinear.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not numpy.isfinite(value)
            ):
                raise ValueError(
                    f"gain_control.non_linear_gain.{name} must be finite"
                )
            if "scaler" in name and value <= 0.0:
                raise ValueError(
                    f"gain_control.non_linear_gain.{name} must be positive"
                )

    def _gain_control_parameters(self, rf_type):
        gain_control = copy.deepcopy(self.parameters.gain_control)
        nonlinear = gain_control.non_linear_gain
        suffix = "ON" if rf_type == "X_ON" else "OFF"
        nonlinear["luminance_gain"] = nonlinear[f"luminance_gain_{suffix}"]
        nonlinear["luminance_scaler"] = nonlinear[f"luminance_scaler_{suffix}"]
        for polarity in ("ON", "OFF"):
            del nonlinear[f"luminance_gain_{polarity}"]
            del nonlinear[f"luminance_scaler_{polarity}"]
        return gain_control

    def _noise_parameters(self, rf_type):
        return self.parameters.noise[rf_type]

    def _validate_receptive_field_parameters(self):
        receptive_field = self.parameters.receptive_field
        try:
            rf_function = eval(receptive_field.func)
        except (AttributeError, NameError, SyntaxError, TypeError) as exc:
            raise ValueError(
                "eccentricity-dependent LGN input requires "
                "receptive_field.func to resolve to cai97.stRF_2d"
            ) from exc
        if rf_function is not cai97.stRF_2d:
            raise ValueError(
                "eccentricity-dependent LGN input supports only cai97.stRF_2d"
            )

        function_parameters = receptive_field.func_params
        missing = [
            name
            for name in (*_CAI97_RF_NUMERIC_PARAMETERS, "subtract_mean")
            if name not in function_parameters
        ]
        if missing:
            raise ValueError(
                "Cai97 receptive_field.func_params is missing required "
                f"parameters: {missing}"
            )
        for name in _CAI97_RF_NUMERIC_PARAMETERS:
            value = function_parameters[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not numpy.isfinite(value)
            ):
                raise ValueError(f"receptive_field.func_params.{name} must be finite")
        for name in ("sigma_c", "sigma_s"):
            if function_parameters[name] <= 0.0:
                raise ValueError(
                    f"receptive_field.func_params.{name} must be positive and finite"
                )
        if function_parameters.subtract_mean is not False:
            raise ValueError(
                "eccentricity-dependent LGN input requires "
                "receptive_field.func_params.subtract_mean=False"
            )

    def _initialize_noise_sources(self):
        sim = self.model.sim
        for rf_type in self.rf_types:
            sheet = self.sheets[rf_type]
            self.ncs_rng[rf_type] = []
            seeds = mozaik.get_seeds((sheet.pop.size,))

            if self.integrated_cs:
                for index, _ in enumerate(sheet.pop.all_cells):
                    if sheet.pop._mask_local[index]:
                        self.ncs_rng[rf_type].append(
                            numpy.random.RandomState(seed=seeds[index])
                        )
                continue

            self.scs[rf_type] = []
            self.ncs[rf_type] = []
            for index, lgn_cell in enumerate(sheet.pop.all_cells):
                step_source = sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])
                if self.parameters.mpi_reproducible_noise:
                    noise_source = sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])
                else:
                    noise_source = sim.NoisyCurrentSource(
                        **self._noise_parameters(rf_type)
                    )

                if sheet.pop._mask_local[index]:
                    self.ncs_rng[rf_type].append(
                        numpy.random.RandomState(seed=seeds[index])
                    )
                    self.scs[rf_type].append(step_source)
                    self.ncs[rf_type].append(noise_source)
                lgn_cell.inject(step_source)
                lgn_cell.inject(noise_source)

    def _initialize_receptive_fields(self):
        receptive_field = self.parameters.receptive_field
        function_parameters = receptive_field.func_params
        reference_center_sigma = float(function_parameters.sigma_c)
        reference_surround_sigma = float(function_parameters.sigma_s)
        temporal_scales_by_type = self._sample_temporal_scales()

        rf_parameters = OrderedDict()
        all_center_sigmas = []
        all_eccentricities = []
        for rf_type in self.rf_types:
            positions = self.sheets[rf_type].canonical_positions_deg
            eccentricities = numpy.hypot(positions[0], positions[1])
            center_sigmas = numpy.asarray(
                self.topography.center_sigma_deg(eccentricities), dtype=float
            )
            scales = center_sigmas / reference_center_sigma
            temporal_scales = temporal_scales_by_type[rf_type]
            durations_ms = receptive_field.duration * temporal_scales
            temporal_samples = numpy.ceil(
                durations_ms / receptive_field.temporal_resolution
            ).astype(int)
            parameters = {
                "eccentricities_deg": eccentricities,
                "center_sigmas_deg": center_sigmas,
                "surround_sigmas_deg": scales * reference_surround_sigma,
                "scales": scales,
                "widths_deg": scales * receptive_field.width,
                "heights_deg": scales * receptive_field.height,
                "temporal_scales": temporal_scales,
                "durations_ms": durations_ms,
                "temporal_samples": temporal_samples,
            }
            for values in parameters.values():
                values.setflags(write=False)
            rf_parameters[rf_type] = parameters
            all_center_sigmas.append(center_sigmas)
            all_eccentricities.append(eccentricities)
        self._rf_parameters = rf_parameters

        minimum_center_sigma = min(
            float(numpy.min(values)) for values in all_center_sigmas
        )
        raw_resolution = (
            minimum_center_sigma / self.parameters.minimum_samples_per_center_sigma
        )
        self.visual_space_resolution_deg = _round_down_to_two_significant_digits(
            raw_resolution
        )
        for center_sigmas in all_center_sigmas:
            samples_per_sigma = center_sigmas / self.visual_space_resolution_deg
            if numpy.any(
                samples_per_sigma + 1e-12
                < self.parameters.minimum_samples_per_center_sigma
            ):
                raise AssertionError(
                    "derived visual-space resolution violates the configured "
                    "minimum samples per centre sigma"
                )

        global_shapes = []
        estimated_dense_local_bytes = 0
        estimated_factorized_local_bytes = 0
        for rf_type in self.rf_types:
            parameters = rf_parameters[rf_type]
            horizontal_samples = numpy.ceil(
                parameters["widths_deg"] / self.visual_space_resolution_deg
            ).astype(int)
            vertical_samples = numpy.ceil(
                parameters["heights_deg"] / self.visual_space_resolution_deg
            ).astype(int)
            shapes = numpy.column_stack(
                (
                    vertical_samples,
                    horizontal_samples,
                    parameters["temporal_samples"],
                )
            )
            global_shapes.extend(map(tuple, shapes))
            local_mask = self.sheets[rf_type].pop._mask_local
            local_shapes = shapes[local_mask]
            estimated_dense_local_bytes += int(
                2
                * numpy.sum(numpy.prod(local_shapes, axis=1))
                * numpy.dtype(float).itemsize
            )
            # Cai97 currently uses two centre/surround factors. The storage
            # model remains general; this estimate describes that current use.
            estimated_factorized_local_bytes += int(
                numpy.sum(
                    2 * local_shapes[:, 0]
                    + 2 * local_shapes[:, 1]
                    + 5 * local_shapes[:, 2]
                    + 2
                )
                * numpy.dtype(float).itemsize
            )

        all_surround_sigmas = [
            values["surround_sigmas_deg"] for values in rf_parameters.values()
        ]
        global_counts = {
            rf_type: int(self.sheets[rf_type].pop.size) for rf_type in self.rf_types
        }
        local_counts = {
            rf_type: int(numpy.count_nonzero(self.sheets[rf_type].pop._mask_local))
            for rf_type in self.rf_types
        }
        logger.info(
            "Eccentricity LGN RF plan: E_max=%g deg; user cap=%s; "
            "resolved mapping cap=%g deg; eccentricity range=[%g, %g] "
            "deg; centre sigma range=[%g, %g] deg; surround sigma "
            "range=[%g, %g] deg; resolution=%g deg/pixel; kernel shape "
            "range=%s..%s; estimated local RF/operator array memory=%.3f MiB "
            "(dense reference %.3f MiB); global counts=%s; local counts=%s",
            self.topography.max_eccentricity_deg,
            self.topography.user_cap_eccentricity_deg,
            self.topography._resolved_mapping_cap_eccentricity_deg,
            min(float(numpy.min(values)) for values in all_eccentricities),
            max(float(numpy.max(values)) for values in all_eccentricities),
            min(float(numpy.min(values)) for values in all_center_sigmas),
            max(float(numpy.max(values)) for values in all_center_sigmas),
            min(float(numpy.min(values)) for values in all_surround_sigmas),
            max(float(numpy.max(values)) for values in all_surround_sigmas),
            self.visual_space_resolution_deg,
            min(global_shapes),
            max(global_shapes),
            estimated_factorized_local_bytes / 2**20,
            estimated_dense_local_bytes / 2**20,
            global_counts,
            local_counts,
        )

        self.input_cells = OrderedDict()
        rf_function = cai97.stRF_2d
        for rf_type in self.rf_types:
            self.input_cells[rf_type] = []
            parameters = rf_parameters[rf_type]
            positions = self.sheets[rf_type].canonical_positions_deg
            gain_control = self._gain_control_parameters(rf_type)
            for global_index in numpy.flatnonzero(self.sheets[rf_type].pop._mask_local):
                per_cell_parameters = copy.deepcopy(function_parameters)
                per_cell_parameters.sigma_c = float(
                    parameters["center_sigmas_deg"][global_index]
                )
                per_cell_parameters.sigma_s = float(
                    parameters["surround_sigmas_deg"][global_index]
                )
                temporal_scale = float(parameters["temporal_scales"][global_index])
                for name in ("t1", "t2", "td"):
                    per_cell_parameters[name] *= temporal_scale
                for name in ("c1", "c2"):
                    per_cell_parameters[name] /= temporal_scale
                receptive_field_for_cell = SpatioTemporalReceptiveField(
                    rf_function,
                    per_cell_parameters,
                    parameters["widths_deg"][global_index],
                    parameters["heights_deg"][global_index],
                    parameters["durations_ms"][global_index],
                    polarity=1.0 if rf_type == "X_ON" else -1.0,
                )
                receptive_field_for_cell.quantize(
                    self.visual_space_resolution_deg,
                    self.visual_space_resolution_deg,
                    receptive_field.temporal_resolution,
                    use_factorization=True,
                )
                expected_shape = (
                    int(
                        numpy.ceil(
                            parameters["heights_deg"][global_index]
                            / self.visual_space_resolution_deg
                        )
                    ),
                    int(
                        numpy.ceil(
                            parameters["widths_deg"][global_index]
                            / self.visual_space_resolution_deg
                        )
                    ),
                    int(parameters["temporal_samples"][global_index]),
                )
                if receptive_field_for_cell.shape != expected_shape:
                    raise AssertionError(
                        "per-cell RF quantization did not produce the "
                        "documented ceil-derived kernel shape"
                    )
                self.input_cells[rf_type].append(
                    EccentricityDependentCellWithReceptiveField(
                        positions[0, global_index],
                        positions[1, global_index],
                        receptive_field_for_cell,
                        gain_control,
                        self.model.input_space,
                        False,
                    )
                )

    def process_input(self, visual_space, stimulus, duration=None, offset=0):
        return SpatioTemporalFilterRetinaLGN.process_input(
            self, visual_space, stimulus, duration, offset
        )

    def inject_currents(self, input_currents, duration=None, offset=0):
        return SpatioTemporalFilterRetinaLGN.inject_currents(
            self, input_currents, duration, offset
        )

    def calculate_null_input(self, duration=None):
        return SpatioTemporalFilterRetinaLGN.calculate_null_input(self, duration)

    def provide_null_input(self, visual_space, duration=None, offset=0):
        return SpatioTemporalFilterRetinaLGN.provide_null_input(
            self, visual_space, duration, offset
        )

    def calculate_kernel_responses(self, visual_space, duration):
        return SpatioTemporalFilterRetinaLGN.calculate_kernel_responses(
            self, visual_space, duration
        )

    def _calculate_input_currents(self, kernel_responses=None):
        return SpatioTemporalFilterRetinaLGN._calculate_input_currents(
            self, kernel_responses
        )
