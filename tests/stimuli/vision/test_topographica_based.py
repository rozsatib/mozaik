# -*- coding: utf-8 -*-
"""
This module contains classes with tests for mozaik/stimuli/vision/topographica_based.py
"""

from mozaik.stimuli.vision.visual_stimulus import VisualStimulus
import imagen
import imagen.random
from imagen.transferfn import TransferFn
import param
from imagen.image import BoundingBox
import pickle
import numpy
import numpy as np
from mozaik.tools.mozaik_parametrized import SNumber, SString
from mozaik.tools.units import cpd
from numpy import pi
from quantities import Hz, rad, degrees, ms, dimensionless
import mozaik.stimuli.vision.topographica_based as topo
from collections import namedtuple

import pytest

# Dummy class to get rid of NotImplementedError
class DummyTBVS(topo.TopographicaBasedVisualStimulus):
    def frames(self):
        return []


class TestTopographicaBasedVisualStimulus:
    def test_nontransparent(self):
        """
        Topographica stimuli do not handle transparency.
        """
        t = DummyTBVS(size_x=1, size_y=1, location_x=0.0, location_y=0.0)
        assert not t.transparent


default_topo = {
    "duration": 100,
    "frame_duration": 1,
    "background_luminance": 50.0,
    "density": 10.0,
    "location_x": 0.0,
    "location_y": 0.0,
    "size_x": 11.0,
    "size_y": 11.0,
}


class TopographicaBasedVisualStimulusTester(object):
    """
    Parent class containing convenience functions for testing stimulus generators in
    mozaik/stimuli/vision/topographica_based.py.
    """

    # Number of frames to test
    num_frames = 100
    # Default parameters for generating TopographicaBasedVisualStimuli frames

    def reference_frames(self, **params):
        """
        Generate or read frames to compare the output of TopographicaBasedVisualStimulus
        (or its child class) frames to. Parameters can be variable depending on child class
        and fixtures used.

        Returns
        -------
        Generator: tuple(numpy.array : frame, list : optional parameter(s), e.g. orientation))
        """
        raise NotImplementedError("Must be implemented by child class.")

    def topo_frames(self, **params):
        """
        Generate frames using TopographicaBasedVisualStimulus (or its child class) functions.

        Returns
        -------
        Generator: tuple(numpy.array : frame, list : optional parameter(s), e.g. orientation))
        """
        raise NotImplementedError("Must be implemented by child class.")

    def check_frames(self, **params):
        """
        Generate reference and TopographicaBased frames and check their equality.
        Check equivalence of reference frames and frames generated by
        topographica_based.py functions.
        """
        rf = self.reference_frames(**params)
        tf = self.topo_frames(**params)
        assert self.frames_equal(rf, tf, self.num_frames)

    def frames_equal(self, g0, g1, num_frames):
        """
        Checks if the first num_frames frames of the two generators are identical.
        """
        for i in range(num_frames):
            f0 = g0.next()
            f1 = g1.next()
            if not (numpy.array_equal(f0[0], f1[0]) and f0[1] == f1[1]):
                return False
        return True

    def test_frames(self, **params):
        """
        Function with a name that pytest will recognize. Use it to call compare_frames
        with the parameters passed from pytest.mark.parameterize to generate tests with
        unique names for each parameter combination. Must be implemented by child class.
        """
        pytest.skip("Must be implemented in child class and call check_frames.")


default_noise = {"grid_size": 11, "grid": False, "time_per_image": 2}


class TestNoise(TopographicaBasedVisualStimulusTester):

    experiment_seed = 0

    @pytest.mark.parametrize("grid_size", [-1, 0, 0.9, 0.9999999999])
    def test_min_grid_size(self, grid_size):
        if type(self) == TestNoise:
            pytest.skip("Only run in child classes.")
        with pytest.raises(ValueError):
            self.check_frames(grid_size=grid_size)


# grid_size, size_x, grid, background_luminance, density
sparse_noise_params = [
    # Some basic parameter combinations
    (10, 10, True, 50, 5.0),
    (15, 15, False, 60, 6.0),
    (5, 5, False, 0.0, 15),
]


class TestSparseNoise(TestNoise):
    """
    Tests for the SparseNoise class.
    """

    def test_init_assert(self):
        """
        Checks the assertion in the init function of the class.
        """
        with pytest.raises(AssertionError):
            t = topo.SparseNoise(time_per_image=1.4, frame_duration=1.5)

    def reference_frames(
        self,
        grid_size=default_noise["grid_size"],
        size_x=default_topo["size_x"],
        grid=default_noise["grid"],
        background_luminance=default_topo["background_luminance"],
        density=default_topo["density"],
    ):

        time_per_image = default_noise["time_per_image"]
        frame_duration = default_topo["frame_duration"]
        aux = imagen.random.SparseNoise(
            grid_density=grid_size * 1.0 / size_x,
            grid=grid,
            offset=0,
            scale=2 * background_luminance,
            bounds=BoundingBox(radius=size_x / 2),
            xdensity=density,
            ydensity=density,
            random_generator=numpy.random.RandomState(seed=self.experiment_seed),
        )
        while True:
            aux2 = aux()
            for i in range(time_per_image / frame_duration):
                yield (aux2, [0])

    def topo_frames(
        self,
        grid_size=default_noise["grid_size"],
        size_x=default_topo["size_x"],
        grid=default_noise["grid"],
        background_luminance=default_topo["background_luminance"],
        density=default_topo["density"],
    ):
        snclass = topo.SparseNoise(
            grid_size=grid_size,
            grid=grid,
            background_luminance=background_luminance,
            density=density,
            size_x=size_x,
            size_y=default_topo["size_y"],
            location_x=default_topo["location_x"],
            location_y=default_topo["location_y"],
            time_per_image=default_noise["time_per_image"],
            frame_duration=default_topo["frame_duration"],
            experiment_seed=self.experiment_seed,
        )
        return snclass._frames

    @pytest.mark.parametrize(
        "grid_size, size_x, grid, background_luminance, density", sparse_noise_params
    )
    def test_frames(self, grid_size, size_x, grid, background_luminance, density):
        self.check_frames(
            grid_size=grid_size,
            size_x=size_x,
            grid=grid,
            background_luminance=background_luminance,
            density=density,
        )


# grid_size, size_x, background_luminance, density
dense_noise_params = [
    # Some basic parameter combinations
    (10, 10, 50, 5.0),
    (15, 15, 60, 6.0),
    (5, 5, 0.0, 15),
]


class TestDenseNoise(TestNoise):
    """
    Tests for the DenseNoise class.
    """

    def test_init_assert(self):
        """
        Checks the assertion in the init function of the class.
        """
        with pytest.raises(AssertionError):
            t = topo.DenseNoise(time_per_image=1.4, frame_duration=1.5)

    def reference_frames(
        self,
        grid_size=default_noise["grid_size"],
        size_x=default_topo["size_x"],
        background_luminance=default_topo["background_luminance"],
        density=default_topo["density"],
    ):
        time_per_image = default_noise["time_per_image"]
        frame_duration = default_topo["frame_duration"]
        aux = imagen.random.DenseNoise(
            grid_density=grid_size * 1.0 / size_x,
            offset=0,
            scale=2 * background_luminance,
            bounds=BoundingBox(radius=size_x / 2),
            xdensity=density,
            ydensity=density,
            random_generator=numpy.random.RandomState(seed=self.experiment_seed),
        )
        while True:
            aux2 = aux()
            for i in range(time_per_image / frame_duration):
                yield (aux2, [0])

    def topo_frames(
        self,
        grid_size=default_noise["grid_size"],
        size_x=default_topo["size_x"],
        background_luminance=default_topo["background_luminance"],
        density=default_topo["density"],
    ):
        snclass = topo.DenseNoise(
            grid_size=grid_size,
            grid=False,
            background_luminance=background_luminance,
            density=density,
            size_x=size_x,
            size_y=default_topo["size_y"],
            location_x=default_topo["location_x"],
            location_y=default_topo["location_y"],
            time_per_image=default_noise["time_per_image"],
            frame_duration=default_topo["frame_duration"],
            experiment_seed=self.experiment_seed,
        )
        return snclass._frames

    @pytest.mark.parametrize(
        "grid_size, size_x, background_luminance, density", dense_noise_params
    )
    def test_frames(self, grid_size, size_x, background_luminance, density):
        self.check_frames(
            grid_size=grid_size,
            size_x=size_x,
            background_luminance=background_luminance,
            density=density,
        )


class TestFullfieldDriftingSinusoidalGrating(TopographicaBasedVisualStimulusTester):
    pass


class TestFullfieldDriftingSquareGrating(TopographicaBasedVisualStimulusTester):
    pass


class TestFullfieldDriftingSinusoidalGratingA(TopographicaBasedVisualStimulusTester):
    pass


class TestFlashingSquares(TopographicaBasedVisualStimulusTester):
    pass


class TestNull(TopographicaBasedVisualStimulusTester):
    pass


class TestMaximumDynamicRange(TransferFn):
    pass


class TestNaturalImageWithEyeMovement(TopographicaBasedVisualStimulusTester):
    pass


class TestDriftingGratingWithEyeMovement(TopographicaBasedVisualStimulusTester):
    pass


class TestDriftingSinusoidalGratingDisk(TopographicaBasedVisualStimulusTester):
    pass


class TestFlatDisk(TopographicaBasedVisualStimulusTester):
    pass


class TestFlashedBar(TopographicaBasedVisualStimulusTester):
    pass


class TestDriftingSinusoidalGratingCenterSurroundStimulus(
    TopographicaBasedVisualStimulusTester
):
    pass


class TestDriftingSinusoidalGratingRing(TopographicaBasedVisualStimulusTester):
    pass


class TestFlashedInterruptedBar(TopographicaBasedVisualStimulusTester):
    pass


class TestFlashedInterruptedCorner(TopographicaBasedVisualStimulusTester):
    pass


class TestVonDerHeydtIllusoryBar(TopographicaBasedVisualStimulusTester):
    pass


class SimpleGaborPatch(TopographicaBasedVisualStimulusTester):
    pass


class TwoStrokeGaborPatch(TopographicaBasedVisualStimulusTester):
    pass


default_gabor = {
    "orientation": 0,
    "phase": 0,
    "spatial_frequency": 2,
    "sigma": 1.0 / 3.0,
    "flash_duration": 1,
    "relative_luminance": 1,
    "x": 0,
    "y": 0,
    "grid": False,
}


class TestGabor:
    saved_frames = dict()

    def get_stimulus():
        pass

    def get_frames(self, **kwargs):
        """
        Generate stimulus, pop frames from its generator and save them into a dictionary
        If the frames have already been used in some different test, just get them from
        the dictionary instead.
        """
        stim = self.get_stimulus(**kwargs)
        kwargs_t = tuple(kwargs.values())
        if kwargs_t in self.saved_frames:
            frames = self.saved_frames[kwargs_t]
        else:
            num_frames = (
                int(getattr(stim, "duration") / getattr(stim, "frame_duration")) + 2
            )
            frames = self.pop_frames(stim, num_frames)
            self.saved_frames[kwargs_t] = frames
        return frames

    def pop_frames(self, stimulus, num_frames):
        return [stimulus._frames.next()[0] for i in range(num_frames)]

    def get_nonblank_mask(self, frame, baseline=0):
        """
        Returns a boolean numpy array, that is false where the input frame is different
        from a specified baseline value.
        """
        base = np.ones(frame.shape) * baseline
        mask = np.isclose(frame, baseline)
        return mask


default_cont_mov_jump = {
    "movement_length": 4,
    "movement_angle": 0,
    "movement_duration": 10,
    "flash_duration": 4,
    "moving_gabor_orientation_radial": True,
}


class TestContinuousGaborMovementAndJump(TestGabor):
    num_tests = 100

    def generate_frame_params(length=1):
        """
        Generate random parameters for ContinuousGaborMovementAndJump class instances
        """
        np.random.seed(0)
        params = []
        for i in xrange(0, length):
            x = (np.random.rand() - 0.5) * default_topo["size_x"] / 2
            y = (np.random.rand() - 0.5) * default_topo["size_y"] / 2
            sigma = 1 / 3 + np.random.rand() * 4 / 3
            orientation = np.random.rand() * np.pi
            movement_length = np.random.rand() * 4
            movement_angle = np.random.rand() * 2 * np.pi
            movement_duration = (
                2 * default_topo["frame_duration"] + np.random.rand() * 30
            )
            flash_duration = default_topo["frame_duration"] + np.random.rand() * 5

            moving_gabor_orientation_radial = np.random.rand() > 0.5
            params.append(
                (
                    x,
                    y,
                    sigma,
                    orientation,
                    movement_length,
                    movement_angle,
                    movement_duration,
                    flash_duration,
                    moving_gabor_orientation_radial,
                )
            )
        return params

    def get_stimulus(
        self,
        x,
        y,
        sigma,
        orientation,
        movement_length,
        movement_angle,
        movement_duration,
        flash_duration,
        moving_gabor_orientation_radial,
    ):
        """
        Return a ContinuousGaborMovementAndJump stimulus with the specified parameters.
        """
        cgb = topo.ContinuousGaborMovementAndJump(
            duration=default_topo["duration"],
            frame_duration=default_topo["frame_duration"],
            background_luminance=default_topo["background_luminance"],
            density=default_topo["density"],
            location_x=default_topo["location_x"],
            location_y=default_topo["location_y"],
            size_x=default_topo["size_x"],
            size_y=default_topo["size_y"],
            orientation=orientation,
            phase=default_gabor["phase"],
            spatial_frequency=default_gabor["spatial_frequency"],
            sigma=sigma,
            movement_duration=movement_duration,
            center_flash_duration=flash_duration,
            moving_relative_luminance=default_gabor["relative_luminance"],
            center_relative_luminance=default_gabor["relative_luminance"],
            x=x,
            y=y,
            movement_length=movement_length,
            movement_angle=movement_angle,
            moving_gabor_orientation_radial=moving_gabor_orientation_radial,
        )
        return cgb

    @pytest.mark.parametrize(
        "x, y, sigma, orientation, movement_length, movement_angle, movement_duration, flash_duration, moving_gabor_orientation_radial",
        generate_frame_params(num_tests),
    )
    def test_no_overlap(
        self,
        x,
        y,
        sigma,
        orientation,
        movement_length,
        movement_angle,
        movement_duration,
        flash_duration,
        moving_gabor_orientation_radial,
    ):
        """
        Test that there is no overlap between the moving Gabor patch and the Gabor
        patch flashed at center
        """
        args = locals()
        args.pop("self")
        frames = self.get_frames(**args)

        movement_num_frames = int(movement_duration / default_topo["frame_duration"])
        movement_mask = numpy.full(frames[0].shape, True, dtype=bool)
        for i in range(movement_num_frames):
            movement_mask = np.logical_and(
                movement_mask,
                self.get_nonblank_mask(frames[i], default_topo["background_luminance"]),
            )

        flash_mask = numpy.full(frames[0].shape, True, dtype=bool)
        for i in range(movement_num_frames, len(frames)):
            flash_mask = np.logical_and(
                flash_mask,
                self.get_nonblank_mask(frames[i], default_topo["background_luminance"]),
            )
        overlap = np.logical_or(movement_mask, flash_mask)
        assert np.all(
            overlap
        ), "There is overlap between movement and center flash Gabors"

    @pytest.mark.parametrize(
        "x, y, sigma, orientation, movement_length, movement_angle, movement_duration, flash_duration, moving_gabor_orientation_radial",
        generate_frame_params(num_tests),
    )
    def test_is_blank_after_stimulus_end(
        self,
        x,
        y,
        sigma,
        orientation,
        movement_length,
        movement_angle,
        movement_duration,
        flash_duration,
        moving_gabor_orientation_radial,
    ):
        """
        Check if class returns blank frames after stimulus duration
        """
        args = locals()
        args.pop("self")
        frames = self.get_frames(**args)
        stimulus_duration = movement_duration + flash_duration
        stimulus_end_id = int(stimulus_duration / default_topo["frame_duration"])
        blank = np.full(
            frames[0].shape, default_topo["background_luminance"], dtype=np.float64
        )
        for i in range(stimulus_end_id, len(frames)):
            np.testing.assert_allclose(frames[i], blank)
