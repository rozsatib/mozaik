from mozaik.space import VisualSpace
from mozaik.models.vision.spatiotemporalfilter import (
    CellWithReceptiveField,
    SpatioTemporalReceptiveField,
)
from mozaik.models.vision import cai97
from parameters import ParameterSet
import numpy as np
from mozaik.stimuli.vision.topographica_based import PixelImpulse
from mozaik.tools.mozaik_parametrized import SNumber
from quantities import dimensionless
import imagen
import pytest
import pylab

class TestCellWithReceptiveField:

    receptive_field_on = None
    receptive_field_off = None
    cell_on = None
    cell_off = None
    visual_space = None

    # Visual stimulus parameters
    vs_params = {
        "frame_duration": 1.0,
        "duration": 1,
        "trial": 1,
        "direct_stimulation_name": "simulation_name",
        "direct_stimulation_parameters": None,
        "background_luminance": 50.0,
        "density": 10.0,
        "location_x": 0.0,
        "location_y": 0.0,
        "size_x": 3.0,
        "size_y": 3.0,
    }

    # Receptive field parameters
    rf_params = {
        "Ac": 1.0,
        "As": 0.15,  # changed from 1/3.0
        "K1": 1.05,
        "K2": 0.7,
        "c1": 0.14,
        "c2": 0.12,
        "n1": 7.0,
        "n2": 8.0,
        "t1": -6.0,  # ms
        "t2": -6.0,  # ms
        "td": 6.0,  # time differece between ON-OFF
        "sigma_c": 0.4,  # 0.4, # Allen 2006 # 1.15 # sigma of center gauss degree
        "sigma_s": 1.0,  # 1.0, # sigma_c*1.5+0.4 Allen 2006 # 2.95 # sigma of surround gauss degree
        "subtract_mean": False,
    }

    @classmethod
    def setup_class(cls):
        cls.visual_space = VisualSpace(
            ParameterSet(
                {
                    "update_interval": cls.vs_params["frame_duration"],
                    "background_luminance": cls.vs_params["background_luminance"],
                }
            )
        )
        cls.receptive_field_on = SpatioTemporalReceptiveField(
            cai97.stRF_2d, ParameterSet(cls.rf_params), 3.0, 3.0, 200
        )
        cls.receptive_field_on.quantize(0.1, 0.1, 1)
        cls.receptive_field_off = SpatioTemporalReceptiveField(
            lambda x, y, t, p: -1.0 * cai97.stRF_2d(x, y, t, p),
            ParameterSet(cls.rf_params),
            3.0,
            3.0,
            200,
        )
        cls.receptive_field_off.quantize(0.1, 0.1, 1)
        gain_params = ParameterSet({"gain": 1.0, "non_linear_gain": None})
        cls.cell_on = CellWithReceptiveField(
            0, 0, cls.receptive_field_on, gain_params, cls.visual_space
        )
        cls.cell_off = CellWithReceptiveField(
            0, 0, cls.receptive_field_off, gain_params, cls.visual_space
        )

    #@pytest.mark.parametrize("x", range(30))
    #@pytest.mark.parametrize("y", range(30))
    #@pytest.mark.parametrize("on", [True,False])
    def est_impulse_response(self, x, y, on):
        stimulus = PixelImpulse(relative_luminance=1.0, x=x, y=y, **self.vs_params)
        self.visual_space.clear()
        self.visual_space.add_object(str(stimulus), stimulus)
        self.visual_space.update()
        # Impulse is 1st frame, 4 frames null stimulus as sanity check
        if on:
            cell = self.cell_on
            rf = self.receptive_field_on
        else:
            cell = self.cell_off
            rf = self.receptive_field_off
        cell.initialize(self.vs_params["background_luminance"], 5)
        cell.view()
        r = cell.response[: rf.kernel.shape[2]]
        np.testing.assert_equal(r, rf.kernel[x, y, :])

    @pytest.mark.parametrize("x", [5])
    @pytest.mark.parametrize("y", [9])
    @pytest.mark.parametrize("on", [True])
    def test_cell_current(self, x, y, on):
        stimulus = PixelImpulse(relative_luminance=1.0, x=x, y=y, **self.vs_params)
        self.visual_space.clear()
        self.visual_space.add_object(str(stimulus), stimulus)
        self.visual_space.update()
        # Impulse is 1st frame, 4 frames null stimulus as sanity check
        if on:
            cell = self.cell_on
            rf = self.receptive_field_on
        else:
            cell = self.cell_off
            rf = self.receptive_field_off
        cell.initialize(self.vs_params["background_luminance"], 500)
        cell.view()
        r = cell.response_current()
        pylab.plot(r)
        pylab.savefig("bla.png")
