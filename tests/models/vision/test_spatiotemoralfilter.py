from mozaik.space import VisualSpace
from mozaik.models.vision.spatiotemporalfilter import CellWithReceptiveField, SpatioTemporalReceptiveField
from mozaik.models.vision import cai97
from parameters import ParameterSet
import numpy as np
from mozaik.stimuli.vision.topographica_based import TopographicaBasedVisualStimulus
from mozaik.tools.mozaik_parametrized import SNumber
from quantities import dimensionless
import imagen
import pytest

class PixelImpulse(TopographicaBasedVisualStimulus):
    relative_luminance = SNumber(dimensionless, doc="")
    x = SNumber(dimensionless, doc="x coordinate of pixel")
    y = SNumber(dimensionless, doc="y coordinate of pixel")
    def frames(self):
        blank = imagen.Constant(scale=self.background_luminance,
                              bounds=imagen.image.BoundingBox(radius=self.size_x/2),
                              xdensity=self.density,
                              ydensity=self.density)()
        impulse = blank.copy()
        impulse[self.x,self.y] *= 1 + self.relative_luminance
        yield (impulse, [self.frame_duration])
        while True:
            yield (impulse, [self.frame_duration])


class TestCellWithReceptiveField():

    receptive_field_kernel = None
    cell = None
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
            'Ac': 1.0,
            'As': 0.15, # changed from 1/3.0
            'K1': 1.05,
            'K2': 0.7,
            'c1': 0.14,
            'c2': 0.12,
            'n1': 7.0,
            'n2': 8.0,
            't1': -6.0, # ms
            't2': -6.0, # ms
            'td': 6.0, # time differece between ON-OFF
            'sigma_c': 0.4, #0.4, # Allen 2006 # 1.15 # sigma of center gauss degree
            'sigma_s': 1.0, #1.0, # sigma_c*1.5+0.4 Allen 2006 # 2.95 # sigma of surround gauss degree
            'subtract_mean': False,
    }

    @classmethod
    def setup_class(cls):
        cls.visual_space = VisualSpace(ParameterSet({"update_interval":cls.vs_params["frame_duration"],"background_luminance" : cls.vs_params["background_luminance"]}))
        cls.receptive_field = SpatioTemporalReceptiveField(cai97.stRF_2d,ParameterSet(cls.rf_params),3.0,3.0,200)
        cls.receptive_field.quantize(0.1,0.1,1)
        gain_params = ParameterSet({"gain" : 1.0, "non_linear_gain" : None})
        cls.cell = CellWithReceptiveField(0,0,cls.receptive_field,gain_params,cls.visual_space)

    @pytest.mark.parametrize("x", range(30))
    @pytest.mark.parametrize("y", range(30))
    def test_impulse_response(self,x,y):
        stimulus = PixelImpulse(relative_luminance=1.0,x=x,y=y,**self.vs_params)
        self.visual_space.clear()
        self.visual_space.add_object(str(stimulus),stimulus)
        self.visual_space.update()
        # Impulse is 1st frame, 4 frames null stimulus as sanity check
        self.cell.initialize(self.vs_params["background_luminance"],5)
        self.cell.view()
        r = self.cell.response[:self.receptive_field.kernel.shape[2]]
        np.testing.assert_equal(r, self.receptive_field.kernel[x,y,:])
