import numpy as np
from mozaik.experiments import Experiment
from parameters import ParameterSet
from mozaik.stimuli import InternalStimulus
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from collections import OrderedDict
from mozaik.sheets.direct_stimulator import OpticalStimulatorArrayChR
from copy import deepcopy


class CorticalStimulationWithOptogeneticArray(Experiment):
    """
    Parent class for optogenetic stimulation of cortical sheets with an array
    of light sources.

    Creates a array of optical stimulators covering an area of cortex, and then
    stimulates the array based on the stimulating_signal function in the 
    localstimulationarray_parameters.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    sheet_list : list
                The list of sheets in which to do stimulation.

    num_trials : int
                Number of trials each stimulus is shown.

    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                must be set by the specific experiments.
    """

    required_parameters = ParameterSet(
        {
            'sheet_list': list,
            'num_trials': int,
            'stimulator_array_parameters' : ParameterSet,
        }
    )

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        stimulator_array_keys = {
            "size",
            "spacing",
            "update_interval",
            "depth_sampling_step",
            "light_source_light_propagation_data",
        }
        assert self.parameters.stimulator_array_parameters.keys() == stimulator_array_keys, "Stimulator array keys must be: %s. Supplied: %s. Difference: %s" % (stimulator_array_keys,self.parameters.stimulator_array_parameters.keys(),set(stimulator_array_keys)^set(self.parameters.stimulator_array_parameters.keys()))

    def append_direct_stim(self, trial, model, stimulator_array_parameters):
        sap = MozaikExtendedParameterSet(deepcopy(stimulator_array_parameters))
        if self.direct_stimulation == None:
            self.direct_stimulation = []
            self.shared_scs = None

        d = OrderedDict()
        for sheet in self.parameters.sheet_list:
            d[sheet] = [OpticalStimulatorArrayChR(model.sheets[sheet],sap,self.shared_scs)]
            self.shared_scs = d[sheet][0].scs

        self.direct_stimulation.append(d)
        self.stimuli.append(
            InternalStimulus(
                frame_duration=sap.stimulating_signal_parameters.duration,
                duration=sap.stimulating_signal_parameters.duration,
                trial=trial,
                direct_stimulation_name="OpticalStimulatorArrayChR",
                direct_stimulation_parameters=sap,
            )
        )


class SingleOptogeneticArrayStimulus(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources.

    Creates a array of optical stimulators covering an area of cortex, and then
    stimulates the array based on the stimulating_signal function in the
    stimulator_array_parameters, *num_trials* times.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    sheet_list : int
                The list of sheets in which to do stimulation.

    num_trials : int
                Number of trials each stimulus is shown.

    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are to be entered separately in this experiment.

    stimulating_signal : str
                Described in more detail in
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR.
                Path to a function that specifying the timecourse of the optogenetic
                array stimulation, should have the following form:

                def stimulating_signal_function(sheet, coor_x, coor_y, update_interval, parameters)

                sheet - Mozaik Sheet to be stimulated
                coor_x - x coordinates of the stimulators
                coor_y - y coordinates of the stimulators
                update_interval - timestep in which the stimulator updates
                parameters - any extra user parameters for the function

                It should return a 3D numpy array of size:
                coor_x.shape[0] x coor_x.shape[1] x (stimulation_duration/update_interval)

    stimulating_signal_parameters : ParameterSet
                Extra user parameters for the `stimulating_signal` function,
                described above.
    """

    required_parameters = ParameterSet({
            'sheet_list': list,
            'num_trials' : int,
            'stimulator_array_parameters' : ParameterSet,
            'stimulating_signal': str,
            'stimulating_signal_parameters': ParameterSet,
    })


    def __init__(self,model,parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model,parameters)
        self.parameters.stimulator_array_parameters["stimulating_signal"] = self.parameters.stimulating_signal
        self.parameters.stimulator_array_parameters["stimulating_signal_parameters"] = self.parameters.stimulating_signal_parameters

        for trial in range(self.parameters.num_trials):
            self.append_direct_stim(trial,model,self.parameters.stimulator_array_parameters)


class OptogeneticArrayStimulusCircles(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources,
    in the pattern of filled circles.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    sheet_list : int
                The list of sheets in which to do stimulation.

    num_trials : int
                Number of trials each stimulus is shown.

    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

    intensity : float
                Intensity of the stimulation. Uniform across the circle.

    radii : list(float (μm))
                List of circle radii (μm) to present

    x_center : float (μm)
                Circle center x coordinate.

    y_center : float (μm)
                Circle center y coordinate.

    inverted: bool
                If true, everything in the circle has value 0, everything outside has
                the value *intensity*

    flash_duration: int
                The duration of a single pattern flash.

    blank_duration: int
                The duration of no stimulation after a pattern flash.
    """

    required_parameters = ParameterSet({
        'sheet_list' : list,
        'num_trials' : int,
        'stimulator_array_parameters' : ParameterSet,
        'intensity': float,
        'radii' : list,
        'x_center' : float,
        'y_center' : float,
        'inverted': bool,
        'flash_duration': int,
        'blank_duration': int,
    })

    def __init__(self,model,parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model,parameters)
        self.parameters.stimulator_array_parameters["stimulating_signal"] = "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        self.parameters.stimulator_array_parameters["stimulating_signal_parameters"] = ParameterSet({
            "shape": "circle-inverted" if self.parameters.inverted else "circle",
            "intensity": self.parameters.intensity,
            "coords": (self.parameters.x_center,self.parameters.y_center),
            "radius": 0,
            "duration": self.parameters.flash_duration + self.parameters.blank_duration,
            "onset_time": 0,
            "offset_time": self.parameters.flash_duration,
        })

        for trial in range(self.parameters.num_trials):
            for r in self.parameters.radii:
                self.parameters.stimulator_array_parameters.stimulating_signal_parameters.radius = r
                self.append_direct_stim(trial,model,self.parameters.stimulator_array_parameters)


class OptogeneticArrayStimulusHexagonalTiling(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources,
    in the pattern of filled hexagons. These hexagons tile the entire span of
    the stimulation array, such that no hexagon has any parts outside of the
    array.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    sheet_list : int
                The list of sheets in which to do stimulation.

    num_trials : int
                Number of trials each stimulus is shown.

    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

    intensity : float
                Intensity of the stimulation. Uniform across the hexagon.

    radius : float (μm)
                Radius (or edge length) of the hexagon to present

    x_center : float (μm)
                Circle center x coordinate.

    y_center : float (μm)
                Circle center y coordinate.

    angle : float (rad)
                Clockwise rotation angle of the hexagons. At angle 0, the hexagons
                are oriented such that two sides are parallel to the horizontal axis:
                 __
                /  \
                \__/

    shuffle: bool
                If true, shuffle the order of hexagon flashes in a single trial.
                The order does not change across trials.

    flash_duration: int
                The duration of a single pattern flash.

    blank_duration: int
                The duration of no stimulation after a pattern flash.
    """

    required_parameters = ParameterSet({
        'sheet_list' : list,
        'num_trials' : int,
        'stimulator_array_parameters' : ParameterSet,
        'intensity': float,
        'radius' : float,
        'x_center' : float,
        'y_center' : float,
        'angle' : float,
        'shuffle': bool,
        'flash_duration': int,
        'blank_duration': int,
    })

    def __init__(self,model,parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model,parameters)
        self.parameters.stimulator_array_parameters["stimulating_signal"] = "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        self.parameters.stimulator_array_parameters["stimulating_signal_parameters"] = ParameterSet({
            "shape": "hexagon",
            "intensity": self.parameters.intensity,
            "angle": self.parameters.angle,
            "coords": (0,0),
            "radius": self.parameters.radius,
            "duration": self.parameters.flash_duration + self.parameters.blank_duration,
            "onset_time": 0,
            "offset_time": self.parameters.flash_duration,
        })

        hc = self.hexagonal_tiling_centers(
            self.parameters.x_center,
            self.parameters.y_center,
            self.parameters.radius,
            self.parameters.angle,
            self.parameters.stimulator_array_parameters.size,
            self.parameters.stimulator_array_parameters.size,
            self.parameters.shuffle,
        )
        for trial in range(self.parameters.num_trials):
            for h in hc:
                self.parameters.stimulator_array_parameters.stimulating_signal_parameters.coords = h
                self.append_direct_stim(trial,model,self.parameters.stimulator_array_parameters)


    def hexagonal_tiling_centers(self, x_c, y_c, radius, angle, xlen, ylen, shuffle=False):
        xmax, ymax, xmin, ymin = xlen / 2, ylen / 2, -xlen / 2, -ylen / 2
        w = radius * np.sqrt(3) / 2
        r = radius

        hex_coords = []
        for y in np.arange(0, ymax, 3 * r):
            for x in np.arange(0, xmax, 2 * w):
                hex_coords.append([x, y])
                hex_coords.append([-x, y])
                hex_coords.append([x, -y])
                hex_coords.append([-x, -y])

        for y in np.arange(3 / 2 * r, ymax, 3 * r):
            for x in np.arange(w, xmax, 2 * w):
                hex_coords.append([x, y])
                hex_coords.append([-x, y])
                hex_coords.append([x, -y])
                hex_coords.append([-x, -y])

        transform = (
            matplotlib.transforms.Affine2D()
            .translate(x_c, y_c)
            .rotate_around(x_c, y_c, angle)
        )
        hex_coords = transform.transform_affine(np.array(hex_coords))

        hc = []
        for coord in hex_coords:
            x, y = coord
            if x + r > xmax or x - r < xmin or y + r > ymax or y - r < ymin:
                continue
            hc.append(coord)

        # Remove duplicate coordinates
        hc = sorted(list(set(map(tuple, hc))))

        if shuffle == True:
            random.shuffle(hc)
        return hc


class OptogeneticArrayStimulusOrientationTuningProtocol(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources, with a
    pattern based on the cortical orientation map, simulating homogeneously oriented
    visual stimuli.

    At each iteration of the orientation tuning protocol, an orientation
    is selected as the primary orientation to maximally stimulate (with *intensity*
    intensity), and the stimulation intensity for the other orientations in the cortical
    orientation map falls off as a Gaussian with the circular distance from the selected
    orientation:

    Stimulation intensity = intensity * e^(-0.5*d^2/sharpness)
    d = circular_dist(selected_orientation-or_map_orientation)

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    sheet_list : int
               The list of sheets in which to do stimulation.

    num_trials : int
                Number of trials each stimulus is shown.

    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

    num_orientations : int
                Number of orientations to present

    intensities : list(float)
                Intensities at which to present the stimulation

    sharpness : float
            Variance of the Gaussian falloff

    duration : float (ms)
            Overall stimulus duration

    onset_time : float (ms)
            Time point when the stimulation turns on

    offset_time : float(ms)
            Time point when the stimulation turns off
    """
    
    required_parameters = ParameterSet({
            'sheet_list' : list,
            'num_trials' : int,
            'stimulator_array_parameters' : ParameterSet,
            'num_orientations' : int,
            'intensities' : list,
            'sharpness' : float,
            'duration': int,
            'onset_time': int,
            'offset_time': int,
    })

    def __init__(self,model,parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model,parameters)
        self.parameters.stimulator_array_parameters["stimulating_signal"] = "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        self.parameters.stimulator_array_parameters["stimulating_signal_parameters"] = ParameterSet({
            "shape": "or_map",
            "intensity": 0,
            "orientation": 0,
            "sharpness": self.parameters.sharpness,
            "duration": self.parameters.duration,
            "onset_time": self.parameters.onset_time,
            "offset_time": self.parameters.offset_time,
        })

        orientations = np.linspace(0,np.pi,self.parameters.num_orientations,endpoint=False)
        for trial in range(self.parameters.num_trials):
            for intensity in self.parameters.intensities:
                for orientation in orientations:
                    self.parameters.stimulator_array_parameters.stimulating_signal_parameters.intensity = intensity
                    self.parameters.stimulator_array_parameters.stimulating_signal_parameters.orientation = orientation
                    self.append_direct_stim(trial,model,self.parameters.stimulator_array_parameters)



class OptogeneticArrayStimulusContrastBasedOrientationTuningProtocol(CorticalStimulationWithOptogeneticArray):
    """
    TODO
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    sheet_list : int
               The list of sheets in which to do stimulation.

    num_trials : int
                Number of trials each stimulus is shown.

    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

    num_orientations : int
                Number of orientations to present

    contrasts : list(float)
                TODO

    naka_params: ParameterSet
                TODO

    sharpness : float
            Variance of the Gaussian falloff

    duration : float (ms)
            Overall stimulus duration

    onset_time : float (ms)
            Time point when the stimulation turns on

    offset_time : float(ms)
            Time point when the stimulation turns off
    """

    required_parameters = ParameterSet({
            'sheet_list' : list,
            'num_trials' : int,
            'stimulator_array_parameters' : ParameterSet,
            'num_orientations' : int,
            'contrasts' : list,
            'naka_params' : ParameterSet,
            'sharpness' : float,
            'duration': int,
            'onset_time': int,
            'offset_time': int,
    })


    def __init__(self,model,parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model,parameters)
        self.parameters.stimulator_array_parameters["stimulating_signal"] = "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        self.parameters.stimulator_array_parameters["stimulating_signal_parameters"] = ParameterSet({
            "shape": "or_map",
            "orientation": 0,
            "sharpness": self.parameters.sharpness,
            "duration": self.parameters.duration,
            "naka_params": self.parameters.naka_params,
            "onset_time": self.parameters.onset_time,
            "offset_time": self.parameters.offset_time,
        })
        orientations = np.linspace(0,np.pi,self.parameters.num_orientations,endpoint=False)
        for trial in range(self.parameters.num_trials):
            for contrast in self.parameters.contrasts:
                for orientation in orientations:
                    self.parameters.stimulator_array_parameters.stimulating_signal_parameters.orientation = orientation
                    self.parameters.stimulator_array_parameters.stimulating_signal_parameters.contrast = contrast
                    self.append_direct_stim(trial,model,self.parameters.stimulator_array_parameters)
