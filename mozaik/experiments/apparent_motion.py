from mozaik.experiments.vision import VisualExperiment
from parameters import ParameterSet
import mozaik.stimuli.vision.topographica_based as topo
import numpy

class MapSimpleGabor(VisualExperiment):
    """
    Map RF with a Gabor patch stimuli.

    This experiment presents a series of flashed Gabor patches at the centers
    of regular hexagonal tides with given range of orientations.

    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    relative_luminance : float
        Luminance of the Gabor patch relative to background luminance.
        0. is dark, 1.0 is double the background luminance.

    central_rel_lum : float
        Luminance of the Gabor patch at the center of the RF relative to
        background luminance.
        0. is dark, 1.0 is double the background luminance.

    orientation : float
        The initial orientation of the Gabor patch.

    phase : float
        The phase of the Gabor patch.

    spatial_frequency : float
        The spatial freqency of the Gabor patch.

    rotations : int
        Number of different orientations at each given place.
        1 only one Gabor patch with initial orientation will be presented at
        given place, N>1 N different orientations will be presented, 
        orientations are uniformly distributed between [0, 2*pi) + orientation.

    size : float
        Size of the tides. From this value the size of Gabor patch is derived 
        so that it fits into a circle with diameter equal to this size.

        Gabor patch size is set so that sigma of Gaussian envelope is size/3

    x : float
        The x corrdinates of the central tide.

    y : float
        The y corrdinates of the central tide.

    flash_duration : float
        The duration of the presentation of a single Gabor patch. 

    duration : float
        The duration of single presentation of the stimulus.

    num_trials : int
        Number of trials each each stimulus is shown.

    circles : int
        Number of "circles" where the Gabor patch is presented.
        1: only at the central point the Gabor patch is presented, 
        2: stimuli are presented at the central hexagonal tide and 6 hexes 
        forming a "circle" around the central

    grid : bool
        If True hexagonal tiding with relative luminance 0 is drawn over the 
        stimmuli.
        Mostly for testing purposes to check the stimuli are generated 
        correctly.

    Note on hexagonal tiding:
    -------------------------
        Generating coordinates of centers of regular (!) hexagonal tidings.
        It is done this way, because the centers of tides are not on circles (!)
        First it generates integer indexed centers like this:
              . . .                (-2,2) (0, 2) (2,2)
             . . . .           (-3,1) (-1,1) (1,1) (3,1)
            . . . . .   ==> (-4,0) (-2,0) (0,0) (2,0) (4,0)     (circles=3)
             . . . .           (-3,-1)(-1,-1)(1,-1)(3,-1)
              . . .                (-2,-2)(0,-2)(2,-2)

        coordinates then multiplied by non-integer factor to get the right position
            x coordinate multiplied by factor 1/2*size
            y coordinate multiplied by factor sqrt(3)/2*size

    Note on central relative luminance:
    -----------------------------------
        In the experiment they had lower luminance for Gabor patches presented
        at the central tide
    """

    required_parameters = ParameterSet({
            'relative_luminance' : float,
            'central_rel_lum' : float,
            'orientation' : float,
            'phase' : float,
            'spatial_frequency' : float,
            'size' : float,
            'flash_duration' : float, 
            'x' : float,
            'y' : float,
            'rotations' : int,
            'duration' : float,
            'num_trials' : int,
            'circles' : int,
            'grid' : bool,
    })


    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        if self.parameters.grid:
            # Grid is currently working only for special cases
            # Check if it is working
            assert self.parameters.x == 0, "X shift not yet implemented"
            assert self.parameters.y == 0, "Y shift not yet implemented"
            assert model.visual_field.size_x == model.visual_field.size_y, "Different sizes not yet implemented"
        for trial in xrange(0, self.parameters.num_trials):
            for rot in xrange(0, self.parameters.rotations):
                for row in xrange(self.parameters.circles-1, -self.parameters.circles,-1):
                    colmax =  2*self.parameters.circles-2 - abs(row)
                    for column in xrange(-colmax, colmax + 1, 2):
                        # central coordinates of presented Gabor patch
                        # relative to the central tide
                        x = column*0.5*self.parameters.size
                        y = row*0.5*self.parameters.size  
                        # different luminance for central tide
                        if column == 0 and row == 0:
                            rel_lum = self.parameters.central_rel_lum
                        else:
                            rel_lum = self.parameters.relative_luminance
                        self.stimuli.append(
                            topo.SimpleGaborPatch(
                                frame_duration = self.frame_duration,
                                duration=self.parameters.duration,
                                flash_duration = self.parameters.flash_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                background_luminance=self.background_luminance,
                                relative_luminance = rel_lum,
                                orientation = (self.parameters.orientation 
                                            + numpy.pi*rot/self.parameters.rotations),
                                density=self.density,
                                phase = self.parameters.phase,
                                spatial_frequency = self.parameters.spatial_frequency,
                                size = self.parameters.size,
                                x = self.parameters.x + x,
                                y = self.parameters.y + y,
                                location_x=0.0,
                                location_y=0.0,
                                trial=trial))

    def do_analysis(self, data_store):
        pass


class MapTwoStrokeGabor(VisualExperiment):
    """
    Map RF with a two stroke Gabor patch stimuli to study response on apparent
    movement. First a Gabor patch is presented for specified time after that
    another Gabor patch is presented at neighbohring tide with same orientation
    and other properties.

    There are two configuration for the movement:
        ISO i.e. Gabor patch moves parallel to its orientation
        CROSS i.e. Gabor patch moves perpendicular to its orientation
        
        In any case it has to move into another tide, therefore orientation 
        determines the configuration

  
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    relative_luminance : float
        Luminance of the Gabor patch relative to background luminance.
        0. is dark, 1.0 is double the background luminance.

    central_rel_lum : float
        Luminance of the Gabor patch at the center of the RF relative to
        background luminance.
        0. is dark, 1.0 is double the background luminance.

    orientation : float
        The initial orientation of the Gabor patch.
        This changes orientation of the whole experiment, i.e. it also rotates 
        the grid (because of the iso and cross configurations of movements).

    phase : float
        The phase of the Gabor patch.

    spatial_frequency : float
        The spatial freqency of the Gabor patch.

    rotations : int
        Number of different orientations at each given place.
        1 only one Gabor patch with initial orientation will be presented at
        given place, N>1 N different orientations will be presented, 
        orientations are uniformly distributed between [0, 2*pi) + orientation.

    size : float
        Size of the tides. From this value the size of Gabor patch is derived 
        so that it fits into a circle with diameter equal to this size.

        Gabor patch size is set so that sigma of Gaussian envelope is size/3

    x : float
        The x corrdinates of the central tide.

    y : float
        The y corrdinates of the central tide.

    stroke_time : float
        The duration of the first stroke of Gabor patch

    flash_duration : float
        The total duration of the presentation of Gabor patches. Therefore,
        the second stroke is presented for time equal: 
            flash_duration - stroke_tim 

    duration : float
        The duration of single presentation of the stimulus.

    num_trials : int
        Number of trials each each stimulus is shown.

    circles : int
        Number of "circles" where the Gabor patch is presented.
        1: only at the central point the Gabor patch is presented, 
        2: stimuli are presented at the central hexagonal tide and 6 hexes 
        forming a "circle" around the central
        Trajectories starting or ending in the given number of circles are
        used, i.e. First Gabor patch can be out of the circles and vice versa.

    grid : bool
        If True hexagonal tiding with relative luminance 0 is drawn over the 
        stimmuli.
        Mostly for testing purposes to check the stimuli are generated 
        correctly.

    Note on hexagonal tiding:
    -------------------------
        Generating coordinates of centers of regular (!) hexagonal tidings.
        It is done this way, because the centers of tides are not on circles (!)
        First it generates integer indexed centers like this:
              . . .                (-2,2) (0, 2) (2,2)
             . . . .           (-3,1) (-1,1) (1,1) (3,1)
            . . . . .   ==> (-4,0) (-2,0) (0,0) (2,0) (4,0)     (circles=3)
             . . . .           (-3,-1)(-1,-1)(1,-1)(3,-1)
              . . .                (-2,-2)(0,-2)(2,-2)

        coordinates then multiplied by non-integer factor to get the right position
            x coordinate multiplied by factor 1/2*size
            y coordinate multiplied by factor sqrt(3)/2*size

    Note on central relative luminance:
    -----------------------------------
        In the experiment they had lower luminance for Gabor patches presented
        at the central tide


    Note on number of circles:
    --------------------------
        For 2 stroke the experiment includes also the trajectories that
        start inside the defined number of circles but get out as well as 
        trajectories starting in the outside layer of tides comming inside.

        For example if we have number of circles = 2 -> that means we have 
        central tide and the first circle of tides around, but for two stroke
        it is possible we start with Gabor patch at the distance 2 tides away
        from the central tide (i.e. tides that are in circles = 3) if we move 
        inside and vice versa.

        This is solved by checking the distance of the final position of the 
        Gabor patch, if the distance is bigger than a radius of a circle
        then opposite direction is taken into account.
        
        Since we have hexagonal tides this check is valid only for 
        n <= 2/(2-sqrt(3)) ~ 7.5 
        which is for given purposes satisfied, but should be mentioned.

    Note on rotations:
    ------------------
        This number is taken as a free parameter, but to replicate hexagonal
        tiding this number has to be 6 or 1 or 2. The code exploits symmetry and
        properties of the hexagonal tiding rather a lot!
        The ISO/CROSS configuration is determined from this number, so any other
        number generates moving paterns but in directions not matching hexes.

    """

    required_parameters = ParameterSet({
            'relative_luminance' : float,
            'central_rel_lum' : float,
            'orientation' : float,
            'phase' : float,
            'spatial_frequency' : float,
            'size' : float,
            'flash_duration' : float, 
            'x' : float,
            'y' : float,
            'rotations' : int,
            'duration' : float,
            'num_trials' : int,
            'circles' : int,
            'stroke_time' : float,
            'grid' : bool,
            })  


    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        # Assert explained in docstring
        assert self.parameters.circles < 7, "Too many circles, this won't work"
        if self.parameters.grid:
            # Grid is currently working only for special cases
            # Check if it is working
            assert self.parameters.orientation == 0., "Rotated grid is not implemented"
            assert self.parameters.x == 0, "X shift not yet implemented"
            assert self.parameters.y == 0, "Y shift not yet implemented"
            assert model.visual_field.size_x == model.visual_field.size_y, "Different sizes not yet implemented"


        for trial in xrange(0, self.parameters.num_trials):
            for rot in xrange(0, self.parameters.rotations):
                for row in xrange(self.parameters.circles-1, -self.parameters.circles,-1):
                    colmax =  2*self.parameters.circles-2 - abs(row)
                    for column in xrange(-colmax, colmax + 1, 2):
                        for direction in (-1,1):
                            # central coordinates of presented Gabor patch
                            # relative to the central tide
                            x = column*0.5*self.parameters.size
                            y = row*0.5*numpy.sqrt(3)*self.parameters.size  
                            # rotation of the Gabor
                            angle = (self.parameters.orientation 
                                    + numpy.pi*rot/self.parameters.rotations)
                            if rot%2 == 0: # even rotations -> iso config
                                # Gabor orientation 0 -> horizontal
                                x_dir = numpy.cos(angle)*self.parameters.size
                                y_dir = numpy.sin(angle)*self.parameters.size
                            else:  # odd rotations -> cross config
                                # cross config means moving into perpendicular
                                # direction (aka + pi/2)
                                x_dir = -numpy.sin(angle)*self.parameters.size
                                y_dir = numpy.cos(angle)*self.parameters.size

                            # starting in the central tide
                            if x == 0 and y == 0:
                                first_rel_lum = self.parameters.central_rel_lum
                                second_rel_lum = self.parameters.relative_luminance
                            # ending in the central tide
                            elif ((abs(x + x_dir*direction) < self.parameters.size/2.) and
                                  (abs(y + y_dir*direction) < self.parameters.size/2.)):
                                first_rel_lum = self.parameters.relative_luminance
                                second_rel_lum = self.parameters.central_rel_lum
                            # far from the central tide
                            else:
                                first_rel_lum = self.parameters.relative_luminance
                                second_rel_lum = self.parameters.relative_luminance


                            # If the Gabor patch ends in outer circle
                            # we want also Gabor moving from outer circle to 
                            # inner circles 
                            # This condition is approximated by concentric 
                            # circles more in docstring
                            outer_circle = numpy.sqrt((x+x_dir*direction)**2 
                                        + (y+y_dir*direction)**2) > (
                                                (self.parameters.circles-1)
                                                *self.parameters.size)

                            # range here is 1 or 2
                            # In case of outer_circle == True generates two
                            # experiments, from and into the outer circle
                            # In case of outer_circle == False generates only
                            # one experiment
                            for inverse in xrange(1+outer_circle):
                                self.stimuli.append(
                                    topo.TwoStrokeGaborPatch(
                                        frame_duration = self.frame_duration,
                                        duration=self.parameters.duration,
                                        flash_duration = self.parameters.flash_duration,
                                        size_x=model.visual_field.size_x,
                                        size_y=model.visual_field.size_y,
                                        background_luminance=self.background_luminance,
                                        first_relative_luminance = first_rel_lum,
                                        second_relative_luminance = second_rel_lum,
                                        orientation = angle,
                                        density=self.density,
                                        phase = self.parameters.phase,
                                        spatial_frequency = self.parameters.spatial_frequency,
                                        size = self.parameters.size,
                                        x = self.parameters.x + x + inverse*x_dir*direction,
                                            # inverse == 0 -> original start
                                            # inverse == 1 -> start from end
                                        y = self.parameters.y + y + inverse*y_dir*direction,
                                        location_x=0.0,
                                        location_y=0.0,
                                        trial=trial,
                                        stroke_time=self.parameters.stroke_time,
                                        x_direction=x_dir*direction*((-1)**inverse),
                                            # (-1)**inverse = 1 for original one
                                            # == -1 for the inverse movement
                                        y_direction=y_dir*direction*((-1)**inverse),
                                        grid=self.parameters.grid,
                                        ))
                                # For the inverse movement we have to 
                                # switch the luminances
                                first_rel_lum, second_rel_lum = second_rel_lum, first_rel_lum

    def do_analysis(self, data_store):
        pass


class CompareSlowVersusFastGaborMotion(VisualExperiment):
    """
    
    """

    required_parameters = ParameterSet({
            'num_trials' : int,
            'x' : float,
            'y' : float,
            'orientation' : float,
            'phase' : float,
            'spatial_frequency' : float,
            'sigma' : float,
            'n_sigmas' : float,
            'center_relative_luminance' : float,
            'surround_relative_luminance' : float,
            'movement_speeds' : list,
            'angles' : list,
            'moving_gabor_orientation_radial' : bool,
            'radius' : int,
    })

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        common_params = {
            'size_x' : model.visual_field.size_x,
            'size_y' : model.visual_field.size_y,
            'location_x' : 0.0,
            'location_y' : 0.0,
            'background_luminance' : self.background_luminance,
            'density' : self.density,
            'frame_duration' : self.frame_duration,
            'x' : self.parameters.x,
            'y' : self.parameters.y,
            'orientation' : self.parameters.orientation,
            'phase' : self.parameters.phase,
            'spatial_frequency' : self.parameters.spatial_frequency,
            'sigma' : self.parameters.sigma,
            'center_relative_luminance' : self.parameters.center_relative_luminance,
        }

        am_specific_params = {
        'surround_relative_luminance' : self.parameters.surround_relative_luminance,
        'surround_gabor_orientation_radial' : self.parameters.moving_gabor_orientation_radial,
        }
        cont_mov_specific_params = {
        'moving_relative_luminance' : self.parameters.surround_relative_luminance,
        'moving_gabor_orientation_radial' : self.parameters.moving_gabor_orientation_radial,
        }

        am_params = common_params.copy()
        am_params.update(am_specific_params)
        cont_mov_params = common_params.copy()
        cont_mov_params.update(cont_mov_specific_params)
        for trial in xrange(0, self.parameters.num_trials):
            for speed in self.parameters.movement_speeds:
                gabor_diameter = 2.0*self.parameters.sigma*self.parameters.n_sigmas
                flash_duration=gabor_diameter/speed*1000
                assert flash_duration >= self.frame_duration, "Gabor flash duration must be at least as long as the frame duration"
                stim_duration = flash_duration * (self.parameters.radius + 1)
                am_params["duration"] = stim_duration
                cont_mov_params["duration"] = stim_duration
                for angle in self.parameters.angles:
                    # Apparent Motion
                    am_stim=topo.RadialGaborApparentMotion(
                        flash_duration=flash_duration,
                        start_angle=angle,
                        end_angle=angle,
                        n_gabors=1,
                        n_circles=self.parameters.radius,
                        symmetric=False,
                        random=False,
                        flash_center=True,
                        centrifugal=False,
                        trial=trial,
                        **am_params
                    )
                    # Center-only stimulation
                    co_stim=topo.SimpleGaborPatch(
                        duration=stim_duration,
                        flash_duration=stim_duration,
                        relative_luminance=self.parameters.center_relative_luminance,
                        trial=trial,
                        **common_params
                    )

                    # Continuous Motion
                    cont_mov_stim=topo.ContinuousGaborMovementAndJump(
                        movement_duration=flash_duration * (self.parameters.radius-1),
                        movement_length=gabor_diameter * (self.parameters.radius-1),
                        movement_angle=angle,
                        center_flash_duration=flash_duration,
                        trial=trial,
                        **cont_mov_params
                    )

                    # Center-only stimulation
                    co_stim_0=topo.SimpleGaborPatch(
                        duration=stim_duration,
                        flash_duration=stim_duration,
                        relative_luminance=self.parameters.center_relative_luminance,
                        trial=trial,
                        **common_params
                    )
                    self.stimuli.append(am_stim)
                    self.stimuli.append(co_stim)
                    self.stimuli.append(cont_mov_stim)
                    self.stimuli.append(co_stim_0)

    def do_analysis(self, data_store):
        pass
