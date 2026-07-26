# encoding: utf-8
r"""
This module contains implementation of vision related sheets.
"""

import numpy
import mozaik
from parameters import ParameterSet
from pyNN import space
from pyNN.errors import NothingToWriteError
from mozaik.sheets import Sheet
        
logger = mozaik.getMozaikLogger()


class ExplicitPositions(space.BaseStructure):
    """PyNN structure backed by a fixed global position array."""

    parameter_names = ()

    def __init__(self, positions):
        try:
            positions = numpy.asarray(positions, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("positions must be a finite array with shape (3, N)") from exc
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("positions must be a finite array with shape (3, N)")
        if not numpy.all(numpy.isfinite(positions)):
            raise ValueError("positions must contain only finite values")
        if not numpy.array_equal(positions[2], numpy.zeros(positions.shape[1])):
            raise ValueError("the third row of positions must contain only zeros")
        self._positions = positions.copy()

    def generate_positions(self, n):
        if n != self._positions.shape[1]:
            raise ValueError(
                "ExplicitPositions contains %d positions, but PyNN requested %d"
                % (self._positions.shape[1], n)
            )
        return self._positions.copy()


class RetinalUniformSheet(Sheet):
    r"""
    Retinal sheet corresponds to a sheet of retinal cells (retinal ganglion cells or photoreceptors). 
    It implicitly assumes the coordinate systems is in degress in visual field.
    
    Other parameters
    ----------------
    
    sx : float (degrees)
        X size of the region.
        
    sy : float (degrees)
        Y size of the region.

    density : int
        Number of neurons along both axis.

    """
    required_parameters = ParameterSet({
        'sx': float,  # degrees, x size of the region
        'sy': float,  # degrees, y size of the region
        'density': int,  # neurons along each axis
    })
    
    def __init__(self, model, parameters):
        Sheet.__init__(self, model,parameters.sx, parameters.sy, parameters)
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__, int(parameters.sx * parameters.sy * parameters.density)))
        rs = space.RandomStructure(boundary=space.Cuboid(self.size_x,self.size_y, 0),
                                   origin=(0.0, 0.0, 0.0),
                                   rng=mozaik.pynn_rng)
        
        if self.parameters.cell.native_nest:
            self.pop = self.sim.Population(int(parameters.sx * parameters.sy * parameters.density),
                                               self.sim.native_cell_type(self.parameters.cell.model)(**self.parameters.cell.params),
                                               structure=rs,
                                               initial_values=self.parameters.cell.initial_values,
                                               label=self.name)
        else:
            self.pop = self.sim.Population(int(parameters.sx * parameters.sy * parameters.density),
                                               getattr(self.model.sim, self.parameters.cell.model)(**self.parameters.cell.params),
                                               structure=rs,
                                               initial_values=self.parameters.cell.initial_values,
                                               label=self.name)
        # Forces PyNN to generate the positions to ensure the reproducibility with multiprocessing
        self.pop.positions

    def size_in_degrees(self):
        return (self.parameters.sx, self.parameters.sy)


class RetinalInhomogeneousDiskSheet(Sheet):
    """Retinal sheet with explicit Cartesian RF centres in visual degrees."""

    def __init__(self, model, parameters, positions_deg, topography):
        try:
            positions_deg = numpy.asarray(positions_deg, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "positions_deg must be a finite array with shape (2, N)"
            ) from exc
        if positions_deg.ndim != 2 or positions_deg.shape[0] != 2:
            raise ValueError("positions_deg must be a finite array with shape (2, N)")
        if positions_deg.shape[1] == 0:
            raise ValueError("RetinalInhomogeneousDiskSheet requires at least one cell")
        if not numpy.all(numpy.isfinite(positions_deg)):
            raise ValueError("positions_deg must contain only finite values")

        topography.validate_visual_position(positions_deg[0], positions_deg[1])
        if numpy.any(
            numpy.hypot(positions_deg[0], positions_deg[1])
            >= topography.max_eccentricity_deg
        ):
            raise ValueError(
                "retinal RF centres must satisfy hypot(x, y) < E_max"
            )

        Sheet.__init__(
            self,
            model,
            2.0 * topography.max_eccentricity_deg,
            2.0 * topography.max_eccentricity_deg,
            parameters,
        )
        self.topography = topography
        self.canonical_positions_deg = positions_deg.copy()
        self.canonical_positions_deg.setflags(write=False)

        positions = numpy.vstack(
            (self.canonical_positions_deg, numpy.zeros(positions_deg.shape[1]))
        )
        structure = ExplicitPositions(positions)
        logger.info(
            "Creating %s with %d neurons."
            % (self.__class__.__name__, positions_deg.shape[1])
        )
        if self.parameters.cell.native_nest:
            cell_type = self.sim.native_cell_type(self.parameters.cell.model)(
                **self.parameters.cell.params
            )
        else:
            cell_type = getattr(self.model.sim, self.parameters.cell.model)(
                **self.parameters.cell.params
            )
        self.pop = self.sim.Population(
            positions_deg.shape[1],
            cell_type,
            structure=structure,
            initial_values=self.parameters.cell.initial_values,
            label=self.name,
        )

        realized_positions = self.pop.positions
        if not numpy.array_equal(
            realized_positions[:2], self.canonical_positions_deg
        ):
            raise AssertionError(
                "PyNN realized retinal positions differ from the canonical "
                "global position array"
            )

    def size_in_degrees(self):
        diameter = 2.0 * self.topography.max_eccentricity_deg
        return (diameter, diameter)


class SheetWithMagnificationFactor(Sheet):
    r"""
    A Sheet that has a magnification factor corresponding to cortical visual area.
    It interprets the coordinates system to be in degrees of visual field, but it allows
    for definition of the layer using parameters in cortical space. It offers 
    number of functions that facilitate conversion between the underlying visual degree
    coordinates and cortical space coordinate systems using the magnification factor. 
    
    Other parameters
    ----------------

    magnification_factor : float (μm/degree)
        The magnification factor.
    
    sx : float (μm)
        X size of the region.
        
    sy : float (μm)
        Y size of the region.

    """
    required_parameters = ParameterSet({
        'magnification_factor': float,  # μm / degree
        'sx': float,      # μm, x size of the region
        'sy': float,      # μm, y size of the region
    })

    def __init__(self, model, parameters):
        r"""
        """
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__, int(parameters.sx*parameters.sy/1000000*parameters.density)))
        Sheet.__init__(self, model, parameters.sx/ parameters.magnification_factor,parameters.sy/parameters.magnification_factor,parameters)
        self.magnification_factor = parameters.magnification_factor

    def vf_2_cs(self, degree_x, degree_y):
        r"""
        vf_2_cs converts the position (degree_x, degree_y) in visual field to
        position in cortical space (in μm) given the magnification_factor.
        
        Parameters
        ----------

        degree_x : float (degrees)
            X coordinate of the position in degrees of visual field
        degree_y : float (degrees)
            Y coordinate of the position in degrees of visual field
        
        Returns
        -------

        microm_meters_x,microm_meters_y : float,float (μm,μm)
            Tuple with the coordinates in cortical space (μm)

        
        """
        return (degree_x * self.magnification_factor,
                degree_y * self.magnification_factor)

    def cs_2_vf(self, micro_meters_x, micro_meters_y):
        r"""
        cs_2_vf converts the position (micro_meters_x, micro_meters_y) in
        cortical space to the position in the visual field (in degrees) given
        the magnification_factor
        
        Parameters
        ----------

        micro_meters_x : float (μm)
            X coordinate of the position in μm of cortical space
        micro_meters_y : float (μm)
            Y coordinate of the position in μm of cortical space
        
        Returns
        -------

        degrees_x,degrees_y : float,float (degrees,degrees)
            Tuple with the coordinates in visual space (degrees)

        """
        return (micro_meters_x / self.magnification_factor,
                micro_meters_y / self.magnification_factor)

    def dvf_2_dcs(self, distance_vf):
        r"""
        dvf_2_dcs converts the distance in visual space to the distance in
        cortical space given the magnification_factor
        
        Parameters
        ----------

        distance_vf : float (degrees)
            The distance in visual field coordinates (degrees).
                 
        Returns
        -------

        distance_cs : float (μm)
            Distance in cortical space.

        """
        return distance_vf * self.magnification_factor

    def size_in_degrees(self):
        r"""
        Returns the size of the sheet in cortical space (μm).
        """
        return self.cs_2_vf(self.parameters.sx, self.parameters.sy)


class VisualCorticalUniformSheet(SheetWithMagnificationFactor):
    r"""
    Represents a visual cortical sheet of neurons, randomly uniformly distributed in cortical space.
    
    Other parameters
    ----------------

    density : float (neurons/mm^2)
        The density of neurons per square milimeter.

    """
    
    required_parameters = ParameterSet({
        'density': float,  # neurons/(mm^2)
    })

    def __init__(self, model, parameters):
        SheetWithMagnificationFactor.__init__(self, model, parameters)
        dx, dy = self.cs_2_vf(parameters.sx, parameters.sy)
        rs = space.RandomStructure(boundary=space.Cuboid(dx, dy, 0),
                                   origin=(0.0, 0.0, 0.0),
                                   rng=mozaik.pynn_rng)

        # Include nestml multisynapse neuron model name here
        if self.parameters.cell.model in set(["aeif_cond_alpha_multisynapse","aeif_cond_beta_multisynapse"]):
            self.multisynapse = True
            #TODO after nestml multisynapse neuron model is implemented
            if self.parameters.cell.native_nest:
                pass

            else:
                receptors= {}
                for (k,v) in self.parameters.cell.receptors.items():
                    receptors[k] = getattr(self.model.sim, v.name)(**v.params)
                    
                celltype = self.sim.PointNeuron(
                    self.sim.AdExp(**self.parameters.cell.params),
                                    **receptors)
                    
                    
                self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density), 
                                                celltype,structure=rs, initial_values=self.parameters.cell.initial_values,
                                                label= self.name)    
        
        else:
            if self.parameters.cell.native_nest:
                self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density),
                                                   self.sim.native_cell_type(self.parameters.cell.model)(**self.parameters.cell.params),
                                                   structure=rs,
                                                   initial_values=self.parameters.cell.initial_values,
                                                   label=self.name)
            else:
                self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density),
                                                   getattr(self.model.sim, self.parameters.cell.model)(**self.parameters.cell.params),
                                                   structure=rs,
                                                   initial_values=self.parameters.cell.initial_values,
                                                   label=self.name)

        # Forces PyNN to generate the positions to ensure the reproducibility with multiprocessing
        self.pop.positions


class VisualCorticalUniformSheet3D(VisualCorticalUniformSheet):
    r"""
    Represents a visual cortical sheet of neurons, randomly uniformly distributed in cortical space.
    In addition to the VisualCorticalUniformSheet it adds 3rd dimension to the neurons that corresponds their depth 
    within cortical sheet (prepandicular to the cortical surface). 
    In the third dimensions, the neurons will be uniformly distributed between the *min_depth* and *max_depth* parameters.
    
    Notes
    -----

    Manny existing Mozaik components that take neural position into consideration will 
    ignore this 3rd dimension. Also unlike the first to dimensions, corresponding to the axis along
    the cortical surface, the third depth dimension is in μm!

    Also note the density is still calculated only per surface unit.
    
    Other parameters
    ----------------
    
    min_depth : float (μm)
        The mininmum depth of neurons.
        
    max_depth : float (μm)
        The maxinmum depth of neurons.

    """
    
    required_parameters = ParameterSet({
        'min_depth': float,  # μm
        'max_depth': float,  # μm
    })

    def __init__(self, model, parameters):
        SheetWithMagnificationFactor.__init__(self, model, parameters)
        dx, dy = self.cs_2_vf(parameters.sx, parameters.sy)

        origin_z = (self.parameters.min_depth + self.parameters.max_depth)/2.0
        width_z = (self.parameters.max_depth - self.parameters.min_depth)

        rs = space.RandomStructure(boundary=space.Cuboid(dx, dy, width_z),
                                   origin=(0.0, 0.0, origin_z),
                                   rng=mozaik.pynn_rng)

        # Include nestml multisynapse neuron model name here
        if self.parameters.cell.model in set(["aeif_cond_alpha_multisynapse","aeif_cond_beta_multisynapse"]):
            self.multisynapse = True
            receptors= {}
            #TODO after nestml multisynapse neuron model is implemented
            if self.parameters.cell.native_nest:
                pass

            else:
                for (k,v) in self.parameters.cell.receptors.items():
                    receptors[k] = getattr(self.model.sim, v.name)(**v.params)

                celltype = self.sim.PointNeuron(
                    self.sim.AdExp(**self.parameters.cell.params),
                                    **receptors)


                self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density),
                                                celltype,structure=rs, initial_values=self.parameters.cell.initial_values,
                                                label= self.name)

        else:
            if self.parameters.cell.native_nest:
                self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density),
                                                   self.sim.native_cell_type(self.parameters.cell.model)(**self.parameters.cell.params),
                                                   structure=rs,
                                                   initial_values=self.parameters.cell.initial_values,
                                                   label=self.name)
            else:
                self.pop = self.sim.Population(int(parameters.sx * parameters.sy/1000000 * parameters.density),
                                                   getattr(self.model.sim, self.parameters.cell.model)(**self.parameters.cell.params),
                                                   structure=rs,
                                                   initial_values=self.parameters.cell.initial_values,
                                                   label=self.name)
        # Forces PyNN to generate the positions to ensure the reproducibility with multiprocessing
        self.pop.positions
