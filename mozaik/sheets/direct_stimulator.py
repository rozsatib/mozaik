# encoding: utf-8
r"""
This file contains the API for direct stimulation of neurons. 
By direct stimulation here we mean a artificial stimulation that 
would happen during electrophisiological experiment - such a injection
of spikes/currents etc into cells. In mozaik this happens at population level - i.e.
each direct stimulator specifies how the given population is stimulated. In general each population can have several
stimultors.
"""
import pylab
from mozaik.core import ParametrizedObject
from parameters import ParameterSet
import numpy
import numpy as np
import numpy.random
import mozaik
from mozaik.tools.stgen import StGen
from mozaik import load_component
from pyNN.parameters import Sequence
from mozaik import load_component
import math
from mozaik.tools.circ_stat import circular_dist ,circ_mean
import pylab
from scipy.integrate import odeint
import pickle
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from mozaik.controller import Global
import matplotlib
from mozaik.analysis.analysis import SingleValue, AnalogSignalList
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
import quantities as qt
from mozaik.tools.units import *
import io
from numba import jit
import nest
from threadpoolctl import threadpool_limits
from typing import NamedTuple

from builtins import zip

from mpi4py import MPI


logger = mozaik.getMozaikLogger()


# Temporary benchmarking switch.  The direct NEST path avoids constructing
# Neo spike trains, while the default path exercises PyNN/Mozaik's MPI-aware
# Population.get_data() implementation.
USE_DIRECT_NEST_SPIKE_RETRIEVAL = True


def _supports_integrated_current_schedules(sheet):
    cell_type = sheet.pop.celltype
    return (
        sheet.parameters.cell.native_nest
        and cell_type.has_parameter("amplitude_times")
        and cell_type.has_parameter("amplitude_values")
    )


class DirectStimulator(ParametrizedObject):
    r"""
    The API for direct stimulation.
    The DirectStimulator specifies how are cells in the assigned population directly stimulated.

    Parameters
    ----------

    parameters : ParameterSet
        The dictionary of required parameters.

    sheet : Sheet
        The sheet in which to stimulate neurons.

    Notes
    -----

    By default the direct stimulation should ensure that it is mpi-safe - this is especially crucial for
    stimulators that involve source of randomness. However, the DirectSimulators also can inspect the mpi_safe
    value of the population to which they are assigned, and if it is False they can switch to potentially
    more efficient implementation that will however not be reproducible across multi-process simulations.

    Important: the function inactivate should only temporarily inactivate the stimulator, a subsequent call to prepare_stimulation
    should activate the stimulator back!

    
    """

    def __init__(self, sheet, parameters):
        ParametrizedObject.__init__(self, parameters)
        self.sheet = sheet

    def prepare_stimulation(self,duration,offset):
        r"""
        Prepares the stimulation during the next period of model simulation lasting `duration` seconds.

        Parameters
        ----------

        duration : double (seconds)
            The period for which to prepare the stimulation

        offset : double (seconds)
            The current simulator time.

               
        """
        raise NotImplemented

    def inactivate(self,offset):
        r"""
        Ensures any influences of the stimulation are inactivated for subsequent simulation of the model.

        Parameters
        ----------

        offset : double (seconds)
            The current simulator time.

        Note that a subsequent call to prepare_stimulation should 'activate' the stimulator again.


        """
        raise NotImplemented

    def save_to_datastore(self,data_store,stimulus):
        r"""
        Save direct stimulation data to the datastore, to be used for analysis and
        visualization.

        Parameters
        ----------

        data_store : DataStore
            The data store into which to store the direct stimulation data.

                   
        """
        pass


class BackgroundActivityBombardment(DirectStimulator):
    r"""
    The BackgroundActivityBombardment simulates the poisson distrubated background bombardment of spikes onto a 
    neuron due to the other 'unsimulated' neurons in its pre-synaptic population.
    
    Parameters
    ----------

    parameters : ParameterSet
        The dictionary of required parameters.
                
    sheet : Sheet
        The sheet in which to stimulate neurons.
    
          
    Other parameters
    ----------------
    
    exc_firing_rate : float
        The firing rate of external neurons sending excitatory inputs to each neuron of this sheet.
 
    inh_firing_rate : float
        The firing rate of external neurons sending inhibitory inputs to each neuron of this sheet.
    
    exc_weight : float
        The weight of the synapses for the excitatory external Poisson input.    

    inh_weight : float
        The weight of the synapses for the inh external Poisson input.    
    
    
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!

    """
    
    
    required_parameters = ParameterSet({
            'exc_firing_rate': float,
            'exc_weight': float,
            'inh_firing_rate': float,
            'inh_weight': float,
    })
        
        
        
    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        
        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight,delay=self.sheet.model.parameters.min_delay)
        inh_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.inh_weight,delay=self.sheet.model.parameters.min_delay)
        
        if not self.sheet.parameters.mpi_safe:
            from pyNN.nest import native_cell_type        
            if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                self.np_exc = self.sheet.sim.Population(
                    len(self.sheet.pop), native_cell_type("poisson_generator")(rate=0)
                )
                self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')
                #self.np_exc = self.sheet.sim.Population(1, native_cell_type("poisson_generator")(rate=0))
                #self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                self.np_inh = self.sheet.sim.Population(
                    len(self.sheet.pop), native_cell_type("poisson_generator")(rate=0)
                )
                self.sheet.sim.Projection(self.np_exc, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')
                #self.np_inh = self.sheet.sim.Population(1, native_cell_type("poisson_generator")(rate=0))
                #self.sheet.sim.Projection(self.np_inh, self.sheet.pop,self.sheet.sim.AllToAllConnector(),synapse_type=inh_syn,receptor_type='inhibitory')
        
        else:
            if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                        self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
                        seeds=mozaik.get_simulation_seeds((self.sheet.pop.size,))
                        self.stgene = [StGen(seed=seeds[i]) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
                        self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory')

            if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                        self.ssai = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
                        seeds=mozaik.get_simulation_seeds((self.sheet.pop.size,))
                        self.stgeni = [StGen(seed=seeds[i]) for i in numpy.nonzero(self.sheet.pop._mask_local)[0]]
                        self.sheet.sim.Projection(self.ssai, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=inh_syn,receptor_type='inhibitory')

    def prepare_stimulation(self,duration,offset):
        if not self.sheet.parameters.mpi_safe:
            for i in range(len(self.np_exc)):
                if self.np_exc._mask_local[i]:
                    self.np_exc[i].set_parameters(rate=self.parameters.exc_firing_rate)
                if self.np_inh._mask_local[i]:
                    self.np_inh[i].set_parameters(rate=self.parameters.inh_firing_rate)
        else:
           if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
                for j,i in enumerate(numpy.nonzero(self.sheet.pop._mask_local)[0]):
                    pp = self.stgene[j].poisson_generator(rate=self.parameters.exc_firing_rate,t_start=0,t_stop=duration).spike_times
                    a = offset + numpy.array(pp)
                    self.ssae[i].set_parameters(spike_times=Sequence(a.astype(float)))
               
           if (self.parameters.inh_firing_rate != 0 or self.parameters.inh_weight != 0):
                for j,i in enumerate(numpy.nonzero(self.sheet.pop._mask_local)[0]):
                    pp = self.stgene[j].poisson_generator(rate=self.parameters.inh_firing_rate,t_start=0,t_stop=duration).spike_times
                    a = offset + numpy.array(pp)
                    self.ssai[i].set_parameters(spike_times=Sequence(a.astype(float)))
        

        
    def inactivate(self,offset):        
        if not self.sheet.parameters.mpi_safe:
            for i in range(len(self.np_exc)):
                if self.np_exc[i]._mask_local:
                    self.np_exc[i].set_parameters(rate=0)
                if self.np_inh[i]._mask_local:
                    self.np_inh[i].set_parameters(rate=0)
            

class Kick(DirectStimulator):
    r"""
    This stimulator sends a kick of excitatory spikes into a specified subpopulation of neurons.
    
    Parameters
    ----------

    parameters : ParameterSet
        The dictionary of required parameters.
                
    sheet : Sheet
        The sheet in which to stimulate neurons.
    
          
    Other parameters
    ----------------
    
    exc_firing_rate : float
        The firing rate of external neurons sending excitatory inputs to each neuron of this sheet.
 
    exc_weight : float
        The weight of the synapses for the excitatory external Poisson input.    
    
    drive_period : float
        Period over which the Kick will deposit the full drive defined by the exc_firing_rate, after this time the 
        firing rates will be linearly reduced to reach zero at the end of stimulation.

    population_selector : ParemeterSet
        Defines the population selector and its parameters to specify to which neurons in the population the 
        background activity should be applied. 

                                        
    Notes
    -----
    
    Currently this experiment does not work with MPI

    """
    
    
    required_parameters = ParameterSet({
            'exc_firing_rate': float,
            'exc_weight': float,
            'drive_period' : float,
            'population_selector' : ParameterSet({
                    'component' : str,
                    'params' : ParameterSet
                    
            })
            
    })

    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        population_selector = load_component(self.parameters.population_selector.component)
        self.ids = population_selector(sheet,self.parameters.population_selector.params).generate_idd_list_of_neurons()
        d = dict((j,i) for i,j in enumerate(self.sheet.pop.all_cells))
        self.to_stimulate_indexes = [d[i] for i in self.ids]

        exc_syn = self.sheet.sim.StaticSynapse(weight=self.parameters.exc_weight,delay=2*self.sheet.model.parameters.min_delay)
        if (self.parameters.exc_firing_rate != 0 or self.parameters.exc_weight != 0):
            self.ssae = self.sheet.sim.Population(self.sheet.pop.size,self.sheet.sim.SpikeSourceArray())
            seeds=mozaik.get_simulation_seeds((self.sheet.pop.size,))
            self.stgene = [StGen(seed=seeds[i]) for i in self.to_stimulate_indexes]
            self.sheet.sim.Projection(self.ssae, self.sheet.pop,self.sheet.sim.OneToOneConnector(),synapse_type=exc_syn,receptor_type='excitatory') 

    def prepare_stimulation(self,duration,offset):

        if (self.parameters.exc_firing_rate != 0 and self.parameters.exc_weight != 0):
           for j,i in enumerate(self.to_stimulate_indexes):
                if self.ssae._mask_local[i]:
                    if self.parameters.drive_period < duration:
                        z = numpy.arange(self.parameters.drive_period+0.001,duration-100,10)
                        times = [0] + z.tolist() 
                        rate = [self.parameters.exc_firing_rate] + ((1.0-numpy.linspace(0,1.0,len(z)))*self.parameters.exc_firing_rate).tolist()
                    else:
                        times = [0]  
                        rate = [self.parameters.exc_firing_rate] 
                    pp = self.stgene[j].inh_poisson_generator(numpy.array(rate),numpy.array(times),t_stop=duration).spike_times
                    a = offset + numpy.array(pp)
                    self.ssae[i].set_parameters(spike_times=Sequence(a.astype(float)))

    def inactivate(self,offset):        
        pass


class Depolarization(DirectStimulator):
    r"""
    This stimulator injects a constant current into neurons in the population.
    
    Parameters
    ----------

    parameters : ParameterSet
        The dictionary of required parameters.
                
    sheet : Sheet
        The sheet in which to stimulate neurons.
    
          
    Other parameters
    ----------------
    
    current : float (mA)
        The current to inject into neurons.

    population_selector : ParemeterSet
        Defines the population selector and its parameters to specify to which neurons in the population the 
        background activity should be applied. 

                                        
    Notes
    -----
    
    Currently the mpi_safe version only works in nest!

    """
    
    
    required_parameters = ParameterSet({
            'current': float,
            'population_selector' : ParameterSet({
                    'component' : str,
                    'params' : ParameterSet
                    
            })
            
    })
        
    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        
        population_selector = load_component(self.parameters.population_selector.component)
        ids = population_selector(sheet,self.parameters.population_selector.params).generate_idd_list_of_neurons()
        d = dict((j,i) for i,j in enumerate(self.sheet.pop.all_cells))
        to_stimulate_indexes = [d[i] for i in ids]
        
        self.scs = self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])
        for i in to_stimulate_indexes:
            self.sheet.pop.all_cells[i].inject(self.scs)

    def prepare_stimulation(self,duration,offset):
        self.scs.set_parameters(times=[offset], amplitudes=[self.parameters.current],copy=False)
        
    def inactivate(self,offset):
        self.scs.set_parameters(times=[offset], amplitudes=[0.0],copy=False)


@jit()
def ChrimsonR_system(y,time,X,sampling_period):
          PhoC1toO1 = 1.0993e-19 * 50
          PhoC2toO2 = 7.1973e-20 * 50
          PhoC1toC2 = 1.936e-21 * 50
          PhoC2toC1 = 1.438e-20 * 50

          O1toC1 = 0.125
          O2toC2 = 0.015
          O2toS  = 0.0001 / 20
          C2toC1 = 1e-7
          StoC1  = 3e-6

          a = int(numpy.floor(time/sampling_period))
          b = time/sampling_period - a

          if a < len(X)-1:
            I = X[a]*(1-b) + b * X[a+1];
          else:
            I = 0

          O1,O2,C1,C2,S = y

          _O1 = - O1toC1 * O1                    + PhoC1toO1 * I * C1
          _O2 = - O2toC2 * O2                    + PhoC2toO2 * I * C2            - O2toS * O2

          _S  = - StoC1 * S + O2toS * O2

          _C1 = O1toC1 * O1    - PhoC1toO1 * I * C1       - PhoC1toC2 * I * C1    + C2toC1 * C2             + PhoC2toC1 * I * C2            + StoC1 * S
          _C2 = O2toC2 * O2    - C2toC1 * C2              - PhoC2toC1 * I * C2    + PhoC1toC2 * I * C1      - PhoC2toO2 * I * C2

          return (_O1,_O2,_C1,_C2,_S)


# Williams ChR2(H134R) model parameter precomputation.
class _ChR2H134RParameters(NamedTuple):
    Gd1: float
    Gd2: float
    Gr: float
    epsilon1: float
    epsilon2: float
    e12dark: float
    e21dark: float
    Ephoton: float
    sigma_retinal: float
    wloss: float
    tauChR2: float

Q10 = ParameterSet(
    {
        "Gd1": 1.97,
        "Gd2": 1.77,
        "epsilon1": 1.46,
        "epsilon2": 2.77,
        "Gr": 2.56,
        "e12dark": 1.10,
        "e21dark": 1.95,
    }
)

# Approximate V1 temperature as 39 °C from macaque occipital cortex
# Hayward et al. (1966), https://doi.org/10.3181/00379727-121-30827
temperature = 39.0
V = -60.0
hc = 1.986446e-25
wavelength = 470.0
temperature_exponent = (temperature - 22.0) / 10.0

ChR2_H134R_parameters = _ChR2H134RParameters(
    Gd1=(0.075 + 0.043 * numpy.tanh((V + 20.0) / -20.0))
    * Q10.Gd1**temperature_exponent,
    Gd2=0.05 * Q10.Gd2**temperature_exponent,
    Gr=0.0000434587
    * numpy.exp(-0.0211539274 * V)
    * Q10.Gr**temperature_exponent,
    epsilon1=0.8535 * Q10.epsilon1**temperature_exponent,
    epsilon2=0.14 * Q10.epsilon2**temperature_exponent,
    e12dark=0.011 * Q10.e12dark**temperature_exponent,
    e21dark=0.008 * Q10.e21dark**temperature_exponent,
    Ephoton=1e9 * hc / wavelength,
    sigma_retinal=12e-20,
    wloss=1.3,
    tauChR2=1.3,
)


# Williams ChR2(H134R): ODE system evaluated by odeint.
@jit()
def ChR2_H134R_system(statevar, t, X, sampling_period):
    """Williams et al. ChR2(H134R) model at -60 mV and 39 degrees Celsius.

    Irradiance ``X`` is in mW/mm^2 and the state order is C1, C2, O1, O2, p.
    Array inputs are held constant over each sampling interval.
    Model variables retain the names used by the ModelDB MATLAB implementation:
    https://modeldb.science/showmodel?model=151549
    """

    parameters = ChR2_H134R_parameters
    index = int(numpy.floor(t / sampling_period))
    Irradiance = X[index] if 0 <= index < len(X) else 0.0

    logphi0 = numpy.log1p(Irradiance / 0.024) if Irradiance > 0.0 else 0.0
    e12 = parameters.e12dark + 0.005 * logphi0
    e21 = parameters.e21dark + 0.004 * logphi0

    flux = 1000.0 * Irradiance / parameters.Ephoton
    F = flux * parameters.sigma_retinal / (parameters.wloss * 1000.0)
    theta = 100.0 * Irradiance
    S0 = 0.5 * (1.0 + numpy.tanh(120.0 * (theta - 0.1)))

    C1, C2, O1, O2, p = statevar
    Fp = F * p
    dC1O1 = parameters.epsilon1 * Fp * C1
    dO1C1 = parameters.Gd1 * O1
    dO1O2 = e12 * O1
    dO2O1 = e21 * O2
    dO2C2 = parameters.Gd2 * O2
    dC2O2 = parameters.epsilon2 * Fp * C2
    dC2C1 = parameters.Gr * C2
    dp = (S0 - p) / parameters.tauChR2
    dC1 = dC2C1 + dO1C1 - dC1O1
    dC2 = dO2C2 - dC2O2 - dC2C1
    dO1 = dC1O1 + dO2O1 - dO1C1 - dO1O2
    dO2 = dC2O2 + dO1O2 - dO2C2 - dO2O1
    return dC1, dC2, dO1, dO2, dp


def _williams_current_density(states):
    """Return Williams ChR2 current density in pA/pF at -60 mV."""

    states = numpy.asarray(states)
    V = -60.0
    A = 10.6408
    B = -14.6408
    C = 42.7671
    Cm = 1.0
    gChR2 = 2.0 / 5.0
    G_ChR2 = gChR2 / Cm
    gamma = 0.1
    O1 = states[..., 2]
    O2 = states[..., 3]
    I_ChR2 = G_ChR2 * (A + B * numpy.exp(-V / C)) * (O1 + gamma * O2)
    return I_ChR2


def _photon_energy_j(wavelength_nm):
    hc = 6.62607015e-34 * 299792458.0  # Planck constant * speed of light, J m
    return hc / (wavelength_nm * 1e-9)


def _mw_per_mm2_to_photons_per_s_per_cm2(irradiance, wavelength_nm):
    return irradiance / (10.0 * _photon_energy_j(wavelength_nm))


def _photons_per_s_per_cm2_to_mw_per_mm2(photon_flux, wavelength_nm):
    return photon_flux * _photon_energy_j(wavelength_nm) * 10.0


class OpticalStimulatorArray(DirectStimulator):   
    r"""
    Simulates an array of optical stimulators delivering light to a cortical sheet.

    The stimulator is a regular 2D grid (defined by `size` and `spacing`). Each
    grid element emits light that spreads through tissue, and contributions from
    all stimulators are summed linearly at each neuron to produce the local light
    intensity (photon flux).

    The input is a spatiotemporal signal provided via `set_input(input_signal)`,
    a 3D array of shape:

        (Nx, Ny, T)

    where Nx and Ny match the stimulator grid, and T is the number of time steps. 
    Each value represents the relative light output of a stimulator at a given time.

    This class computes the resulting light intensity at each neuron. The conversion
    from light to injected current is handled by child classes.

    The temporal resolution is given by `update_interval`, so total stimulation
    duration is:

        T * update_interval

    Parameters
    ----------

    parameters : ParameterSet
        The dictionary of required parameters.

    sheet : Sheet
        The sheet in which to stimulate neurons.

    Other parameters
    ----------------

    size : float (μm) 
        The size of the stimulator grid.

    spacing : float (μm)
        Distance between stimulators.

    update_interval : float (ms)
        Time step of the stimulation signal.

    depth_sampling_step : float (μm)
        Resolution used for approximating neuron depth in light propagation.

    light_source_light_propagation_data : str
        Path to the radial light propagation profile.

    transfection_proportion : float
        Fraction of neurons expressing the opsin (range [0, 1]).

                     
    """
    
    
    required_parameters = ParameterSet({
        'size': float,
        'spacing' : float,
        'update_interval' : float,
        'depth_sampling_step' : float,
        'light_source_light_propagation_data' : str,
        'transfection_proportion' : float,
    })

    def __init__(self, sheet, parameters):
        DirectStimulator.__init__(self, sheet,parameters)
        self.is_active = False
        self.integrated_cs = _supports_integrated_current_schedules(self.sheet)

        assert self.parameters.transfection_proportion >= 0 and self.parameters.transfection_proportion <= 1, "Transfection proportions must be in the range of (0,1)!"

        assert math.fmod(self.parameters.size,self.parameters.spacing) < 0.000000001 , "Error the size has to be multiple of spacing!"
        assert math.fmod(self.parameters.size / self.parameters.spacing /2,2) < 0.000000001 , "Error the size and spacing have to be such that they give odd number of elements!"
        
        axis_coors = numpy.arange(0,self.parameters.size+self.parameters.spacing,self.parameters.spacing) - self.parameters.size/2.0 

        self.n = int(numpy.floor(len(axis_coors)/2.0))
        self.stimulator_coords_y, self.stimulator_coords_x = numpy.meshgrid(axis_coors, axis_coors)

        #let's load up disperssion data and setup interpolation
        f = open(self.parameters.light_source_light_propagation_data,'rb')
        radprofs = pickle.load(f,encoding='latin1')
        f.close()

        light_flux_lookup =  scipy.interpolate.RegularGridInterpolator((np.linspace(0,1080,radprofs.shape[0]),numpy.linspace(0,1,radprofs.shape[1])*299.7*numpy.sqrt(2)), radprofs, method='linear',bounds_error=False,fill_value=0)

        # the constant translating the data in radprofs to photons/s/cm^2
        self.K = 2.97e26
        self.W = 3.9e-10

        # now let's calculate mixing weights, this will be a matrix nxm where n is 
        # the number of neurons in the population and m is the number of stimulators
        x =  self.stimulator_coords_x.flatten()
        y =  self.stimulator_coords_y.flatten()
        # Partition integrated optical calculations across MPI ranks so each rank
        # handles the neurons it owns in PyNN/NEST.
        local_mask = numpy.asarray(self.sheet.pop._mask_local, dtype=bool)
        self.local_population_indices = numpy.flatnonzero(local_mask)
        if self.integrated_cs:
            self.optical_calc_population_indices = self.local_population_indices
        else:
            self.optical_calc_population_indices = numpy.arange(self.sheet.pop.size)

        assert numpy.array_equal(
            self.local_population_indices,
            numpy.array(
                [
                    self.sheet.pop.id_to_index(cell)
                    for cell in self.sheet.pop.local_cells
                ]
            ),
        ), "PyNN local-cell order does not match the population locality mask!"

        positions = self.sheet.pop.positions[:, self.optical_calc_population_indices]
        xx,yy = self.sheet.vf_2_cs(positions[0],positions[1])
        zeros = numpy.zeros(len(x))
        self.mixing_templates=[]
        for depth in numpy.arange(sheet.parameters.min_depth,sheet.parameters.max_depth+self.parameters.depth_sampling_step,self.parameters.depth_sampling_step):
            temp = numpy.reshape(light_flux_lookup(numpy.transpose([zeros+depth,numpy.sqrt(numpy.power(x,2)  + numpy.power(y,2))])),(2*self.n+1,2*self.n+1))
            a  = temp[self.n,self.n:]
            cutof = numpy.argmax((numpy.sum(a)-numpy.cumsum(a))/numpy.sum(a) < 0.01)
            assert numpy.shape(temp[self.n-cutof:self.n+cutof+1,self.n-cutof:self.n+cutof+1]) == (2*cutof+1,2*cutof+1), str(numpy.shape(temp[self.n-cutof:self.n+cutof,self.n-cutof:self.n+cutof])) + 'vs' + str((2*cutof+1,2*cutof+1))
            self.mixing_templates.append((temp[self.n-cutof:self.n+cutof+1,self.n-cutof:self.n+cutof+1],cutof))

        self.nearest_ix = numpy.rint(xx/self.parameters.spacing)+self.n
        self.nearest_iy = numpy.rint(yy/self.parameters.spacing)+self.n
        self.nearest_iz = numpy.rint((numpy.array(positions[2])-sheet.parameters.min_depth)/self.parameters.depth_sampling_step)

        self.nearest_ix[self.nearest_ix<0] = 0
        self.nearest_iy[self.nearest_iy<0] = 0
        self.nearest_ix[self.nearest_ix>2*self.n] = 2*self.n
        self.nearest_iy[self.nearest_iy>2*self.n] = 2*self.n

        # Mask coding which neurons are transfected. We only calculate the
        # ChR ODEs for these neurons.
        self.transfection_mask = np.zeros(self.sheet.pop.size, dtype=bool)
        if self.parameters.transfection_proportion == 1:
            self.transfection_mask[:] = True
        else:
            idx = mozaik.model_rng.choice(
                range(self.sheet.pop.size),
                size=int(
                    self.parameters.transfection_proportion
                    * self.sheet.pop.size
                ),
                replace=False,
            )
            self.transfection_mask[idx] = True

        self.active_cells = np.flatnonzero(
            self.transfection_mask[self.optical_calc_population_indices]
        )
        self.active_population_indices = self.optical_calc_population_indices[
            self.active_cells
        ]
        self.scs = []

        if not self.integrated_cs:
            for i in self.active_population_indices:
                scs = self.sheet.sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])
                self.sheet.pop.all_cells[i].inject(scs)
                self.scs.append(scs)

        # Propagated light is stored in photons/s/cm^2 for every opsin model.
        self.mixed_signals_photo = numpy.zeros(
            (len(self.optical_calc_population_indices),2), dtype=numpy.float64
        )

        
    # A single BLAS worker is faster for these small calculations and keeps
    # results independent of the number of MPI processes.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def calculate_photo(self, input_signal):
        # input_signal is relative source intensity with dimensions
        # space x space x time; the returned propagation result is photons/s/cm^2.
        assert input_signal.shape[:2] == self.stimulator_coords_x.shape, "Spatial dimensions of input signal (%s) and stimulation array (%s) are not equal!" % (input_signal.shape[1:],(self.stimulator_coords_x.shape))
        propagated_photon_flux = numpy.zeros(
            (len(self.optical_calc_population_indices),input_signal.shape[2]),
            dtype=numpy.float64,
        )
        
        # Tibor note: For sure this can be better separated, look into it!
        # find coordinates given spacing and shift by half the array size
        for i in range(0,len(self.optical_calc_population_indices)):
            temp,cutof = self.mixing_templates[int(self.nearest_iz[i])]
            ss = input_signal[max(int(self.nearest_ix[i]-cutof),0):int(self.nearest_ix[i]+cutof+1),max(int(self.nearest_iy[i]-cutof),0):int(self.nearest_iy[i]+cutof+1),:]
            if ss.size != 0:
                temp = temp[max(int(cutof-self.nearest_ix[i]),0):max(int(2*self.n+1+cutof-self.nearest_ix[i]),0),max(int(cutof-self.nearest_iy[i]),0):max(int(2*self.n+1+cutof-self.nearest_iy[i]),0)]
                propagated_photon_flux[i,:] = self.K*self.W*numpy.dot(temp.flatten(),numpy.reshape(ss,(len(temp.flatten()),-1)))
        propagated_photon_flux *= self.transfection_mask[
            self.optical_calc_population_indices, None
        ] # Photon flux is zero for non-transfected cells.
        return propagated_photon_flux

    def set_input(self, input_signal):
        self.stimulation_duration = input_signal.shape[2] * self.parameters.update_interval
        self.mixed_signals_photo = self.calculate_photo(input_signal)

    @staticmethod
    def compress_array(array):
        compressed = io.BytesIO()
        np.savez_compressed(compressed, array)
        return compressed

    @staticmethod
    def decompress_array(array):
        array.seek(0)
        return np.load(array)['arr_0']

    def prepare_stimulation(self,duration,offset):
        self.is_active = True
        assert hasattr(self,"mixed_signals_current"), "Child class has to implement conversion of optical stimulation to current!"
        self.times = numpy.arange(0,self.stimulation_duration,self.parameters.update_interval) + offset

        self._set_stimulation_current_schedule(self.times)

    def _set_stimulation_current_schedule(self, times):
        """Program one current segment on the neurons owned by this rank."""

        if self.integrated_cs:
            amplitude_times = self._integrated_cs_neuron_amplitude_times(times)
            for cell_idx, population_idx in zip(
                self.active_cells, self.active_population_indices
            ):
                cell = self.sheet.pop.all_cells[population_idx]
                assert cell.local, "Integrated current schedules must be set locally!"
                cell.set_parameters(
                    amplitude_times=amplitude_times,
                    amplitude_values=self.mixed_signals_current[
                        cell_idx, :
                    ].flatten()
                    * 1000,
                )
        else:
            for scs_idx, cell_idx in enumerate(self.active_cells):
                self.scs[scs_idx].set_parameters(
                    times=Sequence(times),
                    amplitudes=Sequence(self.mixed_signals_current[cell_idx, :].flatten()),
                    copy=False
                )

    def _integrated_cs_neuron_amplitude_times(self, times):
        times = numpy.array(times, dtype=float, copy=True)
        current_time = self.sheet.sim.get_current_time()
        if current_time > 0 and times[0] < current_time + self.sheet.dt:
            # NEST requires current source updates during a run
            # to start at the next time step.
            times[0] = current_time + self.sheet.dt
            assert (
                len(times) == 1 or times[0] < times[1]
            ), "Integrated optical update intervals must exceed the simulation timestep!"
        return times

    def inactivate(self,offset):
        self.is_active = False
        if self.integrated_cs:
            amplitude_times = self._integrated_cs_neuron_amplitude_times([offset])
            for population_idx in self.active_population_indices:
                cell = self.sheet.pop.all_cells[population_idx]
                assert cell.local, "Integrated current schedules must be set locally!"
                cell.set_parameters(
                    amplitude_times=amplitude_times,
                    amplitude_values=numpy.array([0.0]),
                )
        else:
            for scs in self.scs:
                scs.set_parameters(times=[offset], amplitudes=[0.0],copy=False)




class OpticalStimulatorArrayChR(OpticalStimulatorArray):
    r"""
    Like *OpticalStimulatorArray*, but the light impinging on the neuron is
    transformed via either the Sabatier ChrimsonR model or the Williams ChR2
    model to give the final injected current.

    Note that we approximate the current by ignoring the voltage dependence of the
    channels, as it is very expensive to inject conductance in PyNN. The
    Channelrhodopsin has reverse potential of ~0, and we assume that our neurons
    sits on average at -60mV to calculate the current.
    """
    required_parameters = ParameterSet(
        {
            "channelrhodopsin_model": str,
            "intensity_input_convention": str,
        }
    )

    # Optical transmission factor, modelling dura light absorption, etc.
    T_optical = 1.0
    # Empirical conversion from Williams pA/pF to absolute current. This absorbs
    # effective neuron membrane area/capacitance and ChR2 expression differences;
    # it independent of AdExp membrane capacitance and T_optical.
    effective_current_scaler_pF = 1.0
    _legacy_reference_photon_flux = 3e14 # TODO: Verify and document the exact normalization factor.

    def __init__(self, sheet, parameters):
        self.channelrhodopsin_model = parameters.channelrhodopsin_model.lower()
        assert self.channelrhodopsin_model in ("sabatier", "williams"), (
            "channelrhodopsin_model must be 'Sabatier' or 'Williams'!"
        )
        assert parameters.intensity_input_convention in (
            "legacy",
            "photons/s/cm^2",
            "mW/mm^2",
        ), (
            "intensity_input_convention must be 'legacy', "
            "'photons/s/cm^2', or 'mW/mm^2'!"
        )
        wavelength_label = (
            "590nm" if self.channelrhodopsin_model == "sabatier" else "470nm"
        )
        assert wavelength_label in parameters.light_source_light_propagation_data, (
            "light_source_light_propagation_data must contain '%s' for the %s path!"
            % (wavelength_label, parameters.channelrhodopsin_model)
        )
        OpticalStimulatorArray.__init__(self, sheet,parameters)
        self.mixed_signals_current = np.zeros_like(self.mixed_signals_photo)
        if self.channelrhodopsin_model == "sabatier":
            self.ChR_default_state = np.array([0, 0, 0.2, 0.8, 0])
        else:
            self.ChR_default_state = np.array([1, 0, 0, 0, 0])
        self.ChR_state = np.tile(self.ChR_default_state,(self.mixed_signals_photo.shape[0],1))

    def calculate_photo(self, input_signal):
        convention = self.parameters.intensity_input_convention
        if convention == "legacy":
            relative_source_intensity = input_signal
        elif convention == "photons/s/cm^2":
            relative_source_intensity = (
                input_signal / self._legacy_reference_photon_flux
            )
        else:
            wavelength_nm = (
                590.0 if self.channelrhodopsin_model == "sabatier" else 470.0
            )
            source_photon_flux = _mw_per_mm2_to_photons_per_s_per_cm2(
                input_signal, wavelength_nm
            )
            relative_source_intensity = (
                source_photon_flux / self._legacy_reference_photon_flux
            )
        if self.T_optical != 1.0:
            relative_source_intensity = relative_source_intensity * self.T_optical
        propagated_photon_flux = OpticalStimulatorArray.calculate_photo(
            self, relative_source_intensity
        )
        return propagated_photon_flux

    def calculate_ChR(self, propagated_photon_flux):
        current = np.zeros_like(propagated_photon_flux)
        times = numpy.arange(0,propagated_photon_flux.shape[1] * self.parameters.update_interval,self.parameters.update_interval)
        for i in self.active_cells:
            if self.channelrhodopsin_model == "sabatier":
                res = odeint(ChrimsonR_system,self.ChR_state[i,:],times,args=(propagated_photon_flux[i,:].flatten(),self.parameters.update_interval),hmax=self.parameters.update_interval)
                current[i,:] =  60 * (17.2*res[:,0] + 2.9 * res[:,1])  / 2500 ; # the 60 corresponds to the 60mV difference between ChR reverse potential of 0mV and our expected mean Vm of about 60mV. This happens to end up being in nA which is what pyNN expect for current injection.
            else:
                irradiance_mw_per_mm2 = _photons_per_s_per_cm2_to_mw_per_mm2(
                    propagated_photon_flux[i, :].flatten(), 470.0
                )
                res = odeint(
                    ChR2_H134R_system,
                    self.ChR_state[i, :],
                    times,
                    args=(irradiance_mw_per_mm2, self.parameters.update_interval),
                    hmax=self.parameters.update_interval,
                )
                current[i, :] = (
                    _williams_current_density(res)
                    * self.effective_current_scaler_pF
                    / 1000.0
                )
            self.ChR_state[i,:] = res[-1,:]
        return current

    def set_input(self, input_signal):
        OpticalStimulatorArray.set_input(self,input_signal)
        self.mixed_signals_current = self.calculate_ChR(self.mixed_signals_photo)

    def inactivate(self,offset):
        OpticalStimulatorArray.inactivate(self, offset)
        self.ChR_state[:] = self.ChR_default_state # Reset channelrhodopsin state

class ClosedLoopOpticalStimulatorArray(OpticalStimulatorArrayChR):
    r"""
    Like *OpticalStimulatorArrayChR*, but instead of receiving a predefined input
    signal, the stimulation is generated online based on the activity of neurons
    in the sheet.

    The next stimulation segment is computed slightly before it should start,
    because NEST enforces a small minimum delay between programming a current
    source and injecting that current into cells. The segment start time is
    exposed as :meth:`next_actuation_time`.

    The user-supplied ``input_calculation_function`` returns one update interval
    of dimensionless light intensities with shape
    ``(Nx, Ny, state_update_interval / update_interval)``. The first segment
    starts at time zero; later segments start at the delayed actuation time
    exposed by :meth:`next_actuation_time`.

    In MPI simulations, feedback is collected collectively, but the user-supplied
    state-update and input-calculation functions run only on rank 0. The resulting
    light-array segment is broadcast before each rank calculates and programs the
    currents for its locally owned neurons.

    ``USE_DIRECT_NEST_SPIKE_RETRIEVAL`` temporarily selects whether
    ``last_spike_counts`` is populated through a direct NEST spike recorder or
    whether controllers use the standard PyNN/Mozaik recording data available
    in ``feedback_data``.

    ``actuation_delay`` is the interval in milliseconds between observing the
    network state and starting the resulting stimulation. Set it to ``None`` to
    use the backend minimum. The integrated-current backend requires at least
    ``2 * time_step``. The external ``StepCurrentSource`` backend requires at
    least ``min_delay + 2 * time_step``; ``min_delay`` and ``time_step`` are
    configured in the top-level Mozaik model parameters. Any explicit value must
    meet the selected backend minimum and be a multiple of ``time_step``. The
    effective value is logged on rank 0.

    ``feedback_sheet_names`` is the non-empty list of cortical sheets whose
    configured Mozaik recordings are collected before every controller update.
    Rank 0 receives them in ``feedback_data``, keyed by sheet name.

    Note that, as in *OpticalStimulatorArrayChR*, the current is approximated by
    ignoring the voltage dependence of the channels.
    """
    required_parameters = ParameterSet({
        'size': float,
        'spacing' : float,
        'update_interval' : float,
        'depth_sampling_step' : float,
        'light_source_light_propagation_data' : str,
        'transfection_proportion' : float,
        'state_update_interval': float,
        'actuation_delay': (float, type(None)),
        'feedback_sheet_names': list,
    })

    def __init__(self, sheet, parameters):
        OpticalStimulatorArrayChR.__init__(self, sheet,parameters)
        self.start_time = None
        assert self.parameters.state_update_interval % self.parameters.update_interval == 0
        if self.integrated_cs:
            minimum_actuation_delay = 2 * self.sheet.dt
            backend = "integrated current schedule"
        else:
            current_source_min_delay = (
                self.scs[0].min_delay
                if self.scs
                else self.sheet.model.parameters.min_delay
            )
            minimum_actuation_delay = current_source_min_delay + 2 * self.sheet.dt
            backend = "external StepCurrentSource"

        requested_actuation_delay = self.parameters.actuation_delay
        if requested_actuation_delay is None:
            self._actuation_delay = minimum_actuation_delay
        else:
            self._actuation_delay = requested_actuation_delay
        assert self._actuation_delay >= minimum_actuation_delay, (
            "Closed loop actuation delay %.6g ms is shorter than the %.6g ms "
            "minimum for the %s backend!"
            % (self._actuation_delay, minimum_actuation_delay, backend)
        )
        assert np.isclose(
            self._actuation_delay / self.sheet.dt,
            round(self._actuation_delay / self.sheet.dt),
        ), "Closed loop actuation delay must be a multiple of the simulation timestep!"
        assert (
            self._actuation_delay < self.parameters.state_update_interval
        ), "Closed loop actuation delay must be shorter than the state update interval!"
        if (not mozaik.mpi_comm) or mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
            logger.info(
                "Closed-loop stimulator %s uses the %s backend with an actuation "
                "delay of %.6g ms (minimum %.6g ms)",
                self.sheet.name,
                backend,
                self._actuation_delay,
                minimum_actuation_delay,
            )
        self.calculate_input_function, self.update_state_function, self.state = (
            None,
            None,
            None,
        )
        self.input_signal = None
        self._next_actuation_time = None
        self._last_photon_flux_sample = None
        self._sheet_data_cache = {}
        self.feedback_data = {}
        self.use_direct_nest_spike_retrieval = USE_DIRECT_NEST_SPIKE_RETRIEVAL
        assert self.parameters.feedback_sheet_names, (
            "Closed loop feedback_sheet_names must not be empty!"
        )
        assert len(self.parameters.feedback_sheet_names) == len(
            set(self.parameters.feedback_sheet_names)
        ), "Closed loop feedback_sheet_names must not contain duplicates!"

    @staticmethod
    def _is_controller_rank():
        return (not mozaik.mpi_comm) or (
            mozaik.mpi_comm.rank == mozaik.MPI_ROOT
        )

    @staticmethod
    def _is_mpi_parallel():
        return bool(mozaik.mpi_comm and mozaik.mpi_comm.size > 1)

    # Maybe add calculate input function setter to assert its parameters, etc.?
    def current_time(self): #Maybe make this an attribute and automatically calculate it upon state update?
        if self.start_time is None:
            self.start_time = self.sheet.sim.get_current_time()
        return self.sheet.sim.get_current_time() - self.start_time

    def actuation_delay(self):
        return self._actuation_delay

    def shared_update_schedule(self, stimulators):
        state_update_interval = self.parameters.state_update_interval
        actuation_delay = self.actuation_delay()
        assert all(
            stimulator.parameters.state_update_interval == state_update_interval
            and np.isclose(stimulator.actuation_delay(), actuation_delay)
            for stimulator in stimulators
        ), "All co-active closed loop stimulators must have the same state update interval and actuation delay!"
        return state_update_interval, actuation_delay

    def next_actuation_time(self):
        assert (
            self._next_actuation_time is not None
        ), "Stimulator input has not been initialized!"
        return self._next_actuation_time

    def set_input(self, input_signal):
        # Only called before the experiment
        self._feedback_sheets()
        self.start_time = self.sheet.sim.get_current_time()
        self._next_actuation_time = 0.0
        self._last_photon_flux_sample = None
        self._sheet_data_cache.clear()
        self.feedback_data = {}
        self.stimulation_duration = self.parameters.state_update_interval
        self.set_input_segment()
        self.set_data_recording()

    def set_target_signal(self, target_signal):
        self.target_signal = target_signal

    def _cortical_sheets(self):
        return {
            name: sheet
            for name, sheet in self.sheet.model.sheets.items()
            if callable(getattr(sheet, "vf_2_cs", None))
        }

    def _get_cortical_sheet(self, sheet_name=None):
        if sheet_name is None:
            return self.sheet
        cortical_sheets = self._cortical_sheets()
        assert sheet_name in cortical_sheets, "Unknown cortical sheet: %s" % sheet_name
        return cortical_sheets[sheet_name]

    def _feedback_sheets(self):
        cortical_sheets = self._cortical_sheets()
        unknown_sheets = set(self.parameters.feedback_sheet_names) - set(
            cortical_sheets
        )
        assert not unknown_sheets, "Unknown closed-loop feedback sheets: %s" % sorted(
            unknown_sheets
        )
        return {
            name: cortical_sheets[name]
            for name in self.parameters.feedback_sheet_names
        }

    @staticmethod
    def _time_slice(recording, t_start, t_stop):
        start = recording.t_start if t_start is None else recording.t_start + t_start
        stop = recording.t_stop if t_stop is None else recording.t_start + t_stop
        return recording.time_slice(start, stop)

    def get_recording(self, name, t_start=None, t_stop=None, sheet_name=None):
        allowed_names = ["v", "gsyn_exc", "gsyn_inh", "spikes"]
        assert name in allowed_names, "Requested recording must be one of %s" % str(allowed_names)
        sheet = self._get_cortical_sheet(sheet_name)
        if not self.has_sheet_recorders(sheet) or name not in sheet.to_record:
            return []
        assert sheet.last_recording is not None, "No data have been retrieved for sheet %s!" % sheet.name
        if t_start is not None:
            t_start *= qt.ms
        if t_stop is not None:
            t_stop *= qt.ms
        if name == "spikes":
            if t_start is None and t_stop is None:
                return list(sheet.last_recording.spiketrains)
            return [
                self._time_slice(st, t_start, t_stop)
                for st in sheet.last_recording.spiketrains
            ]
        analogsignals = [
            signal for signal in sheet.last_recording.analogsignals
            if signal.name == name
        ]
        assert len(analogsignals) == 1, "The analogsignal %s is not being recorded!" % name
        analogsignal = analogsignals[0]
        if t_start is None and t_stop is None:
            return np.array(analogsignal)
        return np.array(self._time_slice(analogsignal, t_start, t_stop))

    def has_sheet_recorders(self, sheet=None):
        sheet = self.sheet if sheet is None else sheet
        return bool(getattr(sheet, "to_record", None))

    def recorded_neuron_indices(self,recording_type=None, sheet_name=None):
        sheet = self._get_cortical_sheet(sheet_name)
        if not self.has_sheet_recorders(sheet):
            return list(range(sheet.pop.size))
        if recording_type is None:
            first_recording = next(iter(sheet.to_record.values()))
            assert all(
                v == first_recording for v in sheet.to_record.values()
            ), "If recording_type is None, all recorders must record from the same neurons!"
            recording_type = next(iter(sheet.to_record))
        return sorted(sheet.to_record.get(recording_type, []))

    def recorded_neuron_orientations(self,recording_type=None, sheet_name=None):
        sheet = self._get_cortical_sheet(sheet_name)
        idx = self.recorded_neuron_indices(recording_type, sheet_name)
        if len(idx) == 0:
            return np.array([])
        annotations = sheet.get_neuron_annotations()
        assert all(
            "LGNAfferentOrientation" in annotation for annotation in annotations
        ), "Sheet %s has no LGNAfferentOrientation annotations!" % sheet.name
        return np.array([a['LGNAfferentOrientation'] for a in annotations])[idx]

    def recorded_neuron_positions(self, recording_type=None, sheet_name=None):
        sheet = self._get_cortical_sheet(sheet_name)
        idx = self.recorded_neuron_indices(recording_type, sheet_name)
        positions = sheet.pop.positions
        assert positions.shape[0] == 3, "Closed loop optical stimulation requires a 3D sheet!"
        if len(idx) == 0:
            return np.empty((3, 0))
        positions = positions[:, idx]
        # The stored x and y coordinates are in visual field degrees, while z is in µm.
        x, y = sheet.vf_2_cs(positions[0], positions[1])
        return np.vstack((x, y, positions[2]))

    def recorded_neuron_positions_all_sheets(self, recording_type=None):
        return {
            name: self.recorded_neuron_positions(recording_type, name)
            for name in self._cortical_sheets()
        }

    def recorded_neuron_orientations_all_sheets(self, recording_type=None):
        return {
            name: self.recorded_neuron_orientations(recording_type, name)
            for name in self._cortical_sheets()
        }

    def _retrieve_sheet_data(self, sheet):
        if not self.has_sheet_recorders(sheet):
            return []
        current_time = self.sheet.sim.get_current_time()
        cached = self._sheet_data_cache.get(sheet.name)
        if cached is not None and np.isclose(cached[0], current_time):
            return cached[1]
        data = sheet.get_data(clear=False)
        self._sheet_data_cache[sheet.name] = (current_time, data)
        return data

    def _collect_sheet_data(self):
        """Collect controller-visible PyNN data collectively on every MPI rank."""
        feedback_sheets = self._feedback_sheets()
        if (
            self.use_direct_nest_spike_retrieval
            and list(feedback_sheets) == [self.sheet.name]
            and set(getattr(self.sheet, "to_record", {})) == {"spikes"}
        ):
            self._sheet_data_cache.pop(self.sheet.name, None)
            self.feedback_data = {self.sheet.name: []}
            return

        feedback_data = {}
        for name, sheet in feedback_sheets.items():
            if not self.has_sheet_recorders(sheet):
                data = []
                self._sheet_data_cache[name] = (
                    self.sheet.sim.get_current_time(),
                    data,
                )
            else:
                data = self._retrieve_sheet_data(sheet)
            feedback_data[name] = data
        self.feedback_data = feedback_data

    def get_data_all_sheets(self):
        """Return the feedback data collected before the current callback."""
        return self.feedback_data

    def get_recording_all_sheets(
        self, name, t_start=None, t_stop=None, retrieve=True
    ):
        """Read recordings from the already collected feedback-sheet data."""
        return {
            sheet_name: self.get_recording(
                name, t_start, t_stop, sheet_name=sheet_name
            )
            for sheet_name in self.feedback_data
        }

    def calculate_input_signal(self):
        input_signal = None
        if self._is_controller_rank():
            assert self.calculate_input_function is not None, "Calculate input function not set!"
            input_signal = self.calculate_input_function(self)
        if mozaik.mpi_comm:
            input_signal = mozaik.mpi_comm.bcast(
                input_signal, root=mozaik.MPI_ROOT
            )
        return input_signal

    def _advance_ChR_to_segment_start(self, first_photon_flux_sample):
        if self._last_photon_flux_sample is not None:
            # The controller provides one interval at a time. This private
            # one-step bridge gives ChR the boundary sample it needs without
            # exposing that bookkeeping to controller code.
            self.calculate_ChR(
                np.stack(
                    (self._last_photon_flux_sample, first_photon_flux_sample), axis=1
                )
            )

    def set_input_segment(self):
        # Called at each closed loop iteration
        self.input_signal = self.calculate_input_signal()
        required_samples = int(
            self.parameters.state_update_interval // self.parameters.update_interval
        )
        assert (
            self.input_signal.shape[2] == required_samples
        ), "Closed loop input calculation returned %d samples, but %d are required!" % (
            self.input_signal.shape[2],
            required_samples,
        )
        self.mixed_signals_photo = self.calculate_photo(self.input_signal)
        self._advance_ChR_to_segment_start(self.mixed_signals_photo[:, 0])
        self.mixed_signals_current = self.calculate_ChR(self.mixed_signals_photo)
        self._last_photon_flux_sample = self.mixed_signals_photo[:, -1].copy()

    def set_data_recording(self):
        if not self.use_direct_nest_spike_retrieval:
            self.spike_recorder = None
            return

        if self.has_sheet_recorders() and "spikes" in self.sheet.to_record:
            recorded_indices = self.recorded_neuron_indices("spikes")
        else:
            recorded_indices = self.recorded_neuron_indices()
        self.spike_population_indices = np.asarray(recorded_indices, dtype=int)
        self.nest_ids = np.asarray(
            self.sheet.pop.all_cells[self.spike_population_indices], dtype=int
        )
        self._nest_id_to_spike_index = {
            nest_id: index for index, nest_id in enumerate(self.nest_ids)
        }
        self.spike_counts = np.zeros(len(self.nest_ids))
        self.last_spike_counts = np.zeros(len(self.nest_ids))

        local_mask = np.asarray(self.sheet.pop._mask_local, dtype=bool)[
            self.spike_population_indices
        ]
        self.local_nest_ids = self.nest_ids[local_mask]
        self.spike_recorder = nest.Create("spike_recorder")
        if len(self.local_nest_ids) > 0:
            nest.Connect(self.local_nest_ids.tolist(), self.spike_recorder)

    def _get_spike_counts_directly_from_nest(self):
        events = nest.GetStatus(self.spike_recorder, "events")[0]
        senders = events["senders"]
        local_cumulative_counts = np.zeros(len(self.nest_ids))
        unique_senders, counts = np.unique(senders, return_counts=True)
        for sender, count in zip(unique_senders, counts):
            local_cumulative_counts[
                self._nest_id_to_spike_index[int(sender)]
            ] = count

        if self._is_mpi_parallel():
            cumulative_counts = (
                np.zeros_like(local_cumulative_counts)
                if self._is_controller_rank()
                else None
            )
            mozaik.mpi_comm.Reduce(
                local_cumulative_counts,
                cumulative_counts,
                op=MPI.SUM,
                root=mozaik.MPI_ROOT,
            )
        else:
            cumulative_counts = local_cumulative_counts

        if self._is_controller_rank():
            self.last_spike_counts = cumulative_counts - self.spike_counts
            self.spike_counts = cumulative_counts

    def get_data(self):
        if self.use_direct_nest_spike_retrieval:
            self._get_spike_counts_directly_from_nest()

    # Interface functions
    def update_state(self):
        assert self.update_state_function is not None, "Update state function not set!"
        self._collect_sheet_data()
        self.get_data()
        # At this point the controller has data up to current_time(); its output
        # is scheduled actuation_delay later so NEST can deliver it on time.
        self._next_actuation_time = self.current_time() + self.actuation_delay()
        if self._is_controller_rank():
            self.update_state_function(self)
        self.set_input_segment()
        self.times += self.parameters.state_update_interval
        self._set_stimulation_current_schedule(self.times)

def stimulating_pattern_flash(sheet, coor_x, coor_y, update_interval, parameters):
    r"""
    Stimulation with a stimulation pattern, its exact form is determined
    by the supplied extra parameters. The stimulus turns on at *onset_time*, turns off
    at *offset_time*. The overall duration of the stimulus is *duration* ms.

    Parameters
    ----------

    sheet : Sheet
        The stimulated sheet

    coor_x : numpy array
        X coordinates of all electrodes

    coor_y : numpy array
        Y coordinates of all electrodes

    update_interval : float (ms)
        Timestep in which the stimulator updates

    parameters : Parameters
        Extra parameters for the stimulator functions. They must at minimum
        include the following parameters:

        duration : float (ms)
            Overall stimulus duration

        onset_time : float (ms)
            Time point when the stimulation turns on

        offset_time : float(ms)
            Time point when the stimulation turns off

                        
    """
    signals = numpy.zeros(
        (
            numpy.shape(coor_x)[0],
            numpy.shape(coor_x)[1],
            int(parameters.duration / update_interval),
        )
    )

    t_onset = int(numpy.floor(parameters.onset_time / update_interval))
    t_offset = int(numpy.floor(parameters.offset_time / update_interval))
    stim_length = t_offset - t_onset
    assert t_offset >= t_onset, "offset_time must be greater than onset_time"

    if parameters.shape == "video":
        stim = video_stim(coor_x, coor_y, parameters, stim_length)
        signals[:, :, t_onset:t_offset] = stim
    else:
        mask = generate_2d_stim(sheet, coor_x, coor_y, parameters)
        signals[:, :, t_onset:t_offset] = np.repeat(mask[:, :, np.newaxis], t_offset-t_onset, axis=2)

    return signals

def generate_2d_stim(sheet, coor_x, coor_y, parameters):
    r"""
    Generates a 2d pattern for cortical stimulation.

    Parameters
    ----------

    sheet : Sheet
        The stimulated sheet

    coor_x : numpy array
        X coordinates of all electrodes

    coor_y : numpy array
        Y coordinates of all electrodes

    parameters : Parameters
        Extra parameters for the stimulator functions. They must at minimum
        include the following parameter:

        intensity : float
            Stimulation intensity, going from 0 to 1


    """
    if parameters.shape == "or_map":
        return or_map_mask(sheet, coor_x, coor_y, parameters)
    elif parameters.shape in ["hexagon", "circle","polygon"]:
        return simple_shapes_binary_mask(coor_x, coor_y, parameters.shape, parameters) * parameters.intensity
    elif parameters.shape == "image":
        return image_stim(coor_x, coor_y, parameters)
    else:
        raise ValueError("Unknown shape %s for cortical stimulation!" % parameters.shape)

def video_stim(coor_x, coor_y, parameters, stim_length):
    """
    Generate stimulation in the pattern of a grayscale video, loaded from a .npy
    file containing a 3D numpy array, with values between 0 (black) and 1 (white).

    The mapping between the axes of the numpy array and cortical space
    is 0->X, 1->Y, 2->time.

    If the video has a different aspect ratio or number of pixels as the stimulation
    array, it will be stretched to fit the array.

    Parameters
    ----------
    coor_x : numpy array
                X coordinates of all electrodes

    coor_y : numpy array
                Y coordinates of all electrodes

    parameters : ParameterSet
        intensity : float
                Stimulation intensity constant
        video_path : str
                Path to the .npy file containing the video (3D array)
    stim_length : int
                Number of time steps of the stimulation
    """
    for i in range(coor_x.shape[1]):
        assert np.allclose(coor_x[:, 0], coor_x[:, i]), "X coordinates must be in grid!"
    for i in range(coor_y.shape[0]):
        assert np.allclose(coor_y[0, :], coor_y[i, :]), "Y coordinates must be in grid!"

    A = np.load(parameters.video_path)

    assert len(A.shape) == 3, "The video must be 3D! Instead, the video shape is: %s" % (A.shape)
    assert np.all(A >= 0) and np.all(A <= 1), "All values in the video must be in the range of (0,1)!"

    n_frames = A.shape[2]
    A_interp = np.zeros((coor_x.shape[0], coor_x.shape[1], n_frames))

    x_axis = np.linspace(np.min(coor_x), np.max(coor_x), A.shape[0])
    y_axis = np.linspace(np.min(coor_y), np.max(coor_y), A.shape[1])
    points = np.vstack([coor_x.ravel(), coor_y.ravel()]).T
    
    for t in range(n_frames):
        interp = scipy.interpolate.RegularGridInterpolator((x_axis, y_axis), A[:, :, t], bounds_error=False, fill_value=np.nan)
        frame_interp = interp(points).reshape(coor_x.shape)
        A_interp[:, :, t] = frame_interp
    A_interp *= parameters.intensity

    if n_frames == stim_length:
        return A_interp

    resampled = np.zeros((coor_x.shape[0], coor_x.shape[1], stim_length))
    frame_indices = np.linspace(0, n_frames - 1, stim_length)
    for t in range(stim_length):
        idx = min(int(np.round(frame_indices[t])), n_frames - 1)
        resampled[:, :, t] = A_interp[:, :, idx]
    
    return resampled

def image_stim(coor_x, coor_y, parameters):
    r"""
    Generate stimulation in the pattern of a grayscale image, loaded from a .npy
    file containing a 2D numpy array, with values between 0 (black) and 1 (white).

    The mapping between the axes of the numpy array and cortical space
    is 0->X, 1->Y.

    If the image has a different aspect ratio or number of pixels as the stimulation
    array, it will be stretched to fit the array.

    Parameters
    ----------

    coor_x : numpy array
        X coordinates of all electrodes

    coor_y : numpy array
        Y coordinates of all electrodes

    parameters : ParameterSet
        intensity : float
            Stimulation intensity constant
        image_path : str
            Path to the .npy file containing the image (2D array).

                
    """
    for i in range(coor_x.shape[1]):
        assert np.allclose(coor_x[:, 0], coor_x[:, i]), "X coordinates must be in grid!"
    for i in range(coor_y.shape[0]):
        assert np.allclose(coor_y[0, :], coor_y[i, :]), "Y coordinates must be in grid!"

    A = np.load(parameters.image_path)
    assert len(A.shape) == 2, "The image must be 2D! Instead, the image shape is: %s" % (A.shape)
    assert np.all(A >= 0) and np.all(A <= 1), "All values in the image must be in the range of (0,1)!"

    A_interp = scipy.interpolate.RegularGridInterpolator(
        (np.linspace(np.min(coor_x), np.max(coor_x), A.shape[0]),
         np.linspace(np.min(coor_y), np.max(coor_y), A.shape[1])),
        A, bounds_error=False, fill_value=np.nan)(np.vstack([coor_x.ravel(), coor_y.ravel()]).T)
    A_interp = A_interp.reshape(coor_x.shape)

    return A_interp * parameters.intensity


def or_map_mask(sheet,coor_x,coor_y,parameters):
    r"""
    Stimulating pattern based on the cortical orientation map, where one orientation
    is selected as the primary orientation to maximally stimulate (with *intensity*
    intensity), and the stimulation intensity for the other orientations falls off as a
    Gaussian with the circular distance from the selected orientation:

    Stimulation intensity = intensity * e^(-0.5*d^2/sharpness)
    d = circular_dist(selected_orientation-or_map_orientation)

    Parameters
    ----------

    sheet : Sheet
        Sheet to retrieve neuron orientations (proxy for orientation map) from

    coor_x : numpy array
        X coordinates of all electrodes

    coor_y : numpy array
        Y coordinates of all electrodes

    parameters : ParameterSet
        intensity : float
            Stimulation intensity constant
        sharpness : float
            Variance of the Gaussian falloff
        orientation : float
            Selected orientation to stimulate

                            
    """
    z = sheet.pop.all_cells.astype(int)
    vals = numpy.array([sheet.get_neuron_annotation(i,'LGNAfferentOrientation') for i in range(0,len(z))])
    px,py = sheet.vf_2_cs(sheet.pop.positions[0],sheet.pop.positions[1])
    ors = scipy.interpolate.griddata(list(zip(px,py)), vals, (coor_x, coor_y), method='nearest')

    return parameters.intensity*np.exp(-0.5*np.power(circular_dist(parameters.orientation,ors,np.pi),2)/parameters.sharpness)


def simple_shapes_binary_mask(coor_x, coor_y, shape, parameters):
    r"""
    Generate a stimulation pattern of one or more simple shapes of the same type, in the
    form of a binary mask. The list of coordinates, *coords* defines the number and
    centers of these shapes.

    All coordinates and lengths should be interpreted as cortical coordinates in μm!

    Currently three types of shapes are supported:

    polygon: *points* defines the coordinates of the polygon verices, compared to the
        current coordinate in *coords*. These points can be rotated by *angle*, if
        specified.

    circle:  *radius* defines the radii of circles, with *coords* centers

    hexagon: *radius* defines the radii (or edge length) of hexagons, *coords* their
        center coordinates, and *angle* their rotation (at 0 angle they are
        rotated such that the top and bottom edges are horizontal)

             
    Parameters
    ----------

    coor_x : numpy array
        X coordinates of all electrodes

    coor_y : numpy array
        Y coordinates of all electrodes

    shape : str
        polygon, circle or hexagon

    parameters : ParameterSet
        coords : list((x,y))
            Coordinates of the centers of the shapes to draw
        points : list((x,y))
            Polygon coordinates compared to their center (coords[i])
        radius : float
            Circle or hexagon radius
        angle : float
            Hexagon or polygon rotation
        inverted : bool
            Inverts the pattern if True. Defaults to False if not included.

                
    """
    known_shapes = ["polygon","circle","hexagon"]
    assert shape in known_shapes, "Shape %s not among known shapes: %s" % (shape, known_shapes)

    if "angle" not in parameters:
        parameters.angle = 0
    if "inverted" not in parameters:
        parameters.inverted = False
    if type(parameters.coords[0]) != list and type(parameters.coords[0]) != tuple:
        parameters.coords = [parameters.coords]
    if shape == "polygon":
        points = np.array(parameters.points)
        # Calculate center of mass
        parameters.coords = [(points.T[0].mean(), points.T[1].mean())]
        points = points - np.array(parameters.coords[0])
    elif shape == "hexagon":
        points = (
            np.array(
                [
                    [0, 1],
                    [np.sqrt(3) / 2, 0.5],
                    [np.sqrt(3) / 2, -0.5],
                    [0, -1],
                    [-np.sqrt(3) / 2, -0.5],
                    [-np.sqrt(3) / 2, 0.5],
                ]
            )
            * parameters.radius
        )

    mask = np.full(coor_x.shape, False)
    for x_c, y_c in parameters.coords:
        if shape == "polygon" or shape == "hexagon":
            path = matplotlib.path.Path(points)
        if shape == "circle":
            path = matplotlib.path.Path.circle(radius=parameters.radius)

        coords = np.hstack((coor_x.reshape(-1, 1), coor_y.reshape(-1, 1)))

        transform = (
            matplotlib.transforms.Affine2D()
            .translate(x_c, y_c)
            .rotate_around(x_c, y_c, parameters.angle)
        )
        mask_ = path.contains_points(coords, transform=transform, radius=-0.0001)
        mask_ = mask_.reshape(coor_x.shape[0], coor_x.shape[1])
        mask = np.logical_or(mask, mask_)

    if parameters.inverted:
        mask = np.logical_not(mask)
    return mask

def single_pixel(sheet, coor_x, coor_y, update_interval, parameters):
    r"""
    A simple stimulation pattern where for the entire duration a single stimulator
    pixel is active (with an intensity of 1), all others have a value of 0.

    Parameters
    ----------

    coor_x : numpy array
        X coordinates of all electrodes

    coor_y : numpy array
        Y coordinates of all electrodes

    update_interval : float (ms)
        Timestep in which the stimulator updates

    parameters : ParameterSet
        x : int
            x position of the lit up pixel.

        y : int
            y position of the lit up pixel.

        duration : float (ms)
            Overall stimulus duration

            
    Only used in a for testing.
    
    """
    x, y = parameters["x"], parameters["y"]
    assert x in coor_x and y in coor_y
    signals = np.zeros(
        (
            np.shape(coor_x)[0],
            np.shape(coor_x)[1],
            int(parameters.duration / update_interval),
        )
    )
    signals[np.where(x == coor_x)[0][0], np.where(y == coor_y)[1][0], :] = 1
    return signals
