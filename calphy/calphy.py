from dataclasses import dataclass, asdict, field
from typing import Optional, Union, Tuple, List
import numpy as np
import os
import random
import string
from pyiron_workflow import as_function_node, as_macro_node, as_dataclass_node
from ase import Atoms
import pandas as pd

@dataclass
class MD:
    """
    Molecular dynamics parameters.

    Attributes:
    -----------
    timestep: float
        https://calphy.org/en/latest/inputfile.html#timestep
    n_small_steps: int
        https://calphy.org/en/latest/inputfile.html#n-small-steps
    n_every_steps: int
        https://calphy.org/en/latest/inputfile.html#n-every-steps
    n_repeat_steps: int
        https://calphy.org/en/latest/inputfile.html#n-repeat-steps
    n_cycles: int
        https://calphy.org/en/latest/inputfile.html#n-cycles
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#barostat-damping
    """
    timestep: float = 0.001
    n_small_steps: int = 10000
    n_every_steps: int = 10
    n_repeat_steps: int = 10
    n_cycles: int = 100
    thermostat_damping: float = 0.5
    barostat_damping: float = 0.1

@dataclass
class Tolerance:
    """
    Tolerance parameters.

    Attributes:
    -----------
    spring_constant: float
        https://calphy.org/en/latest/inputfile.html#tol-spring-constant
    solid_fraction: float
        https://calphy.org/en/latest/inputfile.html#tol-solid-fraction
    liquid_fraction: float
        https://calphy.org/en/latest/inputfile.html#tol-liquid-fraction
    pressure: float
        https://calphy.org/en/latest/inputfile.html#tol-pressure
    """
    spring_constant: float = 0.01
    solid_fraction: float = 0.7
    liquid_fraction: float = 0.05
    pressure: float = 1

@dataclass
class NoseHoover:
    """
    Nose-Hoover parameters.

    Attributes:
    -----------
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#nose-hoover-thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#nose-hoover-barostat-damping
    """
    thermostat_damping: float = 0.1
    barostat_damping: float = 0.1

@dataclass
class Berendsen:
    """
    Berendsen parameters.

    Attributes:
    -----------
    thermostat_damping: float
        https://calphy.org/en/latest/inputfile.html#berendsen-thermostat-damping
    barostat_damping: float
        https://calphy.org/en/latest/inputfile.html#berendsen-barostat-damping
    """
    thermostat_damping: float = 100.0
    barostat_damping: float = 100.0

@dataclass
class Queue:
    """
    Queue parameters.
    """
    cores: int = 1

@as_dataclass_node
@dataclass
class InputClass:
    """
    Input parameters for calphy calculations.

    Attributes:
    -----------
    md: MD
        Molecular dynamics parameters.
    tolerance: Tolerance
        Tolerance parameters.
    nose_hoover: NoseHoover
        Nose-Hoover parameters.
    berendsen: Berendsen
        Berendsen parameters.
    queue: Queue
        Queue parameters.
    pressure: int
        https://calphy.org/en/latest/inputfile.html#pressure
    temperature: int
        https://calphy.org/en/latest/inputfile.html#temperature
    npt: bool
        https://calphy.org/en/latest/inputfile.html#npt
    n_equilibration_steps: int
        https://calphy.org/en/latest/inputfile.html#n-equilibration-steps
    n_switching_steps: int
        https://calphy.org/en/latest/inputfile.html#n-switching-steps
    n_print_steps: int
        https://calphy.org/en/latest/inputfile.html#n-print-steps
    n_iterations: int
        https://calphy.org/en/latest/inputfile.html#n-iterations
    equilibration_control: str
        https://calphy.org/en/latest/inputfile.html#equilibration-control
    melting_cycle: bool
        https://calphy.org/en/latest/inputfile.html#melting-cycle
    spring_constants: Optional[float]
        https://calphy.org/en/latest/inputfile.html#spring-constants        
    """
    md: MD = field(default_factory=MD)
    tolerance: Tolerance = field(default_factory=Tolerance)
    nose_hoover: NoseHoover = field(default_factory=NoseHoover)
    berendsen: Berendsen = field(default_factory=Berendsen)
    queue: Queue = field(default_factory=Queue)
    pressure: int = 0
    temperature: Union[int, List[int]] = 300
    npt: bool = True
    n_equilibration_steps: int = 2500
    n_switching_steps: int = 2500
    n_print_steps: int = 1000
    n_iterations: int = 1
    equilibration_control: str = "nose_hoover"
    melting_cycle: bool = True
    reference_phase: Optional[str] = None
    mode: Optional[str] = None
    spring_constants: Optional[float] = None

def _generate_random_string(length: str) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def _prepare_potential_and_structure(potential, structure):
    from pyiron_atomistics.lammps.potential import LammpsPotential, LammpsPotentialFile
    from pyiron_atomistics.lammps.structure import (
        LammpsStructure,
        UnfoldingPrism,
        structure_to_lammps,
    ) 

    potential_df = LammpsPotentialFile().find_by_name(potential)
    potential = LammpsPotential()
    potential.df = potential_df

    pair_style = []
    pair_coeff = []
    
    pair_style.append(" ".join(potential.df["Config"].to_list()[0][0].strip().split()[1:]))
    pair_coeff.append(" ".join(potential.df["Config"].to_list()[0][1].strip().split()[1:]))

    #now prepare the list of elements
    elements = potential.get_element_lst()
    elements_from_pot = potential.get_element_lst()

    lmp_structure = LammpsStructure()
    lmp_structure.potential = potential
    lmp_structure.atom_type = "atomic"
    lmp_structure.el_eam_lst = potential.get_element_lst()
    lmp_structure.structure = structure_to_lammps(structure)

    elements_object_lst = structure.get_species_objects()
    elements_struct_lst = structure.get_species_symbols()

    masses = []
    for element_name in elements_from_pot:
        if element_name in elements_struct_lst:
            index = list(elements_struct_lst).index(element_name)
            masses.append(elements_object_lst[index].AtomicMass)
        else:
            masses.append(1.0)

    file_name = os.path.join(os.getcwd(), _generate_random_string(7)+'.dat')
    lmp_structure.write_file(file_name=file_name)
    potential.copy_pot_files(os.getcwd())
    return pair_style, pair_coeff, elements, masses, file_name

def _prepare_input(inp, potential, structure, mode='fe', reference_phase='solid'):
    from calphy.input import Calculation
    pair_style, pair_coeff, elements, masses, file_name = _prepare_potential_and_structure(potential, structure)
    inpdict = asdict(inp)
    inpdict["pair_style"] = pair_style
    inpdict["pair_coeff"] = pair_coeff
    inpdict["element"] = elements
    inpdict["mass"] = masses
    inpdict['mode'] = mode
    inpdict['reference_phase'] = reference_phase
    inpdict['lattice'] = file_name    
    calc = Calculation(**inpdict)
    return calc

@as_function_node('free_energy')
def SolidFreeEnergy(inp: InputClass, structure: Atoms, potential: str) -> float:
    """
    Calculate the free energy of a solid phase.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.
    
    Returns:
    --------
    float
        Free energy in eV/atom
    """
    from calphy.solid import Solid
    from calphy.routines import routine_fe
    
    calc = _prepare_input(inp, potential, structure, mode='fe', reference_phase='solid')
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    #run calculation
    return job.report.fe

@as_function_node('free_energy')
def LiquidFreeEnergy(inp: InputClass, structure: Atoms, potential: str) -> float:
    """
    Calculate the free energy of a liquid phase.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.
    
    Returns:
    --------
    float
        Free energy in eV/atom
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_fe
    
    calc = _prepare_input(inp, potential, structure, mode='fe', reference_phase='liquid')
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    #run calculation
    return job.report.fe

@as_function_node('temperature', 'free_energy')
def SolidFreeEnergyWithTemperature(inp: InputClass, structure: Atoms, potential: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the free energy of a solid phase as a function of temperature.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Temperature and free energy in K and eV/atom, respectively.
    """
    from calphy.solid import Solid
    from calphy.routines import routine_ts
    
    calc = _prepare_input(inp, potential, structure, mode='ts', reference_phase='solid')
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job)
    #run calculation

    #grab the results
    datafile = os.path.join(os.getcwd(), simfolder, 'temperature_sweep.dat')
    t, f = np.loadtxt(datafile, unpack=True, usecols=(0,1))
    return t, f

@as_function_node('temperature', 'free_energy')
def LiquidFreeEnergyWithTemperature(inp: InputClass, structure: Atoms, potential: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the free energy of a liquid phase as a function of temperature.

    Parameters:
    -----------
    inp: InputClass
        Input parameters for calphy calculations.
    structure: Atoms
        Atomic structure.
    potential: str
        Potential name.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Temperature and free energy in K and eV/atom, respectively.
    """
    from calphy.liquid import Liquid
    from calphy.routines import routine_ts
    
    calc = _prepare_input(inp, potential, structure, mode='ts', reference_phase='liquid')
    simfolder = calc.create_folders()
    job = Liquid(calculation=calc, simfolder=simfolder)
    job = routine_ts(job)
    
    #grab the results
    datafile = os.path.join(os.getcwd(), simfolder, 'temperature_sweep.dat')
    t, f = np.loadtxt(datafile, unpack=True, usecols=(0,1))
    return t, f

@as_function_node('phase_transition_temperature')
def CalculatePhaseTransformationTemperature(t1: np.ndarray, f1: np.ndarray, t2: np.ndarray, f2: np.ndarray, fit_order: int = 4, plot: bool =True) -> float:
    """
    Calculate the phase transformation temperature from free energy data.

    Parameters:
    -----------
    t1: np.ndarray
        Temperature array for phase 1.
    f1: np.ndarray
        Free energy array for phase 1.
    t2: np.ndarray
        Temperature array for phase 2.
    f2: np.ndarray
        Free energy array for phase 2.
    fit_order: int
        Order of the polynomial fit.
    plot: bool
        Plot the free energy data and the fit.
    
    Returns:
    --------
    float
        Phase transformation temperature
    """
    import matplotlib.pyplot as plt

    #do some fitting to determine temps
    t1min = np.min(t1)
    t2min = np.min(t2)
    t1max = np.max(t1)
    t2max = np.max(t2)

    tmin = np.min([t1min, t2min])
    tmax = np.max([t1max, t2max])

    #warn about extrapolation
    if not t1min == t2min:
        warnings.warn(f'free energy is being extrapolated!')
    if not t1max == t2max:
        warnings.warn(f'free energy is being extrapolated!')

    #now fit
    f1fit = np.polyfit(t1, f1, fit_order)
    f2fit = np.polyfit(t2, f2, fit_order)

    #reevaluate over the new range
    fit_t = np.arange(tmin, tmax+1, 1)
    fit_f1 = np.polyval(f1fit, fit_t)
    fit_f2 = np.polyval(f2fit, fit_t)

    #now evaluate the intersection temp
    arg = np.argsort(np.abs(fit_f1-fit_f2))[0]
    transition_temp = fit_t[arg]

    #warn if the temperature is shady
    if np.abs(transition_temp-tmin) < 1E-3:
        warnings.warn('It is likely there is no intersection of free energies')
    elif np.abs(transition_temp-tmax) < 1E-3:
        warnings.warn('It is likely there is no intersection of free energies')

    #plot
    if plot:
        c1lo = '#ef9a9a'
        c1hi = '#b71c1c'
        c2lo = '#90caf9'
        c2hi = '#0d47a1'

        plt.plot(fit_t, fit_f1, color=c1lo, label=f'data1 fit')
        plt.plot(fit_t, fit_f2, color=c2lo, label=f'data2 fit')
        plt.plot(t1, f1, color=c1hi, label='data1', ls='dashed')
        plt.plot(t2, f2, color=c2hi, label='data2', ls='dashed')
        plt.axvline(transition_temp, ls='dashed', c='#37474f')
        plt.ylabel('Free energy (eV/atom)')
        plt.xlabel('Temperature (K)')
        plt.legend(frameon=False)
    return transition_temp

@as_function_node('results')
def CollectResults() -> pd.DataFrame:
    from calphy.postprocessing import gather_results
    df = gather_results('.')
    return df
