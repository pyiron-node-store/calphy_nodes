from dataclasses import dataclass, asdict, field
from typing import Optional, Union
import numpy as np
import os
import random
import string
from pyiron_workflow import as_function_node, as_macro_node

@dataclass
class MD:
    timestep: float = 0.001
    n_small_steps: int = 10000
    n_every_steps: int = 10
    n_repeat_steps: int = 10
    n_cycles: int = 100
    thermostat_damping: float = 0.5
    barostat_damping: float = 0.1

@dataclass
class Tolerance:
    lattice_constant: float = 0.0002
    spring_constant: float = 0.01
    solid_fraction: float = 0.7
    liquid_fraction: float = 0.05
    pressure: float = 0.5

@dataclass
class NoseHoover:
    thermostat_damping: float = 0.1
    barostat_damping: float = 0.1

@dataclass
class Berendsen:
    thermostat_damping: float = 100.0
    barostat_damping: float = 100.0

@dataclass
class Queue:
    cores: int = 1

@dataclass
class InputClass:
    md: MD = field(default_factory=MD)
    tolerance: Tolerance = field(default_factory=Tolerance)
    nose_hoover: NoseHoover = field(default_factory=NoseHoover)
    berendsen: Berendsen = field(default_factory=Berendsen)
    queue: Queue = field(default_factory=Queue)
    pressure: int = 0
    temperature: int = 0
    npt: bool = True
    n_equilibration_steps: int = 15000
    n_switching_steps: int = 25000
    n_print_steps: int = 1000
    n_iterations: int = 1
    equilibration_control: str = "nose_hoover"
    melting_cycle: bool = True
    reference_phase: Optional[str] = None
    mode: Optional[str] = None
    spring_constants: Optional[float] = None

def _generate_random_string(length):
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
def SolidFreeEnergy(inp, structure, potential):
    from calphy.solid import Solid
    from calphy.routines import routine_fe
    
    calc = _prepare_input(inp, potential, structure, mode='fe', reference_phase='solid')
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    #run calculation
    return job.report.fe

@as_function_node('free_energy')
def LiquidFreeEnergy(inp, structure, potential):
    from calphy.solid import Solid
    from calphy.routines import routine_fe
    
    calc = _prepare_input(inp, potential, structure, mode='fe', reference_phase='liquid')
    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    #run calculation
    return job.report.fe

@as_function_node('temperature', 'free_energy')
def SolidFreeEnergyWithTemperature(inp, structure, potential):
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
def LiquidFreeEnergyWithTemperature(inp, structure, potential):
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
def CalculatePhaseTransformationTemperature(t1, f1, t2, f2, fit_order=4, plot=True):
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
def CollectResults():
    from calphy.postprocessing import gather_results
    df = gather_results('.')
    return df
