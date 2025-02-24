from dataclasses import dataclass, asdict
from typing import Optional, Union
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
    md: Optional[MD] = None
    tolerance: Optional[Tolerance] = None
    nose_hoover: Optional[NoseHoover] = None
    berendsen: Optional[Berendsen] = None
    queue: Optional[Queue] = None
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
    
    def __post_init__(self):
        self.md = MD()
        self.tolerance = Tolerance()
        self.nose_hoover = NoseHoover()
        self.berendsen = Berendsen()
        self.queue = Queue()


def _generate_random_string(length):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

@as_function_node()
def SolidFreeEnergy(inp, structure, potential):
    from calphy.solid import Solid
    from calphy.routines import routine_fe
    from calphy.input import Calculation
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

    inpdict = asdict(inp)
    inpdict["pair_style"] = pair_style
    inpdict["pair_coeff"] = pair_coeff
    inpdict["element"] = elements
    inpdict["mass"] = masses
    inpdict['mode'] = 'fe'
    inpdict['reference_phase'] = 'solid'
    
    #write structure
    file_name = os.path.join(os.getcwd(), _generate_random_string(7)+'.dat')
    inpdict['lattice'] = file_name
    
    lmp_structure.write_file(file_name=file_name)
    potential.copy_pot_files(os.getcwd())

    calc = Calculation(**inpdict)

    simfolder = calc.create_folders()
    job = Solid(calculation=calc, simfolder=simfolder)
    job = routine_fe(job)
    #run calculation
    return job
