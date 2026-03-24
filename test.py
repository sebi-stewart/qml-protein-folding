import pyrosetta
import numpy as np

from benchmark import bpti_ryfyn_benchmark, full_bpti_benchmark
from misc import init_generator_params
from rotamer_extraction import extract_top_n_rotamers, TrackedResidue
from h_ising_creation import extract_hamiltonian_tensors, build_ising_hamiltonian, reduce_hamiltonian
from initialisation import initialize_rosetta
from custom_qaoa import qaoa_func_generator, run_qaoa
from h_mixer import custom_xy_mixer_layer

from constants import *
from validation import validate_conformations

initialize_rosetta(pyrosetta, extra_flags="-mute all")


benchmark_pose = full_bpti_benchmark()
rotamer_lib, ig, rot_sets, scorefxn = extract_top_n_rotamers(benchmark_pose)

for key in rotamer_lib:
    print(key, len(rotamer_lib[key].rotamers))
    print("\t", rotamer_lib[key])
    print("=================================")

# Pyrosetta Relevant Code
benchmark_pose = full_bpti_benchmark()
rotamer_lib, ig, rot_sets, scorefxn = extract_top_n_rotamers(benchmark_pose)

# Generating QUBO (Quadratic Unconstrained Binary Optimisation) Model, and then reduce it
h_linear, J_quadratic = extract_hamiltonian_tensors(rotamer_lib, ig)
h_flex_linear, J_flex_quadratic, global_offset = reduce_hamiltonian(h_linear, J_quadratic, rotamer_lib)
generator_params = init_generator_params(h_flex_linear)
for idx in h_linear:
    print(h_linear[idx])
    print(h_flex_linear.get(idx, "None"))
    print("\n---------------------------------------\n")
