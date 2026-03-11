import pyrosetta
import numpy as np

from benchmark import bpti_ryfyn_benchmark
from energy_calculation import evaluate_quantum_energies, evaluate_pyrosetta_energies, compare_energies
from misc import init_generator_params
from rotamer_extraction import extract_top_n_rotamers
from h_ising_creation import extract_hamiltonian_tensors, build_ising_hamiltonian, reduce_hamiltonian
from initialisation import initialize_rosetta
from custom_qaoa import qaoa_func_generator, run_qaoa
from h_mixer import custom_xy_mixer_layer

from constants import *
from validation import validate_conformations

if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    # Pyrosetta Relevant Code
    benchmark_pose = bpti_ryfyn_benchmark()
    rotamer_lib, ig, rot_sets, scorefxn = extract_top_n_rotamers(benchmark_pose)

    # Generating QUBO (Quadratic Unconstrained Binary Optimisation) Model, and then reduce it
    h_linear, J_quadratic = extract_hamiltonian_tensors(rotamer_lib, ig, rot_sets)
    h_flex_linear, J_flex_quadratic, global_offset = reduce_hamiltonian(h_linear, J_quadratic, rotamer_lib)
    generator_params = init_generator_params(h_flex_linear)
    for idx in h_linear:
        print(h_linear[idx])
        print(h_flex_linear.get(idx, "None"))
        print("\n---------------------------------------\n")

    # Generate the actual observable and running functions we will use in the QAOA Algorithm
    H_ising = build_ising_hamiltonian(h_flex_linear, J_flex_quadratic, global_offset, penalty=0.0)
    cost_function, sample_function = qaoa_func_generator(H_ising, custom_xy_mixer_layer, generator_params)

    # Run the Quantum Approximate Optimisation Algorithm and sample the final parameters
    final_params = run_qaoa(cost_function)
    probabilities = sample_function(final_params)

    # Extract the top 100 most probably conformations and check that exactly 1 rotamer for each residue is selected
    top_indices = list(np.argsort(probabilities)[-TOP_CONFORMATION_COUNTS:][::-1])
    valid_conformations = validate_conformations(top_indices, probabilities, generator_params)

    # Calculate both the quantum and pyrosetta energies for comparison
    evaluate_quantum_energies(valid_conformations, h_flex_linear, J_flex_quadratic, global_offset, params=generator_params)
    evaluate_pyrosetta_energies()
    compare_energies()