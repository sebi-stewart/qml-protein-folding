import pyrosetta

from benchmark import full_bpti_benchmark
from energy_calculation import calculate_and_compare_energies
from misc import init_basic_params, default_qaoa_params, BasicParams, QAOAParams
from rotamer_extraction import extract_top_n_rotamers
from h_ising_creation import build_ising_hamiltonian, extract_and_reduce_tensors
from initialisation import initialize_rosetta
from custom_qaoa import qaoa_func_generator, run_qaoa
from h_mixer import custom_xy_mixer_layer

from validation import validate_conformations

if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    # Pyrosetta Relevant Code
    benchmark_pose = full_bpti_benchmark()
    residue_library, ig, rot_sets, scorefxn = extract_top_n_rotamers(benchmark_pose)

    # Generating QUBO (Quadratic Unconstrained Binary Optimisation) Model, and then reduce it
    h_linear, J_quadratic, global_offset = extract_and_reduce_tensors(residue_library, ig)
    basic_params: BasicParams = init_basic_params(h_linear)
    qaoa_params: QAOAParams = default_qaoa_params()

    # Generate the actual observable and running functions we will use in the QAOA Algorithm
    H_ising = build_ising_hamiltonian(h_linear, J_quadratic)
    cost_function, sample_function = qaoa_func_generator(H_ising, custom_xy_mixer_layer, basic_params)

    # Run the Quantum Approximate Optimisation Algorithm and sample the final parameters
    final_params = run_qaoa(cost_function, qaoa_params)
    probabilities = sample_function(final_params)

    # Extract the top 100 most probably conformations and check that exactly 1 rotamer for each residue is selected
    valid_conformations = validate_conformations(probabilities, basic_params)

    # Calculate both the quantum and pyrosetta energies for comparison
    calculate_and_compare_energies(valid_conformations,
                                   h_linear, J_quadratic, global_offset,
                                   benchmark_pose, scorefxn, residue_library,
                                   basic_params)