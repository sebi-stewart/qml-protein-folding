import pyrosetta
import pennylane as qml
import pennylane.numpy as np

from benchmark import bpti_ryfyn_benchmark
from rotamer_extraction import extract_top_n_rotamers
from h_ising_creation import extract_hamiltonian_tensors, build_ising_hamiltonian, reduce_hamiltonian
from initialisation import initialize_rosetta
from custom_qaoa import qaoa_func_generator
from h_mixer import custom_xy_mixer_layer

from constants import *
from validation import validate_conformations

if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    benchmark_pose = bpti_ryfyn_benchmark()
    rotamer_lib, ig, rot_sets = extract_top_n_rotamers(benchmark_pose)

    h_linear, J_quadratic = extract_hamiltonian_tensors(rotamer_lib, ig, rot_sets)
    h_flex_linear, J_flex_quadratic, global_offset = reduce_hamiltonian(h_linear, J_quadratic, rotamer_lib)
    for idx in h_linear:
        print(h_linear[idx])
        print(h_flex_linear.get(idx, "None"))
        print("\n---------------------------------------\n")

    num_qubits = 19
    H_ising = build_ising_hamiltonian(h_flex_linear, J_flex_quadratic, global_offset, penalty=0.0)

    seq_positions = sorted(list(h_flex_linear.keys()))
    wire_offsets = {}
    current_wire = 0
    rotamer_counts = {}
    for seq in seq_positions:
        wire_offsets[seq] = current_wire
        rotamer_counts[seq] = len(h_flex_linear[seq])
        current_wire += len(h_flex_linear[seq])

    generator_params = {
        "wire_offsets": wire_offsets,
        "seq_positions": seq_positions,
        "rotamer_counts": rotamer_counts,
        "num_qubits": num_qubits,
        "use_gpu": IS_LINUX,
    }
    cost_function, sample_function = qaoa_func_generator(H_ising, custom_xy_mixer_layer, generator_params)

    p = QAOA_LAYERS
    np.random.seed(RAND_SEED)
    # Initialize parameters close to zero to avoid barren plateaus
    initial_params = np.random.uniform(low=-0.01, high=0.01, size=(2, p), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=OPTIMISER_STEPSIZE)
    epochs = OPTIMISER_EPOCHS
    current_params = initial_params
    lowest_param_set = (float('inf'), current_params)

    print("Commencing QAOA Optimization...")
    for epoch in range(epochs):
        current_params, cost = opt.step_and_cost(cost_function, current_params)
        if cost < lowest_param_set[0]:
            lowest_param_set = (cost, current_params)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Cost: {cost:.4f}")

    print("Optimization converged.")

    print(f"Lowest parameter cost: {lowest_param_set[0]}")
    # probabilities = sample_function(lowest_param_set[1])
    probabilities = sample_function(current_params)

    # np.argsort returns indices; we take the last 'TOP_CONFORMATION_COUNTS' and reverse them for descending order
    top_indices = list(np.argsort(probabilities)[-TOP_CONFORMATION_COUNTS:][::-1])
    valid_conformations = validate_conformations(top_indices, probabilities, generator_params)


    # Sort the strictly valid conformations by their true biological energy
    valid_conformations.sort(key=lambda x: x['energy'])
    best_conformation = valid_conformations[0]

    print(f"Optimal Valid Sequence: {best_conformation['bitstring']}")
    print(f"Classical Energy: {best_conformation['energy']} kcal/mol")
    print(f"Valid to Non-Valid Ration: {len(valid_conformations)} - {len(top_indices) - len(valid_conformations)}")

    # print(H_target)
