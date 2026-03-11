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
        "num_qubits": num_qubits
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

    top_k = 100
    # np.argsort returns indices; we take the last 'top_k' and reverse them for descending order
    top_indices = list(np.argsort(probabilities)[-top_k:][::-1])
    valid_conformations = []

    def int_to_bitstring(idx, length):
        return [int(x) for x in format(idx, f'0{length}b')]


    # 2. Enforce the One-Hot Constraint
    for idx in top_indices:
        bitstring = int_to_bitstring(idx, num_qubits)
        is_valid = True

        # Iterate through each residue's allocated wires using your existing `wire_offsets`
        # and the known length of h_flex[seq]
        for seq in seq_positions:
            start_wire = wire_offsets[seq]
            num_rots = len(h_flex_linear[seq])

            # Sum the bits corresponding to this residue's rotamers
            residue_sum = sum(bitstring[start_wire: start_wire + num_rots])

            if residue_sum != 1:
                is_valid = False
                break  # Fails the penalty constraint

        if is_valid:
            # 3. Calculate True Biological Energy (Classical PyRosetta Equation)
            # using the valid bitstring against the original h_flex and J_flex tensors.
            bio_energy = 0  # calculate_classical_energy(bitstring, h_flex, J_flex, global_offset)
            valid_conformations.append({
                "bitstring": bitstring,
                "probability": probabilities[idx],
                "energy": bio_energy
            })
    print(wire_offsets)
    if not valid_conformations:
        raise ValueError(
            "Zero valid conformations found in the top sampled states. You must increase QAOA depth 'p' or increase the penalty multiplier.")

    # Sort the strictly valid conformations by their true biological energy
    valid_conformations.sort(key=lambda x: x['energy'])
    best_conformation = valid_conformations[0]

    print(f"Optimal Valid Sequence: {best_conformation['bitstring']}")
    print(f"Classical Energy: {best_conformation['energy']} kcal/mol")
    print(f"Valid to Non-Valid Ration: {len(valid_conformations)} - {len(top_indices) - len(valid_conformations)}")

    # print(H_target)
