import itertools
import pennylane as qml
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory

from rotamer_extraction import TrackedResidue


def extract_hamiltonian_tensors(residue_library: dict[int, TrackedResidue], ig: InteractionGraphFactory):
    """
    Extracts the linear (one-body) and quadratic (two-body) energy tensors.
    """
    h_linear = {}
    J_quadratic = {}

    # Map sequence positions to their moltenres_id
    seq_positions = list(residue_library.keys())

    # 1. Store linear terms (One-body energies)
    for i in seq_positions:
        tracked_res = residue_library[i]
        h_linear[i] = {idx: tracked_rotamer.one_body_energy for idx, tracked_rotamer in enumerate(tracked_res.rotamers)}

    # 2. Extract quadratic terms (Two-body energies)
    # Iterate over all unique pairs of residues
    for seq_i, seq_j in itertools.combinations(seq_positions, 2):
        res_i = residue_library[seq_i]
        res_j = residue_library[seq_j]

        molten_i = res_i.moltenres_idx
        molten_j = res_j.moltenres_idx

        # Check if an edge exists in the Interaction Graph
        if not ig.get_edge_exists(molten_i, molten_j): continue
        edge = ig.find_edge(molten_i, molten_j)

        # Iterate through the top rotamers of residues i,j
        # and create an interaction graph between all their possible rotamer conformations
        interaction_matrix = {}

        for q_idx_i, rotamer_i in enumerate(res_i.rotamers):
            for q_idx_j, rotamer_j in enumerate(res_j.rotamers):
                # Query the C++ backend for the pairwise energy
                pair_energy = edge.get_two_body_energy(
                    rotamer_i.original_pyrosetta_index,
                    rotamer_j.original_pyrosetta_index
                )

                # Store mapping: (qubit_offset_i, qubit_offset_j) -> Energy
                interaction_matrix[(q_idx_i, q_idx_j)] = pair_energy

        if interaction_matrix:
            J_quadratic[(seq_i, seq_j)] = interaction_matrix

    return h_linear, J_quadratic

def reduce_hamiltonian(
        h_linear: dict[int, dict[int, float]],
        J_quadratic: dict[tuple[int, int], dict[tuple[int, int], float]],
        residue_library: dict[int, TrackedResidue]):

    fixed_res = [idx for idx, res in residue_library.items() if len(res.rotamers) == 1]
    flex_res = [idx for idx, res in residue_library.items() if len(res.rotamers) > 1]

    # 1. Initialize new dictionaries for the quantum-ready flexible residues
    h_flex = {res: h_linear[res].copy() for res in flex_res}
    J_flex = {}  # Will only contain edges between flex_res
    global_offset = 0.0

    for f in fixed_res:
        global_offset += h_linear[f][0]
        for f2 in fixed_res:
            if f >= f2: continue
            edge = (f, f2)
            if edge not in J_quadratic: continue

            global_offset += J_quadratic[edge][(0, 0)]

    # 3. Absorb Fixed-to-Flexible interactions into Flexible Linear Terms
    for v in flex_res:
        for f in fixed_res:
            edge = (min(v, f), max(v, f))
            if edge not in J_quadratic: continue

            flex_res_entry = residue_library[v]
            for rot_v in range(len(flex_res_entry.rotamers)):
                rot_edge = (rot_v, 0) if v < f else (0, rot_v)
                h_flex[v][rot_v] += J_quadratic[edge][rot_edge]

    # 4. Retain only Flexible-to-Flexible interactions
    for (i, j), interactions in J_quadratic.items():
        if i in flex_res and j in flex_res:
            J_flex[(i, j)] = interactions

    return h_flex, J_flex, global_offset

def extract_and_reduce_tensors(residue_library: dict[int, TrackedResidue], ig: InteractionGraphFactory):
    h_linear, J_quadratic = extract_hamiltonian_tensors(residue_library, ig)
    h_flex_linear, J_flex_quadratic, global_offset = reduce_hamiltonian(h_linear, J_quadratic, residue_library)
    return h_flex_linear, J_flex_quadratic, global_offset


def build_ising_hamiltonian(h_flex, J_flex) -> tuple[qml.Hamiltonian, int]:
    """
    Compiles the reduced classical PyRosetta tensors into a PennyLane Pauli-Z Hamiltonian,
    incorporating the background thermodynamic offset.
    """
    seq_positions = sorted(list(h_flex.keys()))
    num_residues = len(seq_positions)

    # Track dynamic wire allocation (Prefix Sum)
    wire_offsets = {}
    current_wire = 0
    for seq in seq_positions:
        wire_offsets[seq] = current_wire
        current_wire += len(h_flex[seq])

    num_qubits = current_wire

    w_linear = {k: 0.0 for k in range(num_qubits)}
    W_quadratic = {(k, l): 0.0 for k in range(num_qubits) for l in range(num_qubits) if k < l}

    # 1.1 Populate unified weights - Biological Terms for Linear
    for seq in seq_positions:
        rotamers_in_cur_res = len(h_flex[seq])
        base_wire = wire_offsets[seq]

        for rot in range(rotamers_in_cur_res):
            k = base_wire + rot
            w_linear[k] = h_flex[seq][rot]

    # 1.2 Populate unified weights - Biological terms for quadratic term
    for (seq_i, seq_j), interactions in J_flex.items():
        for (rot_i, rot_j), energy in interactions.items():
            k = wire_offsets[seq_i] + rot_i
            l = wire_offsets[seq_j] + rot_j

            if k > l: k, l = l, k
            W_quadratic[(k, l)] += energy

    coeffs = []
    observables = []

    # Full equation
    # Linear term:
    # W_k * (1 - Z_k)/2 = W_k/2 - W_k/2 * Z_k

    # Quadratic Term:
    # W_kl * (1 - Z_k)/2 * (1 - Z_l)/2 = W_kl/4 - W_kl/4 * Z_k - W_kl/4 * Z_l + W_kl/4 * Z_k * Z_l

    # Some of these terms depend on Z_k or Z_l (or both), these terms are added to the hamiltonian in the below for loops
    # Terms such as W_k/2 and W_kl/4 are constant, and therefore added to the constant C_id global offset
    # This acts as an optimisation

    # 2. Add the Classical Global Offset to the Identity Term
    # C_id = Offset + sum(w_k / 2) + sum(W_kl / 4)
    C_id = (sum(w_linear.values()) / 2.0) + (sum(W_quadratic.values()) / 4.0)

    near_zero_value = lambda x: abs(x) < 1e-6 # helper function to avoid extra computation by near zero values aka. rounding issues

    if not near_zero_value(C_id):
        coeffs.append(C_id)
        observables.append(qml.Identity(wires=0))

    # 3. Z and ZZ terms (Standard Ising Substitution)
    for k in range(num_qubits):
        C_k = -w_linear[k] / 2.0
        for l in range(num_qubits):
            if k < l:
                C_k -= W_quadratic[(k, l)] / 4.0
            elif l < k:
                C_k -= W_quadratic[(l, k)] / 4.0

        if near_zero_value(C_k): continue

        coeffs.append(C_k)
        observables.append(qml.PauliZ(wires=k))

    for (k, l), W_kl in W_quadratic.items():
        C_kl = W_kl / 4.0
        if near_zero_value(C_kl): continue

        coeffs.append(C_kl)
        observables.append(qml.PauliZ(wires=k) @ qml.PauliZ(wires=l))

    print(f"Reduced Hamiltonian built: {num_qubits} Qubits, {len(coeffs)} Pauli strings.")
    return qml.dot(coeffs, observables), len(coeffs)