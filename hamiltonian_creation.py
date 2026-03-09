import itertools
import pennylane as qml
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory


def extract_hamiltonian_tensors(rotamer_library: dict, ig: InteractionGraphFactory, rot_sets: RotamerSets):
    """
    Extracts the linear (one-body) and quadratic (two-body) energy tensors.
    """
    h_linear = {}
    J_quadratic = {}
    
    # Map sequence positions to their moltenres_id
    seq_to_molten = {rot_sets.moltenres_2_resid(m): m for m in range(1, rot_sets.nmoltenres() + 1)}
    seq_positions = list(rotamer_library.keys())
    
    # 1. Store linear terms (One-body energies)
    for i in seq_positions:
        # We enforce a strict 0 to 3 internal index for the 4 qubits per residue
        h_linear[i] = {qubit_idx: data[0] for qubit_idx, data in enumerate(rotamer_library[i])}
        
    # 2. Extract quadratic terms (Two-body energies)
    # Iterate over all unique pairs of residues
    for i, j in itertools.combinations(seq_positions, 2):
        molten_i = seq_to_molten[i]
        molten_j = seq_to_molten[j]
        
        # Check if an edge exists in the Interaction Graph
        if ig.get_edge_exists(molten_i, molten_j):
            J_quadratic[(i, j)] = {}
            
            # Iterate through the top 4 rotamers of residue i
            for q_idx_i, data_i in enumerate(rotamer_library[i]):
                rot_index_i = data_i[1] # The original C++ state index
                
                # Iterate through the top 4 rotamers of residue j
                for q_idx_j, data_j in enumerate(rotamer_library[j]):
                    rot_index_j = data_j[1]
                    
                    # Query the C++ backend for the pairwise energy
                    pair_energy = ig.get_two_body_energy_for_edge(
                        molten_i, molten_j, rot_index_i, rot_index_j
                    )
                    
                    # Store mapping: (qubit_offset_i, qubit_offset_j) -> Energy
                    J_quadratic[(i, j)][(q_idx_i, q_idx_j)] = pair_energy
        else:
            # If nodes are too far apart (e.g., > 10 Angstroms), interaction is 0
            J_quadratic[(i, j)] = { (q_i, q_j): 0.0 for q_i in range(4) for q_j in range(4) }
            
    return h_linear, J_quadratic

def reduce_hamiltonian(h_linear, J_quadratic, rotamer_library):
    fixed_res = [res for res, rots in rotamer_library.items() if len(rots) == 1]
    flex_res = [res for res, rots in rotamer_library.items() if len(rots) > 1]

    # 1. Initialize new dictionaries for the quantum-ready flexible residues
    h_flex = {res: h_linear[res].copy() for res in flex_res}
    J_flex = {}  # Will only contain edges between flex_res
    global_offset = 0.0

    for f in fixed_res:
        global_offset += h_linear[f][0]
        for f2 in fixed_res:
            if f < f2 and (f, f2) not in J_quadratic:
                continue
            global_offset += J_quadratic[(f, f2)][(0, 0)]

    # 3. Absorb Fixed-to-Flexible interactions into Flexible Linear Terms
    for v in flex_res:
        for rot_v in range(len(rotamer_library[v])):
            for f in fixed_res:
                edge = (min(v, f), max(v, f))
                if edge in J_quadratic:
                    idx = (rot_v, 0) if v < f else (0, rot_v)
                    h_flex[v][rot_v] += J_quadratic[edge][idx]

    # 4. Retain only Flexible-to-Flexible interactions
    for (i, j), interactions in J_quadratic.items():
        if i in flex_res and j in flex_res:
            J_flex[(i, j)] = interactions

    return h_flex, J_flex, global_offset


def build_ising_hamiltonian(h_flex, J_flex, global_offset, penalty=500.0):
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

    # 1. Populate unified weights (Biology + Penalty)
    for seq in seq_positions:
        rotamers_per_res = len(h_flex[seq])
        base_wire = wire_offsets[seq]

        for rot in range(rotamers_per_res):
            k = base_wire + rot
            w_linear[k] = h_flex[seq][rot] - penalty

            # Intra-residue penalty
            for rot_other in range(rot + 1, rotamers_per_res):
                l = base_wire + rot_other
                W_quadratic[(k, l)] = 2.0 * penalty

    # Inter-residue biological energies
    for (seq_i, seq_j), interactions in J_flex.items():
        for (rot_i, rot_j), energy in interactions.items():
            k = wire_offsets[seq_i] + rot_i
            l = wire_offsets[seq_j] + rot_j

            if k > l: k, l = l, k
            W_quadratic[(k, l)] += energy

    coeffs = []
    observables = []

    # 2. Add the Classical Global Offset to the Identity Term
    # C_id = Offset + (N * lambda) + sum(w_k / 2) + sum(W_kl / 4)
    C_id = global_offset + (num_residues * penalty) + (sum(w_linear.values()) / 2.0) + (sum(W_quadratic.values()) / 4.0)

    if abs(C_id) > 1e-6:
        coeffs.append(C_id)
        observables.append(qml.Identity(0))

    # 3. Z and ZZ terms (Standard Ising Substitution)
    for k in range(num_qubits):
        C_k = -w_linear[k] / 2.0
        for l in range(num_qubits):
            if k < l:
                C_k -= W_quadratic[(k, l)] / 4.0
            elif l < k:
                C_k -= W_quadratic[(l, k)] / 4.0

        if abs(C_k) > 1e-6:
            coeffs.append(C_k)
            observables.append(qml.PauliZ(k))

    for (k, l), W_kl in W_quadratic.items():
        C_kl = W_kl / 4.0
        if abs(C_kl) > 1e-6:
            coeffs.append(C_kl)
            observables.append(qml.PauliZ(k) @ qml.PauliZ(l))

    print(f"Reduced Hamiltonian built: {num_qubits} Qubits, {len(coeffs)} Pauli strings.")
    return qml.Hamiltonian(coeffs, observables)