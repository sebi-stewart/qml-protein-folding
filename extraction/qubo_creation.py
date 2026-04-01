import itertools
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory

from rotamer_extraction import TrackedResidue

import numpy as np

def _extract_hamiltonian_tensors(residue_library: dict[int, TrackedResidue], ig: InteractionGraphFactory):
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

def _reduce_hamiltonian(
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
    h_linear, J_quadratic = _extract_hamiltonian_tensors(residue_library, ig)
    h_flex_linear, J_flex_quadratic, global_offset = _reduce_hamiltonian(h_linear, J_quadratic, residue_library)
    return h_flex_linear, J_flex_quadratic, global_offset


def build_dense_qubo(h_linear: dict, J_quadratic: dict, num_qubits: int, wire_offsets: dict) -> tuple[np.ndarray, np.ndarray]:
    h_dense = np.zeros(num_qubits, dtype=np.float64)
    J_dense = np.zeros((num_qubits, num_qubits), dtype=np.float64)

    # Map Linear Terms (FIXED: Using .items() for nested dictionaries)
    for seq, energies in h_linear.items():
        base_wire = wire_offsets[seq]
        for rot_idx, e_val in energies.items():
            h_dense[base_wire + rot_idx] = e_val

    # Map Quadratic Terms (Populates an Upper-Triangular Matrix)
    for (seq_i, seq_j), interactions in J_quadratic.items():
        for (rot_i, rot_j), e_val in interactions.items():
            k = wire_offsets[seq_i] + rot_i
            l = wire_offsets[seq_j] + rot_j
            # Places the floating point energy directly into the matrix
            J_dense[k, l] = e_val

    return h_dense, J_dense