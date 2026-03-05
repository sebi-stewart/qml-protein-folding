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


def build_ising_hamiltonian(h_linear, J_quadratic, penalty=500.0, rotamers_per_res=4):
    """
    Algebraically pre-compiles classical PyRosetta tensors into a PennyLane Pauli-Z Hamiltonian.
    """
    # 1. Map sequence positions to deterministic wire (qubit) indices
    seq_positions = sorted(list(h_linear.keys()))
    num_residues = len(seq_positions)
    num_qubits = num_residues * rotamers_per_res
    
    # Helper function to get wire index
    def get_wire(seq_idx, rot_idx):
        res_offset = seq_positions.index(seq_idx)
        return (res_offset * rotamers_per_res) + rot_idx

    # 2. Initialize unified weight arrays
    w_linear = {k: 0.0 for k in range(num_qubits)}
    W_quadratic = {(k, l): 0.0 for k in range(num_qubits) for l in range(num_qubits) if k < l}

    # 3. Populate unified weights (Biology + Penalty)
    for seq in seq_positions:
        for rot in range(rotamers_per_res):
            k = get_wire(seq, rot)
            # Biological one-body energy - lambda penalty
            w_linear[k] = h_linear[seq][rot] - penalty
            
            # Intra-residue penalty (2 * lambda)
            for rot_other in range(rot + 1, rotamers_per_res):
                l = get_wire(seq, rot_other)
                W_quadratic[(k, l)] = 2.0 * penalty
                
    # Add Inter-residue biological energies (J_quadratic)
    for (seq_i, seq_j), interactions in J_quadratic.items():
        for (rot_i, rot_j), energy in interactions.items():
            k = get_wire(seq_i, rot_i)
            l = get_wire(seq_j, rot_j)
            
            # Ensure k < l for the quadratic dictionary
            if k > l:
                k, l = l, k
            W_quadratic[(k, l)] += energy

    # 4. Convert to Pauli-Z Ising Coefficients
    coeffs = []
    observables = []
    
    # Constant term (Identity)
    C_id = (num_residues * penalty) + sum(w_linear.values()) / 2.0 + sum(W_quadratic.values()) / 4.0
    coeffs.append(C_id)
    observables.append(qml.Identity(0)) # Identity on any wire acts globally

    # Z terms
    for k in range(num_qubits):
        C_k = -w_linear[k] / 2.0
        # Subtract W_kl / 4 for all connected edges
        for l in range(num_qubits):
            if k < l:
                C_k -= W_quadratic[(k, l)] / 4.0
            elif l < k:
                C_k -= W_quadratic[(l, k)] / 4.0
                
        if abs(C_k) > 1e-6: # Sparsity check
            coeffs.append(C_k)
            observables.append(qml.PauliZ(k))

    # ZZ terms
    for (k, l), W_kl in W_quadratic.items():
        C_kl = W_kl / 4.0
        if abs(C_kl) > 1e-6:
            coeffs.append(C_kl)
            observables.append(qml.PauliZ(k) @ qml.PauliZ(l))

    # 5. Return the PennyLane Hamiltonian
    print(f"Constructed Hamiltonian with {len(coeffs)} Pauli strings.")
    return qml.Hamiltonian(coeffs, observables)