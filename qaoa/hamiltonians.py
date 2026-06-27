import logging
import pennylane as qml

def extract_ising_items(h_flex, J_flex, logger: logging.Logger) -> tuple[list[float], list, int]:
    """
    Compiles the reduced classical PyRosetta tensors into a PennyLane Pauli-Z Hamiltonian,
    incorporating the background thermodynamic offset.
    """
    seq_positions = sorted(list(h_flex.keys()))

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

    logger.info(f"Reduced Hamiltonian built: {num_qubits} Qubits, {len(coeffs)} Pauli strings.")
    return coeffs, observables, num_qubits