def validate_conformations(conformations, probabilities, params):
    num_qubits = params["num_qubits"]
    wire_offsets = params['wire_offsets']
    seq_positions = params['seq_positions']
    rotamer_counts = params['rotamer_counts']

    valid_conformations = []

    def int_to_bitstring(idx, length):
        return [int(x) for x in format(idx, f'0{length}b')]


    # 2. Enforce the One-Hot Constraint
    for idx in conformations:
        bitstring = int_to_bitstring(idx, num_qubits)
        is_valid = True

        # Iterate through each residue's allocated wires using your existing `wire_offsets`
        # and the known length of h_flex[seq]
        for seq in seq_positions:
            start_wire = wire_offsets[seq]
            num_rots = rotamer_counts[seq]

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

    valid_conformations.sort(key=lambda x: x['energy'])
    best_conformation = valid_conformations[0]

    print(f"Optimal Valid Sequence: {best_conformation['bitstring']}")
    print(f"Classical Energy: {best_conformation['energy']} kcal/mol")
    print(f"Valid to Non-Valid Ration: {len(valid_conformations)} - {len(conformations) - len(valid_conformations)}")

    return valid_conformations

