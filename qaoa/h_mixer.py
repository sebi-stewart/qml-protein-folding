import pennylane as qml

def custom_xy_mixer_layer(beta, wire_offsets, seq_positions, rotamer_counts):

    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        for i in range(num_rots):
            for j in range(i + 1, num_rots):
                w_i = base_wire + i
                w_j = base_wire + j

                qml.IsingXY(beta, wires=[w_i, w_j])

def ring_xy_mixer_layer(beta, wire_offsets, seq_positions, rotamer_counts):
    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        if num_rots == 1: continue
        if num_rots == 2:
            qml.IsingXY(beta, wires=[base_wire, base_wire + 1])
            continue

        for i in range(num_rots):
            wire_a = base_wire + i
            wire_b = base_wire + ((i+1) % num_rots)

            qml.IsingXY(beta, wires=[wire_a, wire_b])