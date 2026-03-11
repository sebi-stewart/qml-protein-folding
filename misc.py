from constants import IS_LINUX

def init_generator_params(h_flex_linear):
    print("initializing generator_params")

    seq_positions = sorted(list(h_flex_linear.keys()))
    wire_offsets = {}
    current_wire = 0
    rotamer_counts = {}
    for seq in seq_positions:
        wire_offsets[seq] = current_wire
        rotamer_counts[seq] = len(h_flex_linear[seq])
        current_wire += len(h_flex_linear[seq])

    num_qubits = rotamer_counts.values()


    generator_params = {
        "wire_offsets": wire_offsets,
        "seq_positions": seq_positions,
        "rotamer_counts": rotamer_counts,
        "num_qubits": num_qubits,
        "use_gpu": IS_LINUX,
    }

    return generator_params