
def evaluate_quantum_energies(valid_conformations, h_flex, J_flex, global_offset, params):
    wire_offsets = params["wire_offsets"]



    for conformation in valid_conformations:
        bitstring = conformation["bitstring"]
        current_energy = global_offset

        # One body energies
        for seq, energies in h_flex.items():
            base_wire = wire_offsets[seq]
            for rot, e_val in enumerate(energies):
                if bitstring[base_wire + rot] == 1:
                    current_energy += e_val

        # Two body energies
        for (seq_i, seq_j), interactions in J_flex.items():
            for (rot_i, rot_j), e_val in interactions.items():
                k = wire_offsets[seq_i] + rot_i
                l = wire_offsets[seq_j] + rot_j
                pass

        conformation['quantum_energy'] = current_energy
    raise NotImplementedError("Not yet implemented")

def evaluate_pyrosetta_energies():
    raise NotImplementedError("Not yet implemented")

def compare_energies():
    raise NotImplementedError("Not yet implemented")
