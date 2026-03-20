from dataclasses import dataclass
import numpy as np
from typing import List
from validation import Conformation


def evaluate_quantum_energies(valid_conformations: List[Conformation], h_flex, J_flex, global_offset, params):
    wire_offsets = params["wire_offsets"]
    rotamer_counts = params["rotamer_counts"]

    for conformation in valid_conformations:
        energy = evaluate_singular_quantum_energy(conformation, h_flex, J_flex, global_offset, wire_offsets, rotamer_counts)
        conformation.quantum_energy = energy

def evaluate_singular_quantum_energy(conformation, h_flex, J_flex, global_offset, wire_offsets, rotamer_counts) -> np.float64:
    bitstring = conformation.bitstring
    current_energy = np.float64(global_offset)

    # One body energies
    for seq, energies in h_flex.items():
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        residue_bits = bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        current_energy += energies[local_rotamer_idx]

    # Two body energies
    for (seq_i, seq_j), interactions in J_flex.items():
        for (rot_i, rot_j), e_val in interactions.items():
            k = wire_offsets[seq_i] + rot_i
            l = wire_offsets[seq_j] + rot_j

            if bitstring[k] == 1 and bitstring[l] == 1:
                current_energy += e_val
    return current_energy

def evaluate_pyrosetta_energies(valid_conformations: List[Conformation], original_pose, scorefxn, rotamer_library, params):
    for conformation in valid_conformations:
        new_pose = evaluate_singular_pyrosetta_energy(conformation, original_pose, rotamer_library, params)

        conformation.pose = new_pose
        conformation.biological_energy = np.float64(scorefxn(new_pose))

def evaluate_singular_pyrosetta_energy(conformation: Conformation, pose, rotamer_library, params):
    new_pose = pose.clone()

    seq_positions = params["seq_positions"]
    wire_offsets = params["wire_offsets"]
    rotamer_counts = params["rotamer_counts"]

    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        residue_bits = conformation.bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        res_entry = rotamer_library[seq]
        rotamer_entry = res_entry[local_rotamer_idx]

        new_pose.replace_residue(seq, rotamer_entry.residue, False)
    return new_pose

def compare_energies(valid_conformations: List[Conformation]):
    for conf in valid_conformations:
        assert None not in (conf, conf.quantum_energy, conf.biological_energy, conf.pose), f"Some item in the conformation class is None {conf}"

        diff = abs(conf.quantum_energy - conf.biological_energy)
        if diff > 1e-02:
            raise AssertionError(f"The difference was too large to ignore for conformation: {conf}")
        elif diff > 1e-05:
            print(f"The difference wasn't significant, however still over 1e-05 ({abs(diff)}): {conf}")

def calculate_and_compare_energies(valid_conformations: List[Conformation], h_flex, J_flex, global_offset, original_pose, scorefxn, rotamer_library, params):
    print("==================== ENERGY OPERATIONS ====================")
    print(f"Calculating Quantum Energies for all {len(valid_conformations)} conformations")
    evaluate_quantum_energies(valid_conformations, h_flex, J_flex, global_offset, params)

    print(f"Calculating Pyrosetta for all {len(valid_conformations)} conformations ")
    evaluate_pyrosetta_energies(valid_conformations, original_pose, scorefxn, rotamer_library, params)

    print(f"Comparing both energy types for all {len(valid_conformations)} conformations ")
    compare_energies(valid_conformations)

    print("==================== ENERGY OPERATIONS COMPLETE ====================")