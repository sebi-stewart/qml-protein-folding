import itertools
import numpy as np
from typing import List

from rotamer_extraction import TrackedRotamer
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

def evaluate_pyrosetta_energies(valid_conformations: List[Conformation],
                                original_pose, scorefxn,
                                rotamer_library: dict[int, list[TrackedRotamer]], params):
    for conformation in valid_conformations:
        new_pose = evaluate_singular_pyrosetta_energy(conformation, original_pose, rotamer_library, params)

        conformation.pose = new_pose
        conformation.biological_energy = np.float64(scorefxn(new_pose))

def evaluate_singular_pyrosetta_energy(conformation: Conformation, pose,
                                       rotamer_library: dict[int, list[TrackedRotamer]], params):
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

def evaluate_quantum_energies_alt(valid_conformations, h_flex, J_flex, global_offset, params):
    wire_offsets = params["wire_offsets"]

    for conformation in valid_conformations:
        bitstring = conformation["bitstring"]
        current_energy = global_offset

        # One body energies
        for seq, energies in h_flex.items():
            base_wire = wire_offsets[seq]
            for rot, e_val in energies.items():
                if bitstring[base_wire + rot] == 1:
                    current_energy += e_val

        # Two body energies
        for (seq_i, seq_j), interactions in J_flex.items():
            for (rot_i, rot_j), e_val in interactions.items():
                k = wire_offsets[seq_i] + rot_i
                l = wire_offsets[seq_j] + rot_j

                if bitstring[k] == 1 and bitstring[l] == 1:
                    current_energy += e_val

        conformation['quantum_energy'] = current_energy
    # raise NotImplementedError("Not yet implemented")

def print_match_scores(valid_conformations):
    valid_conformations.sort(key=lambda conf: conf['probability'], reverse=True)
    for i, conf in enumerate(valid_conformations):
        conf['idx'] = i
    valid_conformations.sort(key=lambda conf: conf['quantum_energy'])
    rank_match = [abs(conf['idx'] - idx) for idx, conf in enumerate(valid_conformations)]

    results = (np.mean(rank_match[:10]), np.mean(rank_match), np.mean(rank_match[-10:]))
    results2 = (np.median(rank_match[:10]), np.median(rank_match), np.median(rank_match[-10:]))

    print("Rank match means", results)
    print("Rank match medians", results2)

def evaluate_pyrosetta_energies_alt(pose, valid_conformations, scorefxn, rotamer_library, params):
    for conformation in valid_conformations:

        bitstring = conformation["bitstring"]
        bio_energy, pose = evaluate_singular_pyrosetta_energy_alt(pose, bitstring, scorefxn, rotamer_library, params)
        conformation['biological_energy'] = bio_energy
        conformation['pose'] = pose

def evaluate_singular_pyrosetta_energy_alt(pose, bitstrings, scorefxn, rotamer_library, params):
    test_pose = pose.clone()

    seq_positions = params["seq_positions"]
    wire_offsets = params["wire_offsets"]
    rotamer_counts = params["rotamer_counts"]

    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        residue_bits = bitstrings[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        res_entry = rotamer_library[seq]
        rotamer_entry = res_entry[local_rotamer_idx]

        test_pose.replace_residue(seq, rotamer_entry.residue, False)

    bio_energy = scorefxn(test_pose)
    return bio_energy, test_pose


def evaluate_two_energies_alt(pose, valid_conformations, scorefxn, rotamer_library, params, ig, rot_sets, h_linear, J_quadratic):
    for conformation in valid_conformations:

        bitstring = conformation["bitstring"]
        bio_energy, _ = evaluate_singular_pyrosetta_energy_alt(pose, bitstring, scorefxn, rotamer_library, params)
        conformation['bio_en'] = bio_energy

        quant_enery = evaluate_quantum_energy_alt(bitstring, rotamer_library, params, ig, rot_sets, h_linear, J_quadratic)
        conformation['quant_en'] = quant_enery


def evaluate_bio_energy_alt(pose, bitstring, scorefxn, rotamer_library: dict[int, list[TrackedRotamer]], params):
    test_pose = pose.clone()

    seq_positions = params["seq_positions"]
    wire_offsets = params["wire_offsets"]
    rotamer_counts = params["rotamer_counts"]

    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        residue_bits = bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        res_entry = rotamer_library[seq]
        rotamer_entry: TrackedRotamer = res_entry[local_rotamer_idx]

        test_pose.replace_residue(seq, rotamer_entry.residue, False)

    bio_energy = scorefxn(test_pose)
    return bio_energy, test_pose

def evaluate_quantum_energy_alt(bitstring, rotamer_library: dict[int, list[TrackedRotamer]], params, ig, rot_sets, h_linear, J_quadratic):
    seq_positions = params["seq_positions"]
    seq_to_molten = {rot_sets.moltenres_2_resid(m): m for m in range(1, rot_sets.nmoltenres() + 1)}

    wire_offsets = params["wire_offsets"]
    rotamer_counts = params["rotamer_counts"]


    def get_picked_rotamer_idx(seq_idx):
        base_wire = wire_offsets[seq_idx]
        num_rots = rotamer_counts[seq_idx]

        residue_bits = bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        rotamer_seq_entry = rotamer_library[seq_idx]
        rotamer_entry = rotamer_seq_entry[local_rotamer_idx]

        return rotamer_entry.original_pyrosetta_index

    one_body_energies = 0
    for seq_idx in seq_positions:
        base_wire = wire_offsets[seq_idx]
        num_rots = rotamer_counts[seq_idx]

        residue_bits = bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        single_energy = rotamer_library[seq_idx][local_rotamer_idx].one_body_energy
        alt_single_energy = h_linear[seq_idx][local_rotamer_idx]

        assert single_energy == alt_single_energy, f"Energies do not match {seq_idx}{local_rotamer_idx}"

        one_body_energies += single_energy

    alt_one_body_energies = 0

    for seq, energies in h_linear.items():
            base_wire = wire_offsets[seq]
            # print(base_wire)
            for rot, e_val in energies.items():
                # print("\t", rot, "=>", base_wire+rot)
                if bitstring[base_wire + rot] == 1:
                    # print("\t\t", "Found item in bitstring", rot, e_val)
                    alt_one_body_energies += e_val
    print(one_body_energies, alt_one_body_energies, one_body_energies == alt_one_body_energies)

    two_body_energies = 0
    for seq_i, seq_j in itertools.combinations(seq_positions, 2):
        molten_i = seq_to_molten[seq_i]
        molten_j = seq_to_molten[seq_j]

        if not ig.get_edge_exists(molten_i, molten_j): continue
        edge = ig.find_edge(molten_i, molten_j)

        rot_index_i = get_picked_rotamer_idx(seq_i)
        rot_index_j = get_picked_rotamer_idx(seq_j)

        pair_energy = edge.get_two_body_energy(rot_index_i, rot_index_j)
        two_body_energies += pair_energy

    alt_two_body_energies = 0
    for (seq_i, seq_j), interactions in J_quadratic.items():
            for (rot_i, rot_j), e_val in interactions.items():
                k = wire_offsets[seq_i] + rot_i
                l = wire_offsets[seq_j] + rot_j

                if bitstring[k] == 1 and bitstring[l] == 1:
                    alt_two_body_energies += e_val
    # print(two_body_energies, alt_two_body_energies, two_body_energies == alt_two_body_energies)

    return one_body_energies + two_body_energies


