import itertools
import logging

import numpy as np
from typing import List

from misc import BasicParams
from rotamer_extraction import TrackedResidue
from validation import Conformation


def evaluate_quantum_energies(valid_conformations: List[Conformation], h_flex, J_flex, global_offset, params: BasicParams):
    wire_offsets = params.wire_offsets
    rotamer_counts = params.rotamer_counts

    for conformation in valid_conformations:
        energy = evaluate_singular_quantum_energy(conformation.bitstring, h_flex, J_flex, global_offset, wire_offsets, rotamer_counts)
        conformation.quantum_energy = energy

def evaluate_singular_quantum_energy(bitstring, h_flex, J_flex, global_offset, wire_offsets, rotamer_counts) -> np.float64:
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
                                residue_library: dict[int, TrackedResidue], params: BasicParams):
    for conformation in valid_conformations:
        new_pose = evaluate_singular_pyrosetta_energy(conformation, original_pose, residue_library, params)

        conformation.biological_energy = np.float64(scorefxn(new_pose))

def evaluate_singular_pyrosetta_energy(conformation: Conformation, pose,
                                       residue_library: dict[int, TrackedResidue], params: BasicParams):
    new_pose = pose.clone()

    seq_positions = params.seq_positions
    wire_offsets = params.wire_offsets
    rotamer_counts = params.rotamer_counts

    #Flexible rotamers
    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        residue_bits = conformation.bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        res_entry = residue_library[seq]
        rotamer_entry = res_entry.rotamers[local_rotamer_idx]

        new_pose.replace_residue(seq, rotamer_entry.residue, False)

    # Set "fixed" rotamers
    all_seq = [key for key in residue_library]
    for seq in all_seq:
        if seq in seq_positions: continue

        res_entry = residue_library[seq]
        rotamer_entry = res_entry.rotamers[0]

        new_pose.replace_residue(seq, rotamer_entry.residue, False)

    return new_pose

def extract_anomaly_zones(rank_match, energies, logger: logging.Logger):
    # 1. Find the exact indices where the mismatch occurred
    breaking_indices = [idx for idx, diff in enumerate(rank_match) if diff != 0]

    # 2 & 3. Create and merge the anomaly zones (context windows of +/- 1)
    anomaly_zones = []
    max_idx = len(energies) - 1

    for idx in breaking_indices:
        # Define the window, ensuring we don't drop below 0 or go past the list length
        start = max(0, idx - 1)
        end = min(max_idx, idx + 1)

        if not anomaly_zones:
            # Add the first zone
            anomaly_zones.append([start, end])
        else:
            # Check if the current zone overlaps or touches the previous zone
            last_zone = anomaly_zones[-1]
            if start <= last_zone[1]:
                # Merge them by extending the end of the previous zone
                last_zone[1] = max(last_zone[1], end)
            else:
                # No overlap, so we add it as a new distinct zone
                anomaly_zones.append([start, end])

    # 4. Print out the energies within our nicely merged anomaly zones
    for start, end in anomaly_zones:
        logger.debug(f"\n--- Anomaly Zone: Indices {start} to {end} ---")

        # Loop through the specific merged range and print the energy comparisons
        for i in range(start, end + 1):
            q_energy = energies[i]['quantum_energy']
            b_energy = energies[i]['biological_energy']

            # Optional: Add a marker (like '>>>') to instantly see which items actually broke the rank
            marker = ">>>" if i in breaking_indices else "   "
            logger.debug(f"{marker} IDX {i}: q-energy={q_energy:13.6f} | b-energy={b_energy:13.6f} | {(q_energy - b_energy):.6f}")

def compare_energies(valid_conformations: List[Conformation], logger: logging.Logger):
    deltas = []
    for conf in valid_conformations:
        assert None not in (conf, conf.quantum_energy, conf.biological_energy), f"Some item in the conformation class is None {conf}"

        delta = conf.quantum_energy - conf.biological_energy
        deltas.append(delta)
    logger.debug(f"Value Deltas: {np.mean(deltas)} | {np.std(deltas)}")


    # Check the pyrosetta and QUBO oderings match
    energies = [{"quantum_energy": conf.quantum_energy, "biological_energy": conf.biological_energy,
                 'quant_idx': -1, 'bio_idx': -1} for conf in valid_conformations]

    energies.sort(key=lambda x: x['quantum_energy'])
    for i, eng in enumerate(energies):
        eng['quant_idx'] = i

    energies.sort(key=lambda x: x['biological_energy'])
    for i, eng in enumerate(energies):
        eng['bio_idx'] = i

    rank_match = [abs(conf['quant_idx'] - conf['bio_idx']) for idx, conf in enumerate(energies)]
    if not all(match == 0 for match in rank_match):
        logger.error(f"ERROR: ================== Not all ranks matched, {rank_match}\n")
        extract_anomaly_zones(rank_match, energies, logger)


    if np.std(deltas) > 0.1:
        raise AssertionError(f"Deltas std deviation was too high: {np.std(deltas)}")


def calculate_and_compare_energies(valid_conformations: List[Conformation],
                                   h_flex, J_flex, global_offset,
                                   original_pose, scorefxn, residue_library: dict[int, TrackedResidue],
                                   params: BasicParams, logger: logging.Logger):
    logger.debug("==================== ENERGY OPERATIONS ====================")
    logger.debug(f"Calculating Quantum Energies for all {len(valid_conformations)} conformations")
    evaluate_quantum_energies(valid_conformations, h_flex, J_flex, global_offset, params)

    logger.debug(f"Calculating Pyrosetta for all {len(valid_conformations)} conformations ")
    evaluate_pyrosetta_energies(valid_conformations, original_pose, scorefxn, residue_library, params)

    logger.debug(f"Comparing both energy types for all {len(valid_conformations)} conformations ")
    compare_energies(valid_conformations, logger)

    logger.debug("==================== ENERGY OPERATIONS COMPLETE ====================\n")

def extract_lowest_energy_bitstrings(valid_bitstrings, h_linear, J_quadratic, logger, epsilon, params: BasicParams) -> set[int]:
    def bitstring_to_int(bitstring):
        return int(''.join(map(str, bitstring)), 2)

    wire_offsets = params.wire_offsets
    rotamer_counts = params.rotamer_counts

    energies = []
    for valid_bitstring in valid_bitstrings:
        qubo_energy = evaluate_singular_quantum_energy(valid_bitstring, h_linear, J_quadratic, 0.0, wire_offsets, rotamer_counts)
        energies.append(qubo_energy)
    min_energy = min(energies)
    lowest_energy_bitstrings = set()
    for bitstring, energy in zip(valid_bitstrings, energies):
        if abs(energy - min_energy) < epsilon:
            lowest_energy_bitstrings.add(bitstring_to_int(bitstring))
    return lowest_energy_bitstrings

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

def evaluate_two_energies_alt(pose, valid_conformations: list[Conformation], scorefxn, residue_library: dict[int, TrackedResidue], params, ig, rot_sets, h_linear, J_quadratic):
    for conformation in valid_conformations:

        bitstring = conformation.bitstring
        bio_energy, _ = evaluate_singular_pyrosetta_energy_alt(pose, bitstring, scorefxn, residue_library, params)
        # conformation.biological_energy = bio_energy

        quant_energy = evaluate_quantum_energy_alt(bitstring, residue_library, params, ig, rot_sets, h_linear, J_quadratic)
        # conformation.quantum_energy = quant_energy
        print("Bio:", bio_energy, conformation.biological_energy, bio_energy - conformation.biological_energy)
        print("Qua:", quant_energy, conformation.quantum_energy, quant_energy - conformation.quantum_energy)
        print("Diff:", bio_energy-quant_energy)


def evaluate_singular_pyrosetta_energy_alt(pose, bitstrings, scorefxn, residue_library: dict[int, TrackedResidue], params):
    test_pose = pose.clone()

    seq_positions = params["seq_positions"]
    wire_offsets = params["wire_offsets"]
    rotamer_counts = params["rotamer_counts"]

    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        residue_bits = bitstrings[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        res_entry: TrackedResidue = residue_library[seq]
        rotamer_entry = res_entry.rotamers[local_rotamer_idx]

        test_pose.replace_residue(seq, rotamer_entry.residue, False)

    bio_energy = scorefxn(test_pose)
    return bio_energy, test_pose

def evaluate_quantum_energy_alt(bitstring, residue_library: dict[int, TrackedResidue], params, ig, rot_sets, h_linear, J_quadratic):
    seq_positions = params["seq_positions"]
    seq_to_molten = {rot_sets.moltenres_2_resid(m): m for m in range(1, rot_sets.nmoltenres() + 1)}

    wire_offsets = params["wire_offsets"]
    rotamer_counts = params["rotamer_counts"]


    def get_picked_rotamer_idx(seq_idx):
        base_wire = wire_offsets[seq_idx]
        num_rots = rotamer_counts[seq_idx]

        residue_bits = bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        rotamer_seq_entry = residue_library[seq_idx]
        rotamer_entry = rotamer_seq_entry.rotamers[local_rotamer_idx]

        return rotamer_entry.original_pyrosetta_index

    one_body_energies = 0
    for seq_idx in seq_positions:
        base_wire = wire_offsets[seq_idx]
        num_rots = rotamer_counts[seq_idx]

        residue_bits = bitstring[base_wire : base_wire + num_rots]
        local_rotamer_idx = residue_bits.index(1)

        single_energy = residue_library[seq_idx].rotamers[local_rotamer_idx].one_body_energy
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

    return one_body_energies + two_body_energies