from extraction.rotamers import TrackedResidue
from phase2.objects import RescoringConformation

import numpy as np
from qaoa.objects import BasicParams

def compare_scoring_results(scored_conformations: list[RescoringConformation], base_conformation: RescoringConformation, logger: logging.Logger):
    # Compare the biological energies of the new conformations with the original pose
    for conf in scored_conformations:
        energy_diff = conf.biological_energy - base_conformation.biological_energy
        conf.energy_diff = energy_diff
        logger.debug(f"Bitstring: {conf.bitstring}, Biological Energy: {conf.biological_energy:.4f}, Energy Difference: {energy_diff:.4f}")


def evaluate_pyrosetta_energies(unique_bitstrings: list[list[int]],
                                original_pose, scorefxn,
                                residue_library: dict[int, TrackedResidue], params: BasicParams):
    conformations = []
    for bitstring in unique_bitstrings:
        new_pose = evaluate_singular_pyrosetta_energy(bitstring, original_pose, residue_library, params)
        biological_energy = np.float64(scorefxn(new_pose))
        conformations.append(
            RescoringConformation(
                bitstring=bitstring,
                pose=new_pose,
                biological_energy=biological_energy
            )
        )
    return conformations

def evaluate_singular_pyrosetta_energy(bitstring: list[int], pose,
                                       residue_library: dict[int, TrackedResidue], params: BasicParams):
    new_pose = pose.clone()

    seq_positions = params.seq_positions
    wire_offsets = params.wire_offsets
    rotamer_counts = params.rotamer_counts

    #Flexible rotamers
    for seq in seq_positions:
        base_wire = wire_offsets[seq]
        num_rots = rotamer_counts[seq]

        residue_bits = bitstring[base_wire : base_wire + num_rots]
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