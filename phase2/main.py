import logging
from collections import defaultdict
from dataclasses import dataclass

# from extraction import
import pyrosetta

from extraction.initialisation import initialize_rosetta
from extraction.main import TestInstanceFactory, ExtractionTestInstance, run_pyrosetta_obj_extraction, \
    from_energies_to_tensors
from extraction.rotamers import TrackedResidue
from logging_setup import setup_logging

import numpy as np
import pennylane as qml

from qaoa.devices import get_cached_device
from qaoa.generators import qaoa_func_generator
from qaoa.h_mixer import ring_xy_mixer_layer
from qaoa.hamiltonians import extract_ising_items
from qaoa.objects import init_basic_params, BasicParams


def extract_one_body_energies_from_instance(inst: ExtractionTestInstance, logger: logging.Logger):
    pose, residue_library, ig, rot_sets, scorefxn = run_pyrosetta_obj_extraction(
        inst.pose_func,
        logger=logger,
        n=inst.rotamer_count,
        active_start=inst.residue_start,
        active_end=inst.residue_end)

    one_body, two_body, global_offset = from_energies_to_tensors(residue_library, ig)
    return one_body, two_body, pose, residue_library, ig, rot_sets, scorefxn

def get_sample_function_for_phase_2(logger: logging.Logger, one_body, two_body, shots):
    basic_params: BasicParams = init_basic_params(one_body)

    coeffs, observables, num_qubits = extract_ising_items(one_body, two_body, logger)
    cost_hamiltonian = qml.dot(coeffs, observables)

    device_type = 'lightning.qubit'
    dev = get_cached_device(num_qubits, device_type)
    logger.info(f"Running on {device_type} for {num_qubits} qubits")

    cost_func, sample_function = qaoa_func_generator(dev, cost_hamiltonian, ring_xy_mixer_layer, basic_params, shots)
    return sample_function, basic_params

def extract_best_qaoa_params(qaoa_file_path: str):
    assert qaoa_file_path.endswith(".npz"), "Expected a .npz file containing the QAOA results"
    data = np.load(qaoa_file_path, allow_pickle=True)
    print(data)

    # Act as an oracle to extract the relevant data for the next phase
    # We refrain from looking at the actual values, but we know the structure of the saved data from the layered_run function
    optimised_params = data['optimized_params']
    return optimised_params



def get_and_process_shot_results(sample_func, best_params, logger: logging.Logger):
    shot_results = {
        seed: sample_func(best_params[seed])
        for seed in range(30)  # Assuming 30 seeds as per the original code
    }

    # return shots results with duplicate bitstrings removed, and a dict of all unique bitstrings and their counts across all seeds
    unique_bitstrings = set()
    processed_results = {}

    for seed, shots in shot_results.items():
        unique_shots = set(tuple(shot) for shot in shots)
        unique_bitstrings.update(unique_shots)
        processed_results[seed] = unique_shots

    return processed_results, list(unique_bitstrings)

@dataclass
class RescoringConformation:
    bitstring: list[int] | None
    pose: pyrosetta.Pose = None
    biological_energy: np.float64 = None
    energy_diff: np.float64 = None

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

def compare_scoring_results(scored_conformations: list[RescoringConformation], base_conformation: RescoringConformation, logger: logging.Logger):
    # Compare the biological energies of the new conformations with the original pose
    for conf in scored_conformations:
        energy_diff = conf.biological_energy - base_conformation.biological_energy
        conf.energy_diff = energy_diff
        logger.info(f"Bitstring: {conf.bitstring}, Biological Energy: {conf.biological_energy:.4f}, Energy Difference: {energy_diff:.4f}")

def extract_best_conformation_for_seeds(processed_results: dict[int, set[tuple[int]]], scored_conformations: list[RescoringConformation]):
    best_conformations_per_seed = {}

    for seed, bitstrings in processed_results.items():
        best_conf = None
        best_energy_diff = float('inf')

        for conf in scored_conformations:
            if tuple(conf.bitstring) not in bitstrings: continue
            if conf.energy_diff < best_energy_diff:
                best_energy_diff = conf.energy_diff
                best_conf = conf

        best_conformations_per_seed[seed] = best_conf

    return best_conformations_per_seed


if __name__ == "__main__":
    logger = setup_logging("rescoring_phase2", "5PTI")
    initialize_rosetta(pyrosetta, extra_flags="-mute all")
    fac = TestInstanceFactory()



    # inst = fac.create_test_instance_from_func(
    #     pose_func=lambda : pyrosetta.pose_from_pdb("data/AF-P00974-F1-model_v6.pdb"),
    #     test_name="AF-5PTI",
    #     start=13,
    #     end=17,
    #     rot_count=5
    # )

    inst = fac.create_test_instance(
        protein="5PTI",
        start=13,
        end=17,
        rot_count=5
    )

    results_file = "5pti/metrics/5PTI_13_17_5_12_layers.npz"

    one_body, two_body, pose, residue_library, ig, rot_sets, scorefxn = extract_one_body_energies_from_instance(inst, logging.getLogger("rescoring_phase2"))
    sample_func, basic_params = get_sample_function_for_phase_2(logging.getLogger("rescoring_phase2"), one_body, two_body, shots=1000)
    best_params = extract_best_qaoa_params(results_file)
    processed_results, unique_bitstrings = get_and_process_shot_results(sample_func, best_params, logging.getLogger("rescoring_phase2"))

    scored_conformations = evaluate_pyrosetta_energies(unique_bitstrings, pose, scorefxn, residue_library, basic_params)
    base_conformation = RescoringConformation(bitstring=None, pose=pose, biological_energy=np.float64(scorefxn(pose)))
    compare_scoring_results(scored_conformations, base_conformation, logger)

    best_conf_per_seed = extract_best_conformation_for_seeds(processed_results, scored_conformations)
    epsilon_value = 1.5
    for seed, conf in best_conf_per_seed.items():
        logger.info(f"Seed {seed}: Best Bitstring: {conf.bitstring}, Biological Energy: {conf.biological_energy:.4f}, Energy Difference: {conf.energy_diff:.4f}")
        if abs(conf.energy_diff) < epsilon_value:
            logger.info(f"Seed {seed} is too close to the original pose (Energy difference {conf.energy_diff:.4f} < {epsilon_value}), it's a tie.")
        elif conf.energy_diff < 0:
            logger.info(f"Seed {seed} has a better conformation than the original pose (Energy difference {conf.energy_diff:.4f} < 0), it's a success!")
        else:
            logger.info(f"Seed {seed} has a worse conformation than the original pose (Energy difference {conf.energy_diff:.4f} > 0), it's a failure.")



