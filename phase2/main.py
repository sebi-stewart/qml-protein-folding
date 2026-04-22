import logging
from dataclasses import dataclass
import pathlib

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
import pandas as pd

from qaoa.scoring import _get_valid_bitstrings_matrix

from collections import Counter


def extract_one_body_energies_from_instance(inst: ExtractionTestInstance, logger: logging.Logger):
    pose, residue_library, ig, rot_sets, scorefxn = run_pyrosetta_obj_extraction(
        inst.pose_func,
        logger=logger,
        n=inst.rotamer_count,
        active_start=inst.residue_start,
        active_end=inst.residue_end)

    one_body, two_body, global_offset = from_energies_to_tensors(residue_library, ig)
    return one_body, two_body, pose, residue_library, ig, rot_sets, scorefxn

qubit_to_shot_map = {
    5: 10, 6: 10, 7: 100, 8: 10, 9: 10, 10: 100, 11: 100, 12: 100, 13: 100, 14: 100, 18: 100, 22: 400
}

def get_sample_function_for_phase_2(logger: logging.Logger, one_body, two_body):
    basic_params: BasicParams = init_basic_params(one_body)
    num_qubits = basic_params.num_qubits
    shots = qubit_to_shot_map.get(num_qubits, 500)  # Default to 500 shots if not specified

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
        unique_shots = set(tuple(map(int, shot)) for shot in shots)
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

def exhaustively_evaluate_all_conformations(unique_bitstrings, original_pose, scorefxn, residue_library: dict[int, TrackedResidue], params: BasicParams):
    conformations = []
    for nd_bitstring in unique_bitstrings:
        bitstring = list(map(int, nd_bitstring))
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

def compare_scoring_results(scored_conformations: list[RescoringConformation], base_conformation: RescoringConformation, logger: logging.Logger):
    # Compare the biological energies of the new conformations with the original pose
    for conf in scored_conformations:
        energy_diff = conf.biological_energy - base_conformation.biological_energy
        conf.energy_diff = energy_diff
        logger.debug(f"Bitstring: {conf.bitstring}, Biological Energy: {conf.biological_energy:.4f}, Energy Difference: {energy_diff:.4f}")

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

def main(logger: logging.Logger, fac: TestInstanceFactory, results_file: str = "0/phase2_5_to_10_qubits/AF-5PTI_52_55_4_12_layers.npz", exhaustive_evaluation=False):
    stripped_file_name = results_file.split("/")[-1].split(".")[0]
    test_name, start_str, end_str, rot_count_str, _, _ = stripped_file_name.split("_")
    start, end, rot_count = int(start_str), int(end_str), int(rot_count_str)

    inst = fac.create_test_instance_from_func(
        pose_func=lambda: pyrosetta.pose_from_pdb("../pdb_files/AF-P00974-F1-model_v6.pdb"),
        test_name=test_name,
        start=start,
        end=end,
        rot_count=rot_count
    )

    logger.info(f"Extracting one-body and two-body energies for instance {inst.test_name}...")
    phase2_rescoring_logger = logging.getLogger("qaoa.rescoring_phase2")
    hidden_logger = logging.getLogger("qaoa.hidden")
    one_body, two_body, pose, residue_library, ig, rot_sets, scorefxn = extract_one_body_energies_from_instance(inst, hidden_logger)
    hidden_logger.info(f"Extracted residues: {residue_library.keys()} --- {residue_library}")
    hidden_logger.info(f"Extracted one-body energies: {one_body.keys()} --- {one_body}")

    sample_func, basic_params = get_sample_function_for_phase_2(hidden_logger, one_body, two_body)

    logger.info("Extracting best QAOA parameters from file...")
    best_params = extract_best_qaoa_params(results_file)
    processed_results, unique_bitstrings = get_and_process_shot_results(sample_func, best_params, phase2_rescoring_logger)

    logger.info(f"Evaluating PyRosetta energies for unique bitstrings... Total unique conformations to evaluate: {len(unique_bitstrings)}")
    scored_conformations = evaluate_pyrosetta_energies(unique_bitstrings, pose, scorefxn, residue_library, basic_params)
    base_conformation = RescoringConformation(bitstring=None, pose=pose, biological_energy=np.float64(scorefxn(pose)))
    compare_scoring_results(scored_conformations, base_conformation, logger)

    best_conf_per_seed = extract_best_conformation_for_seeds(processed_results, scored_conformations)
    epsilon_value = 1.5

    results = []
    win_count, tie_count, loss_count = 0, 0, 0
    for seed, conf in best_conf_per_seed.items():
        cur_result = {
            'seed': seed,
            'bitstring': conf.bitstring,
            'biological_energy': conf.biological_energy,
            'energy_diff': conf.energy_diff,
            'protein': inst.test_name.split("_")[0],
            'residues': f"{inst.residue_start}_{inst.residue_end}",
            'residue_count': inst.residue_end - inst.residue_start + 1,
            'rotamers': inst.rotamer_count,
            'num_qubits': basic_params.num_qubits,
        }

        if abs(conf.energy_diff) <= epsilon_value:
            cur_result["classification"] = 'tie'
        elif conf.energy_diff < 0:
            cur_result["classification"] = 'win'
        else:
            cur_result["classification"] = 'loss'
        results.append(cur_result)

    classification_counts = Counter(result['classification'] for result in results)
    logger.info(f"Summary of results: {classification_counts} losses out of {len(best_conf_per_seed)} seeds.")

    if not exhaustive_evaluation:
        logger.debug("Skipping exhaustive evaluation of all conformations. To enable this, set exhaustive_evaluation=True when calling main().")
        return results

    X_matrix, indices = _get_valid_bitstrings_matrix(basic_params, logger)
    logger.info("Exhaustively evaluating all conformations from all possible bitstrings (not just the best per seed) to get a more complete picture of the energy landscape...")
    logger.info(f"Total unique bitstrings to evaluate: {len(X_matrix)}")
    logger.debug(X_matrix)

    exhaustive_conf = exhaustively_evaluate_all_conformations(X_matrix, pose, scorefxn, residue_library, basic_params)
    logger.info("Completed exhaustive evaluation of all conformations. Comparing results to identify any additional winning conformations that were not sampled by the QAOA runs...")
    compare_scoring_results(exhaustive_conf, base_conformation, logger)
    logger.info("Conformations with better energy than the original pose:")
    exhaustive_conf = sorted(exhaustive_conf, key=lambda conf: conf.energy_diff)
    for conf in exhaustive_conf:
        if conf.energy_diff < 0:
            logger.info(
                f"Bitstring: {conf.bitstring}, Biological Energy: {conf.biological_energy:.4f}, Energy Difference: {conf.energy_diff:.4f}")
        else:
            break
    logger.info("Best conformation from the exhaustive search:")
    best_exhaustive_conf = min(exhaustive_conf, key=lambda conf: conf.energy_diff)
    worst_exhaustive_conf = max(exhaustive_conf, key=lambda conf: conf.energy_diff)
    logger.info(
        f"Bitstring: {best_exhaustive_conf.bitstring}, Biological Energy: {best_exhaustive_conf.biological_energy:.4f}, Energy Difference: {best_exhaustive_conf.energy_diff:.4f}")
    logger.info(
        f"Worst Conformation - Bitstring: {worst_exhaustive_conf.bitstring}, Biological Energy: {worst_exhaustive_conf.biological_energy:.4f}, Energy Difference: {worst_exhaustive_conf.energy_diff:.4f}. Dumped pose to worst_conformation.pdb for further analysis."
    )
    return results, worst_exhaustive_conf

if __name__ == "__main__":
    logger = setup_logging("rescoring_phase2", "5PTI")
    initialize_rosetta(pyrosetta, extra_flags="-mute all")
    fac = TestInstanceFactory()

    results = []
    worst_confs = []
    for file in pathlib.Path("Oldruns_without_extra_rotamers").rglob("*_12_layers.npz"):
        # if "oldruns" in str(file).lower(): continue
        # if "5PTI_57_60_4_12_layers" not in str(file): continue  # Temporary filter to focus on a specific instance, remove this to run on all files

        logger.info(f"Processing file: {str(file)}")
        cur_results, worst_conf = main(logger, fac, str(file), exhaustive_evaluation=True)
        worst_confs.append(worst_conf)

        results.extend(cur_results)

        logger.info(f"Completed processing for {str(file)}. Current aggregated results count: {len(results)}\n\n\n")

    worst_exhaustive_conf = max(worst_confs, key=lambda conf: conf.energy_diff)
    logger.info("Overall Worst Conformation Across All Instances:")
    logger.info(
        f"Bitstring: {worst_exhaustive_conf.bitstring}, Biological Energy: {worst_exhaustive_conf.biological_energy:.4f}, Energy Difference: {worst_exhaustive_conf.energy_diff:.4f}. Dumped pose to worst_conformation.pdb for further analysis."
    )
    worst_pose = worst_exhaustive_conf.pose
    worst_pose.dump_pdb("worst_conformation.pdb")


    pd.DataFrame(results).to_pickle("phase2_old_runs2.pkl")

