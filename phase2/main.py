import argparse
import os, sys
sys.path.append(os.getcwd())  # Ensures import work fine when running from the root directory of the project
import utils.make_paths_absolute # Important for file paths
import fastparquet as fp

from phase2.biological_rescoring import evaluate_pyrosetta_energies, compare_scoring_results
from phase2.exhaustive_evaluation import run_exhaustive_evaluation
from phase2.qaoa_shots import take_qaoa_shots
import logging
import pathlib

import pyrosetta

from extraction.initialisation import initialize_rosetta
from extraction.main import TestInstanceFactory, ExtractionTestInstance, run_pyrosetta_obj_extraction, \
    from_energies_to_tensors
from utils.logging_setup import setup_logging
from phase2.objects import RescoringConformation

import numpy as np
import pandas as pd
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


def extract_best_qaoa_params(qaoa_file_path: str):
    assert qaoa_file_path.endswith(".npz"), "Expected a .npz file containing the QAOA results"
    data = np.load(qaoa_file_path, allow_pickle=False)

    # Act as an oracle to extract the relevant data for the next phase
    # We refrain from looking at the actual values, but we know the structure of the saved data from the layered_run function
    optimised_params = data['optimized_params']
    return optimised_params


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


def main(logger: logging.Logger, fac: TestInstanceFactory, results_file, input_pdb: str, exhaustive_evaluation=False,):
    inst = fac.create_test_instance_from_results_file(results_file, input_pdb)

    logger.info(f"Extracting one-body and two-body energies for instance {inst.test_name}...")
    phase2_rescoring_logger = logging.getLogger("qaoa.rescoring_phase2")
    hidden_logger = logging.getLogger("qaoa.hidden")
    one_body, two_body, pose, residue_library, ig, rot_sets, scorefxn = extract_one_body_energies_from_instance(inst, hidden_logger)
    hidden_logger.info(f"Extracted residues: {residue_library.keys()} --- {residue_library}")
    hidden_logger.info(f"Extracted one-body energies: {one_body.keys()} --- {one_body}")

    logger.info("Extracting best QAOA parameters from file...")
    best_params = extract_best_qaoa_params(results_file)
    processed_results, unique_bitstrings, basic_params = take_qaoa_shots(one_body, two_body, best_params, phase2_rescoring_logger, hidden_logger)

    logger.info(f"Evaluating PyRosetta energies for unique bitstrings... Total unique conformations to evaluate: {len(unique_bitstrings)}")
    scored_conformations = evaluate_pyrosetta_energies(unique_bitstrings, pose, scorefxn, residue_library, basic_params)
    base_conformation = RescoringConformation(bitstring=None, pose=pose, biological_energy=np.float64(scorefxn(pose)))
    compare_scoring_results(scored_conformations, base_conformation, logger)

    best_conf_per_seed = extract_best_conformation_for_seeds(processed_results, scored_conformations)
    epsilon_value = 1.5

    results = []
    for seed, conf in best_conf_per_seed.items():
        cur_result = {
            'seed': seed,
            'bitstring': list(conf.bitstring),
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
    logger.info(f"Summary of results: {classification_counts}")

    if not exhaustive_evaluation:
        logger.debug("Skipping exhaustive evaluation of all conformations. To enable this, set exhaustive_evaluation=True when calling main().")
        return results, best_conf_per_seed
    return results, run_exhaustive_evaluation(logger, basic_params, pose, scorefxn, residue_library, base_conformation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pdb", type=str, default="inputs/structural_files/AF-P00974-F1-model_v6.pdb", help="Path to the input PDB file")
    parser.add_argument("--qaoa_results_folder", type=str, default="outputs/npz_files/test", help="Path to the folder containing QAOA results files (NPZ format)")
    parser.add_argument("--file_search_pattern", type=str, default="*_12_layers.npz", help="Pattern to search for QAOA result files in the specified folder")

    parser.add_argument("--log_dir", type=str, default="outputs/logs", help="Directory to save the logs")
    parser.add_argument("--log_file", type=str, default="results_rescoring", help="Name of the log file")

    parser.add_argument("--exhaustive_evaluation", action="store_true", help="Whether to perform exhaustive evaluation of all conformations, and save the best and worst poses to PDB files. If set, this will override the --save_best_pose flag.")
    parser.add_argument("--parquet_output_dir", type=str, default="outputs/dataframes/test", help="Path to save the aggregated results in Parquet format")
    parser.add_argument("--parquet_file_name", type=str, default="test.parquet", help="Name of the Parquet file to save the aggregated results")
    parser.add_argument("--save_best_pose", action="store_true", help="Whether to save the best pose to PDB (ignored during exhaustive evaluation)")
    parser.add_argument("--pose_output_dir", type=str, default="outputs/structural_files", help="Directory to save the best and worst poses in PDB format (ignored if both exhaustive evaluation and save_best_pose are disabled)")

    args = parser.parse_args()

    logger = setup_logging(args.log_dir, args.log_file)
    initialize_rosetta(pyrosetta, extra_flags="-mute all")
    fac = TestInstanceFactory()

    if args.exhaustive_evaluation or args.save_best_pose:
        pathlib.Path(args.pose_output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for file in pathlib.Path(args.qaoa_results_folder).rglob(args.file_search_pattern):
        if "old" in str(file).lower() or "partial" in str(file).lower(): continue

        logger.info(f"Processing file: {str(file)}")
        cur_results, conformations = main(logger, fac, results_file=str(file), input_pdb=args.input_pdb, exhaustive_evaluation=args.exhaustive_evaluation)
        if args.exhaustive_evaluation:
            worst_conf = conformations["worst"]
            best_conf = conformations["best"]
            logger.info(f"Best and worst conformations from exhaustive evaluation for {str(file)} - Dumping poses to best_pose_{file.stem}.pdb and worst_conformation_{file.stem}.pdb for further analysis. Located in {args.pose_output_dir}.")
            best_conf.pose.dump_pdb(f"{args.pose_output_dir}/best_pose_{file.stem}.pdb")
            worst_conf.pose.dump_pdb(f"{args.pose_output_dir}/worst_conformation_{file.stem}.pdb")
        elif args.save_best_pose:
            best_conf = min(conformations.values(), key=lambda conf: conf.energy_diff)
            logger.info(f"Best conformation for {str(file)}: Bitstring: {best_conf.bitstring}, Biological Energy: {best_conf.biological_energy:.4f}, Energy Difference: {best_conf.energy_diff:.4f}. Dumped pose to best_pose_{file.stem}.pdb for further analysis. Located in {args.pose_output_dir}.")
            best_conf.pose.dump_pdb(f"{args.pose_output_dir}/best_pose_{file.stem}.pdb")

        results.extend(cur_results)

        logger.info(f"Completed processing for {str(file)}. Current aggregated results count: {len(results)}\n\n\n")

    output_dir = pathlib.Path(args.parquet_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(output_dir.joinpath(args.parquet_file_name), engine="fastparquet")