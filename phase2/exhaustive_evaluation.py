import logging

import numpy as np

from extraction.rotamers import TrackedResidue
from phase2.biological_rescoring import evaluate_singular_pyrosetta_energy, compare_scoring_results
from phase2.objects import RescoringConformation
from qaoa.objects import BasicParams
from qaoa.scoring import get_valid_bitstrings_matrix

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

def run_exhaustive_evaluation(logger: logging.Logger, basic_params, pose, scorefxn, residue_library, base_conformation):
    X_matrix, indices = get_valid_bitstrings_matrix(basic_params, logger)
    logger.info(
        "Exhaustively evaluating all conformations from all possible bitstrings (not just the best per seed) to get a more complete picture of the energy landscape...")
    logger.info(f"Total unique bitstrings to evaluate: {len(X_matrix)}")
    logger.debug(X_matrix)

    exhaustive_conf = exhaustively_evaluate_all_conformations(X_matrix, pose, scorefxn, residue_library, basic_params)
    logger.info(
        "Completed exhaustive evaluation of all conformations. Comparing results to identify any additional winning conformations that were not sampled by the QAOA runs...")
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
    return {"best": best_exhaustive_conf, "worst": worst_exhaustive_conf}