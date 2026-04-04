import logging
import pathlib
import pickle
import time

import numpy as np

from custom_qaoa import get_cached_device
from h_mixer import ring_xy_mixer_layer
from logging_setup import setup_logging
from misc import init_basic_params, BasicParams
from qaoa.execution import batched_qaoa
from qaoa.generators import qaoa_func_generator
from qaoa.hamiltonians import extract_ising_items
from constants import IS_LINUX

import pennylane as qml

from qaoa.metrics import calculate_epsilon_success, extract_metrics_for_serialization
from qaoa.objects import QAOAParams
from qaoa.scoring import extract_lowest_energy_bitstrings


def main(h_linear, J_quadratic, qaoa_layers, artifact_path):
    basic_params: BasicParams = init_basic_params(h_linear)

    logger = logging.getLogger("qaoa.main")
    coeffs, observables, num_qubits = extract_ising_items(h_linear, J_quadratic, logger)
    cost_hamiltonian = qml.dot(coeffs, observables)

    device_type = 'lightning.gpu' if IS_LINUX else 'lightning.qubit'
    dev = get_cached_device(num_qubits, device_type)
    logger.debug(f"Running on {device_type} for {num_qubits} qubits")

    qaoa_params = QAOAParams(layers=qaoa_layers, optimiser_stepsize=0.01, epochs=150)
    max_memory_gb = 10 # ADJUST THIS BASED ON YOUR GPU CAPACITY
    seed_versions = list(range(30)) # ADJUST THIS BASED ON YOUR DESIRED NUMBER OF SEEDS

    start_time = time.perf_counter()
    cost_func, sample_func = qaoa_func_generator(dev, cost_hamiltonian, ring_xy_mixer_layer, basic_params)
    final_probs, cost_history = batched_qaoa(cost_func, sample_func, qaoa_params, seed_versions, num_qubits, max_memory_gb, logger)

    target_indices, valid_conformations = extract_lowest_energy_bitstrings(
        h_linear, J_quadratic,
        logger, 1e-6, basic_params
    )

    success_metric = calculate_epsilon_success(final_probs, target_indices)
    target_probs, conf_prob_map, best_idx = extract_metrics_for_serialization(final_probs, target_indices, valid_conformations)

    np.savez(artifact_path,
             cost_history=cost_history,

             target_probs=target_probs,

             target_indices=target_indices,
             best_target_index=best_idx,
             conformation_map=np.array(conf_prob_map, dtype=object)
    )

    total_time_taken = time.perf_counter() - start_time
    logger.info(f"Saved results to {artifact_path} - took {total_time_taken:.2f} seconds | avg. {total_time_taken/len(seed_versions):.3f} seconds over {len(seed_versions)} seeds")
    logger.debug(f"Success Metric (P_success) per seed: {success_metric}")

def load_qaoa_data(source_path):
    with open(source_path, 'rb') as f:
        energies = pickle.load(f)
    return energies['one_body'], energies['two_body']

if __name__ == '__main__':
    setup_logging("new_runs_qaoa")

    pathlib.Path("qaoa_results").mkdir(exist_ok=True)

    # Example usage - adjust paths and parameters as needed
    h_linear, J_quadratic = load_qaoa_data("../extraction/energies/5PTI_20_24_4.pkl")
    main(h_linear, J_quadratic, qaoa_layers=3, artifact_path="qaoa_results/5PTI_qaoa_results.npz")





