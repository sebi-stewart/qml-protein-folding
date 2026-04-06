import logging
import pathlib
import pickle
import time

import numpy as np

from qaoa.devices import get_cached_device
from qaoa.h_mixer import ring_xy_mixer_layer
from logging_setup import setup_logging
from qaoa.execution import batched_qaoa
from qaoa.generators import qaoa_func_generator
from qaoa.hamiltonians import extract_ising_items
from constants import IS_LINUX

import pennylane as qml

from qaoa.metrics import calculate_epsilon_success, extract_metrics_for_serialization
from qaoa.objects import QAOAParams, init_basic_params, BasicParams
from qaoa.scoring import extract_lowest_energy_bitstrings

import jax
jax.config.update("jax_enable_x64", True)

BASE_EPOCHS = 150
BASE_STEPSIZE = 0.01

def load_qaoa_data(source_path):
    with open(source_path, 'rb') as f:
        energies = pickle.load(f)
    return energies['one_body'], energies['two_body']

def layered_run(cost_func, sample_func, target_indices, valid_conformations, num_qubits, qaoa_layers, result_path, previous_params=None):
    # previous_params=None
    logger = logging.getLogger(f"qaoa.main.p_{qaoa_layers}")

    qaoa_params = QAOAParams(layers=qaoa_layers, optimiser_stepsize=BASE_STEPSIZE, epochs=BASE_EPOCHS) if previous_params is None else (
        QAOAParams(layers=qaoa_layers, optimiser_stepsize=BASE_STEPSIZE/10, epochs=BASE_EPOCHS))
    logger.info(f"Starting layered QAOA for {num_qubits} qubits with parameters: {qaoa_params}")

    max_memory_gb = 10 # ADJUST THIS BASED ON YOUR GPU CAPACITY
    seed_versions = list(range(30)) # ADJUST THIS BASED ON YOUR DESIRED NUMBER OF SEEDS

    start_time = time.perf_counter()
    final_probs, cost_history, optimized_params = batched_qaoa(cost_func, sample_func, qaoa_params, seed_versions, num_qubits, max_memory_gb, logger, previous_params=previous_params)

    success_metric = calculate_epsilon_success(final_probs, target_indices)
    target_probs, conf_prob_map, best_idx = extract_metrics_for_serialization(final_probs, target_indices, valid_conformations)

    np.savez(result_path,
             cost_history=cost_history,

             target_probs=target_probs,

             target_indices=target_indices,
             best_target_index=best_idx,
             conformation_map=np.array(conf_prob_map, dtype=object)
    )

    total_time_taken = time.perf_counter() - start_time
    logger.info(f"Saved results to {result_path} - took {total_time_taken:.2f} seconds | avg. {total_time_taken/len(seed_versions):.3f} seconds over {len(seed_versions)} seeds")
    logger.debug(f"Success Metric (P_success) per seed: {success_metric}")

    return optimized_params

def main(file_path, logger, results_dir):
    artifact_base_name = file_path.split("/")[-1].split(".")[0]


    one_body, two_body = load_qaoa_data(file_path)

    basic_params: BasicParams = init_basic_params(one_body)

    coeffs, observables, num_qubits = extract_ising_items(one_body, two_body, logger)
    cost_hamiltonian = qml.dot(coeffs, observables)

    device_type = 'lightning.gpu' if IS_LINUX else 'lightning.qubit'
    dev = get_cached_device(num_qubits, device_type)
    logger.info(f"Running on {device_type} for {num_qubits} qubits")

    cost_func, sample_func = qaoa_func_generator(dev, cost_hamiltonian, ring_xy_mixer_layer, basic_params)
    target_indices, valid_conformations = extract_lowest_energy_bitstrings(
        one_body, two_body,
        logger, 1.5, basic_params
    )

    qaoa_layer_tests = [2, 4, 6, 8, 12]
    cached_params = None

    for layers in qaoa_layer_tests:
        result_path = f"{results_dir}/{artifact_base_name}_{layers}_layers.npz"
        cached_params = layered_run(cost_func, sample_func, target_indices, valid_conformations, num_qubits, layers, result_path, cached_params)


if __name__ == '__main__':
    results_dir = "qaoa_results_9_warm_start_at_2"
    logger = setup_logging("new_runs_qaoa", "warm_start_at_2")
    pathlib.Path(results_dir).mkdir(exist_ok=True)



    # Example usage - adjust paths and parameters as needed
    energy_files = list(pathlib.Path("energies/small").glob("*.pkl"))
    first = False
    for energy_file in energy_files:
        # if not first:
        #     first = True
        #     continue
        logger.info(f"Starting QAOA runs for {energy_file.name}")
        start = time.perf_counter()
        main(energy_file.as_posix(), logger, results_dir)
        elapsed = time.perf_counter() - start

        logger.info(f"Completed QAOA runs for {energy_file.name} in {elapsed:.2f} seconds\n")





