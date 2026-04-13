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

    qaoa_params = QAOAParams(layers=qaoa_layers, optimiser_stepsize=BASE_STEPSIZE, epochs=BASE_EPOCHS//10) if previous_params is None else (
        QAOAParams(layers=qaoa_layers, optimiser_stepsize=BASE_STEPSIZE/10, epochs=BASE_EPOCHS//10))
    logger.info(f"Starting layered QAOA for {num_qubits} qubits with parameters: {qaoa_params}")

    max_memory_gb = 10 # ADJUST THIS BASED ON YOUR GPU CAPACITY
    seed_versions = list(range(30)) # ADJUST THIS BASED ON YOUR DESIRED NUMBER OF SEEDS

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
    # Run QAOA for these qubit counts
    qubit_counts = [14, 18, 22]
    limit_files_per_qubit = 5 # Adjust this to limit the number of files processed per qubit count
    start_file_idx = 0
    all_energy_files = {num_qubits: list(pathlib.Path(f"extraction/alt_energies/{num_qubits}").glob("*.pkl"))for num_qubits in qubit_counts}

    temp_base = "temp4"

    # Limit the number of files processed per qubit count to manage total runtime
    energy_files = {num_qubits: [] for num_qubits in qubit_counts}
    for num_qubits, files in all_energy_files.items():
        if start_file_idx > len(files): continue

        if start_file_idx + limit_files_per_qubit > len(files): energy_files[num_qubits] = files[start_file_idx:]
        else: energy_files[num_qubits] = files[start_file_idx:start_file_idx+limit_files_per_qubit]

    # Assume exponential time increase, 1.5x per additional qubit as a rough estimate, and adjust the order of processing accordingly
    multiplicative_factor = 20
    exponential_factor = 1.41
    additive_factor = 50

    total_processing_estimate = sum(len(files) * (additive_factor + multiplicative_factor * (exponential_factor ** num_qubits)) for num_qubits, files in energy_files.items())


    temp_dir = f"{temp_base}/{start_file_idx}"
    results_dir = f"{temp_dir}/qaoa_only_5_8_no_batching"
    logger = setup_logging(f"{temp_dir}", "No_batching_5_8_qubits")
    pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)

    logger.info("Starting QAOA Runs for qubit counts: " + ", ".join(
        f"{num_qubits} ({len(files)} files)" for num_qubits, files in all_energy_files.items()))
    logger.info("Starting at file index: " + str(start_file_idx))

    current_processed = 0
    for qubit_count in reversed(qubit_counts):  # high to low
        cur_energy_files = energy_files[qubit_count]
        logger.info(f"Processing {len(cur_energy_files)} files for {qubit_count} qubits")
        for energy_file in cur_energy_files:
            logger.info(f"Starting QAOA runs for {energy_file.name}")
            start = time.perf_counter()
            main(energy_file.as_posix(), logger, results_dir)
            qaoa_time = time.perf_counter() - start

            current_processed += additive_factor + (multiplicative_factor * (exponential_factor ** qubit_count))
            logger.info(f"Completed QAOA runs for {energy_file.name} in {qaoa_time:.2f} seconds - completed {current_processed/total_processing_estimate*100:.3f}% of estimated total processing time\n")

    logger.info("\n =============== COMPLETED ALL RUNS WITH NO BATCHING =============== \n")