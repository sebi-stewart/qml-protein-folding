import logging
import os
import pathlib
import pickle
import time

import numpy as np

from qaoa.devices import get_cached_device
from qaoa.h_mixer import ring_xy_mixer_layer
from logging_setup import setup_logging
from qaoa.execution import batched_qaoa, sequential_qaoa
from qaoa.generators import qaoa_func_generator
from qaoa.hamiltonians import extract_ising_items
from constants import IS_LINUX

import pennylane as qml

from qaoa.metrics import calculate_epsilon_success, extract_metrics_for_serialization
from qaoa.objects import QAOAParams, init_basic_params, BasicParams
from qaoa.scoring import extract_lowest_energy_bitstrings

QAOA_EXECUTION_MODE = os.getenv("QAOA_EXECUTION_MODE", "batched").lower()
QAOA_MAX_MEMORY_GB = float(os.getenv("QAOA_MAX_MEMORY_GB", "60"))
QAOA_NUM_SEEDS = int(os.getenv("QAOA_NUM_SEEDS", "30"))


def _run_qaoa(cost_func, sample_func, qaoa_params, seed_versions, num_qubits, max_memory_gb, logger, previous_params):
    if QAOA_EXECUTION_MODE == "sequential":
        return sequential_qaoa(
            cost_func,
            sample_func,
            qaoa_params,
            seed_versions,
            num_qubits,
            max_memory_gb,
            logger,
            previous_params=previous_params,
        )
    return batched_qaoa(
        cost_func,
        sample_func,
        qaoa_params,
        seed_versions,
        num_qubits,
        max_memory_gb,
        logger,
        previous_params=previous_params,
    )

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

    max_memory_gb = QAOA_MAX_MEMORY_GB
    seed_versions = list(range(QAOA_NUM_SEEDS))

    final_probs, cost_history, optimized_params = _run_qaoa(
        cost_func,
        sample_func,
        qaoa_params,
        seed_versions,
        num_qubits,
        max_memory_gb,
        logger,
        previous_params,
    )

    success_metric = calculate_epsilon_success(final_probs, target_indices)
    target_probs, conf_prob_map, best_idx = extract_metrics_for_serialization(final_probs, target_indices, valid_conformations)

    np.savez(result_path,
             cost_history=cost_history,

             target_probs=target_probs,

             target_indices=target_indices,
             best_target_index=best_idx,
             conformation_map=np.array(conf_prob_map, dtype=object),

             optimized_params=optimized_params,
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
    logger.info(
        f"JIT backend | mode={QAOA_EXECUTION_MODE} | seeds={QAOA_NUM_SEEDS}"
    )

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

def find_limit_energy_files(qubit_counts, limit_files_per_qubit, start_file_idx, source_folder="extraction/alt_energies"):
    all_energy_files = {num_qubits: list(pathlib.Path(f"{source_folder}/{num_qubits}").glob("*.pkl")) for num_qubits in qubit_counts}
    energy_files = {num_qubits: [] for num_qubits in qubit_counts}
    for num_qubits, files in all_energy_files.items():
        if start_file_idx > len(files): continue

        if start_file_idx + limit_files_per_qubit > len(files): energy_files[num_qubits] = files[start_file_idx:]
        else: energy_files[num_qubits] = files[start_file_idx:start_file_idx+limit_files_per_qubit]

    return energy_files

MULTIPLICATIVE_FACTOR = 20
EXPONENTIAL_FACTOR = 1.41
ADDITIVE_FACTOR = 50

def define_total_processing_estimate(energy_files):
    return sum(len(files) * (ADDITIVE_FACTOR + MULTIPLICATIVE_FACTOR * (EXPONENTIAL_FACTOR ** num_qubits)) for num_qubits, files in energy_files.items())

if __name__ == '__main__':
    # Run QAOA for these qubit counts
    qubit_counts = [5, 6, 7, 8, 9, 10]
    limit_files_per_qubit = 5 # Adjust this to limit the number of files processed per qubit count
    start_file_idx = 0
    temp_base = "Phase2"

    # Limit the number of files processed per qubit count to manage total runtime
    energy_files = find_limit_energy_files(qubit_counts, limit_files_per_qubit, start_file_idx, source_folder="extraction/Phase2_energies")
    total_processing_estimate = define_total_processing_estimate(energy_files)


    temp_dir = f"{temp_base}/{start_file_idx}"
    results_dir = f"{temp_dir}/phase2_5_to_10_qubits"
    logger = setup_logging(f"{temp_dir}", "phase2_5_to_10_qubits_qaoa_runs")
    pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)

    logger.info("Starting QAOA Runs for qubit counts: " + ", ".join(
        f"{num_qubits} ({len(files)} files)" for num_qubits, files in energy_files.items()))

    current_processed = 0
    for qubit_count in reversed(qubit_counts):  # high to low
        cur_energy_files = energy_files[qubit_count]
        logger.info(f"Processing {len(cur_energy_files)} files for {qubit_count} qubits")
        for energy_file in cur_energy_files:
            logger.info(f"Starting QAOA runs for {energy_file.name}")
            start = time.perf_counter()
            main(energy_file.as_posix(), logger, results_dir)
            qaoa_time = time.perf_counter() - start

            current_processed += ADDITIVE_FACTOR + (MULTIPLICATIVE_FACTOR * (EXPONENTIAL_FACTOR ** qubit_count))
            logger.info(f"Completed QAOA runs for {energy_file.name} in {qaoa_time:.2f} seconds - completed {current_processed/total_processing_estimate*100:.3f}% of estimated total processing time\n")

    logger.info("\n =============== COMPLETED ALL RUNS WITH NO BATCHING =============== \n")