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

def layered_run(cost_func, sample_func, target_indices, valid_conformations, num_qubits, qaoa_layers, result_path, previous_params=None, disable_jit=False):
    # previous_params=None
    logger = logging.getLogger(f"qaoa.main.p_{qaoa_layers}")

    qaoa_params = QAOAParams(layers=qaoa_layers, optimiser_stepsize=BASE_STEPSIZE, epochs=BASE_EPOCHS//10) if previous_params is None else (
        QAOAParams(layers=qaoa_layers, optimiser_stepsize=BASE_STEPSIZE/10, epochs=BASE_EPOCHS//10))
    logger.info(f"Starting layered QAOA for {num_qubits} qubits with parameters: {qaoa_params}")

    max_memory_gb = 10 # ADJUST THIS BASED ON YOUR GPU CAPACITY
    seed_versions = list(range(5)) # ADJUST THIS BASED ON YOUR DESIRED NUMBER OF SEEDS

    start_time = time.perf_counter()
    if disable_jit:
        logger.info("[JIT Config] Running with JIT disabled")
        with jax.disable_jit():
            final_probs, cost_history, optimized_params = batched_qaoa(cost_func, sample_func, qaoa_params, seed_versions, num_qubits, max_memory_gb, logger, previous_params=previous_params)
    else:
        final_probs, cost_history, optimized_params = batched_qaoa(cost_func, sample_func, qaoa_params, seed_versions, num_qubits, max_memory_gb, logger, previous_params=previous_params)
    total_time_taken = time.perf_counter() - start_time

    success_metric = calculate_epsilon_success(final_probs, target_indices)
    target_probs, conf_prob_map, best_idx = extract_metrics_for_serialization(final_probs, target_indices, valid_conformations)

    np.savez(result_path,
             cost_history=cost_history,

             target_probs=target_probs,

             target_indices=target_indices,
             best_target_index=best_idx,
             conformation_map=np.array(conf_prob_map, dtype=object)
    )

    logger.info(f"Saved results to {result_path} - took {total_time_taken:.2f} seconds | avg. {total_time_taken/len(seed_versions):.3f} seconds over {len(seed_versions)} seeds")
    logger.debug(f"Success Metric (P_success) per seed: {success_metric}")

    return optimized_params, total_time_taken

def main(file_path, logger, results_dir, disable_jit=False, disable_adjoint=False):
    artifact_base_name = file_path.split("/")[-1].split(".")[0]


    one_body, two_body = load_qaoa_data(file_path)

    basic_params: BasicParams = init_basic_params(one_body)

    coeffs, observables, num_qubits = extract_ising_items(one_body, two_body, logger)
    cost_hamiltonian = qml.dot(coeffs, observables)

    device_type = 'lightning.gpu' if IS_LINUX else 'lightning.qubit'
    dev = get_cached_device(num_qubits, device_type)
    logger.info(f"Running on {device_type} for {num_qubits} qubits")

    cost_func, sample_func = qaoa_func_generator(dev, cost_hamiltonian, ring_xy_mixer_layer, basic_params, disable_adjoint)
    target_indices, valid_conformations = extract_lowest_energy_bitstrings(
        one_body, two_body,
        logger, 1.5, basic_params
    )

    qaoa_layer_tests = [2, 4, 6, 8, 12]
    cached_params = None

    summed_time = 0

    for layers in qaoa_layer_tests:
        result_path = f"{results_dir}/{artifact_base_name}_{layers}_layers.npz"
        cached_params, total_time_taken = layered_run(cost_func, sample_func, target_indices, valid_conformations, num_qubits, layers, result_path, cached_params, disable_jit=disable_jit)
        summed_time += total_time_taken
    return summed_time


if __name__ == '__main__':
    # Run QAOA for these qubit counts
    qubit_counts = [5, 6, 7, 8, 9] #, 9, 10, 11, 12]
    limit_files_per_qubit = 1 # Adjust this to limit the number of files processed per qubit count
    start_file_idx = 4
    all_energy_files = {num_qubits: list(pathlib.Path(f"extraction/alt_energies/{num_qubits}").glob("*.pkl"))for num_qubits in qubit_counts}

    temp_base = "temp3"

    # Limit the number of files processed per qubit count to manage total runtime
    energy_files = {num_qubits: [] for num_qubits in qubit_counts}
    for num_qubits, files in all_energy_files.items():
        if start_file_idx > len(files): continue

        if start_file_idx + limit_files_per_qubit > len(files): energy_files[num_qubits] = files[start_file_idx:]
        else: energy_files[num_qubits] = files[start_file_idx:start_file_idx+limit_files_per_qubit]

    temp_dir = f"{temp_base}/{start_file_idx}"
    results_dir = f"{temp_dir}/qaoa_only_5_8_no_adjoint"
    logger = setup_logging(f"{temp_dir}", "No_adjoint_5_8_qubits")
    pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)

    logger.info("Starting QAOA Runs for qubit counts: " + ", ".join(f"{num_qubits} ({len(files)} files)" for num_qubits, files in all_energy_files.items()))
    logger.info("Starting at file index: " + str(start_file_idx))

    for qubit_count in reversed(qubit_counts): # high to low
        cur_energy_files = energy_files[qubit_count]
        logger.info(f"Processing {len(cur_energy_files)} files for {qubit_count} qubits")
        for energy_file in cur_energy_files:
            logger.info(f"Starting QAOA runs for {energy_file.name}")
            qaoa_time = main(energy_file.as_posix(), logger, results_dir, disable_jit=True, disable_adjoint=True)
            logger.info(f"Completed QAOA runs for {energy_file.name} in {qaoa_time:.2f} seconds - completed ?% of estimated total processing time\n")

    logger.info("\n =============== COMPLETED ALL RUNS WITH NO ADJOINT =============== \n")

    temp_dir = f"{temp_base}/{start_file_idx}"
    results_dir = f"{temp_dir}/qaoa_only_5_8_no_jit"
    logger = setup_logging(f"{temp_dir}", "No_jit_5_8_qubits")
    pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)

    logger.info("Starting QAOA Runs for qubit counts: " + ", ".join(f"{num_qubits} ({len(files)} files)" for num_qubits, files in all_energy_files.items()))
    logger.info("Starting at file index: " + str(start_file_idx))

    for qubit_count in reversed(qubit_counts): # high to low
        cur_energy_files = energy_files[qubit_count]
        logger.info(f"Processing {len(cur_energy_files)} files for {qubit_count} qubits")
        for energy_file in cur_energy_files:
            logger.info(f"Starting QAOA runs for {energy_file.name}")
            qaoa_time = main(energy_file.as_posix(), logger, results_dir, disable_jit=True, disable_adjoint=False)
            logger.info(f"Completed QAOA runs for {energy_file.name} in {qaoa_time:.2f} seconds - completed ?% of estimated total processing time\n")

    logger.info("\n =============== COMPLETED ALL RUNS WITH NO JIT =============== \n")

    temp_dir = f"{temp_base}/{start_file_idx}"
    results_dir = f"{temp_dir}/qaoa_only_5_8_no_batching"
    logger = setup_logging(f"{temp_dir}", "No_batching_5_8_qubits")
    pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)

    logger.info("Starting QAOA Runs for qubit counts: " + ", ".join(
        f"{num_qubits} ({len(files)} files)" for num_qubits, files in all_energy_files.items()))
    logger.info("Starting at file index: " + str(start_file_idx))

    for qubit_count in reversed(qubit_counts):  # high to low
        cur_energy_files = energy_files[qubit_count]
        logger.info(f"Processing {len(cur_energy_files)} files for {qubit_count} qubits")
        for energy_file in cur_energy_files:
            logger.info(f"Starting QAOA runs for {energy_file.name}")
            qaoa_time = main(energy_file.as_posix(), logger, results_dir, disable_jit=False, disable_adjoint=False)
            logger.info(
                f"Completed QAOA runs for {energy_file.name} in {qaoa_time:.2f} seconds - completed ?% of estimated total processing time\n")

    logger.info("\n =============== COMPLETED ALL RUNS WITH NO BATCHING =============== \n")



    # Example usage - adjust paths and parameters as needed
    # energy_files = list(pathlib.Path("energies/small").glob("*.pkl"))
    # first = False
    # for energy_file in energy_files:
    #     # if not first:
    #     #     first = True
    #     #     continue
    #     logger.info(f"Starting QAOA runs for {energy_file.name}")
    #     start = time.perf_counter()
    #     main(energy_file.as_posix(), logger, results_dir)
    #     elapsed = time.perf_counter() - start
    #
    #     logger.info(f"Completed QAOA runs for {energy_file.name} in {elapsed:.2f} seconds\n")