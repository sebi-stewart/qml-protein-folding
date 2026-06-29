"""
Main QAOA runner and orchestration utilities to launch layered QAOA experiments.
This module loads reduced Hamiltonians, prepares devices and executes layered runs while saving results.
"""

import os, sys

from qaoa.file_sourcing import find_limit_energy_files, define_processing_estimate

sys.path.append(os.getcwd())  # Ensures import work fine when running from the root directory of the project
import utils.make_paths_absolute # Important for file paths # noqa

from utils.loading_energy_mappings import load_energy_mappings
import logging
import pathlib
import time

import numpy as np

from qaoa.devices import get_cached_device
from qaoa.h_mixer import ring_xy_mixer_layer
from utils.logging_setup import setup_logging
from qaoa.execution import batched_qaoa, sequential_qaoa
from qaoa.generators import qaoa_func_generator
from qaoa.hamiltonians import extract_ising_items

import pennylane as qml

from qaoa.metrics import calculate_epsilon_success, extract_metrics_for_serialization
from qaoa.objects import QAOAParams, init_basic_params, BasicParams
from qaoa.scoring import extract_lowest_energy_bitstrings


class Constants:
    _instance = None

    # Input File Vars
    SOURCE_FOLDER: str = "intermediates/energy_mappings/AF-5PTI_moderate_confidence_region"
    START_AT_FILE_IDX: int = 0 # If multiple instances are used to run the same set of files, this can be used to start at a specific file index for each instance
    LIMIT_FILES_PER_QUBIT: int = 10
    QUBIT_COUNTS_TO_ANALYSE: list[int] = [5, 10, 14, 18, 22] # Qubit counts to analyse, can be adjusted to limit the number of qubits analysed for testing purposes

    # QAOA Vars
    BASE_EPOCHS = 150
    BASE_STEPSIZE = 0.01
    QAOA_LAYERS = [2, 4, 6, 8, 12]
    STEPSIZE_REDUCTION_FACTOR = 10
    HIGH_TO_LOW_QUBIT_ORDER = True

    USE_GPU = False

    QAOA_SEED_COUNT = 30
    QAOA_EXECUTION_MODE = "batched"  # Options: "batched" or "sequential"
    QAOA_MAX_MEMORY_GB = 12.0  # Maximum memory in GB


    # Output Vars
    OUTPUTS_FOLDER = "outputs/npz_files"
    LOGS_FOLDER = "outputs/logs"
    LOGS_FILE = "qaoa_results"


    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            # Put any initialization here.
        return cls._instance
    
constants = Constants.instance()

def update_constants(**kwargs) -> None:
    """
    Update Constants instance values conditionally from provided keyword arguments.
    Only updates attributes that are explicitly provided in kwargs.
    
    Args:
        **kwargs: Keyword arguments matching constant names (converted from arg names).
                  For example: source_folder, start_at_file_idx, etc.
                  
    Example:
        update_constants(
            source_folder="new/path",
            base_epochs=200,
            use_gpu=True
        )
    """
    # Map of argparse argument names to Constants class attribute names
    arg_to_constant_map = {
        'source_folder': 'SOURCE_FOLDER',
        'start_at_file_idx': 'START_AT_FILE_IDX',
        'limit_files_per_qubit': 'LIMIT_FILES_PER_QUBIT',
        'qubit_counts_to_analyse': 'QUBIT_COUNTS_TO_ANALYSE',
        'base_epochs': 'BASE_EPOCHS',
        'base_stepsize': 'BASE_STEPSIZE',
        'qaoa_layers': 'QAOA_LAYERS',
        'stepsize_reduction_factor': 'STEPSIZE_REDUCTION_FACTOR',
        'high_to_low_qubit_order': 'HIGH_TO_LOW_QUBIT_ORDER',
        'use_gpu': 'USE_GPU',
        'qaoa_seed_count': 'QAOA_SEED_COUNT',
        'qaoa_execution_mode': 'QAOA_EXECUTION_MODE',
        'qaoa_max_memory_gb': 'QAOA_MAX_MEMORY_GB',
        'outputs_folder': 'OUTPUTS_FOLDER',
        'logs_folder': 'LOGS_FOLDER',
        'logs_file': 'LOGS_FILE',
    }
    
    for arg_name, value in kwargs.items():
        if value is not None:  # Only update if value was explicitly provided
            constant_name = arg_to_constant_map.get(arg_name)
            if constant_name:
                setattr(constants, constant_name, value)
            else:
                raise ValueError(f"Unknown constant argument: {arg_name}")

def _run_qaoa(cost_func, sample_func, qaoa_params, seed_versions, num_qubits, max_memory_gb, logger, previous_params):
    if constants.QAOA_EXECUTION_MODE == "sequential":
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




def layered_run(cost_func, sample_func, target_indices, valid_conformations, num_qubits, qaoa_layers, result_path, previous_params=None):
    # previous_params=None
    logger = logging.getLogger(f"qaoa.main.p_{qaoa_layers}")

    qaoa_params = QAOAParams(layers=qaoa_layers, optimiser_stepsize=constants.BASE_STEPSIZE, epochs=constants.BASE_EPOCHS) if previous_params is None else (
        QAOAParams(layers=qaoa_layers, optimiser_stepsize=constants.BASE_STEPSIZE/constants.STEPSIZE_REDUCTION_FACTOR, epochs=constants.BASE_EPOCHS))
    logger.info(f"Starting layered QAOA for {num_qubits} qubits with parameters: {qaoa_params}")

    max_memory_gb = constants.QAOA_MAX_MEMORY_GB
    seed_versions = list(range(constants.QAOA_SEED_COUNT))

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
             allow_pickle=False,

             target_probs=target_probs,
             optimized_params=optimized_params,
    )

    logger.debug(f"Success Metric (P_success) per seed: {success_metric}")

    return optimized_params

def inner_loop(file_path, logger, results_dir):

    artifact_base_name = file_path.split("/")[-1].split(".")[0]

    one_body, two_body = load_energy_mappings(file_path)

    basic_params: BasicParams = init_basic_params(one_body)

    coeffs, observables, num_qubits = extract_ising_items(one_body, two_body, logger)
    cost_hamiltonian = qml.dot(coeffs, observables)

    device_type = 'lightning.gpu' if constants.USE_GPU else 'lightning.qubit'
    dev = get_cached_device(num_qubits, device_type)
    logger.info(f"Running on {device_type} for {num_qubits} qubits")
    logger.info(
        f"JIT backend | mode={constants.QAOA_EXECUTION_MODE} | seeds={constants.QAOA_SEED_COUNT}"
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

def outer_loop():
    energy_files = find_limit_energy_files(
        qubit_counts=constants.QUBIT_COUNTS_TO_ANALYSE,
        source_folder=constants.SOURCE_FOLDER,
        limit_files_per_qubit=constants.LIMIT_FILES_PER_QUBIT,
        start_at_file_idx=constants.START_AT_FILE_IDX)
    total_processing_estimate = define_processing_estimate(energy_files)


    pathlib.Path(constants.OUTPUTS_FOLDER).mkdir(exist_ok=True, parents=True)
    logger = setup_logging(constants.LOGS_FOLDER, constants.LOGS_FILE)

    logger.info("Starting QAOA Runs for qubit counts: " + ", ".join(
        f"{num_qubits} ({len(files)} files)" for num_qubits, files in energy_files.items()
    ))

    ANALYSIS_ORDER = reversed(sorted(constants.QUBIT_COUNTS_TO_ANALYSE)) \
        if constants.HIGH_TO_LOW_QUBIT_ORDER else sorted(constants.QUBIT_COUNTS_TO_ANALYSE)

    current_processed = 0
    for qubit_count in ANALYSIS_ORDER:  # high to low
        cur_energy_files = energy_files[qubit_count]
        logger.info(f"Processing {len(cur_energy_files)} files for {qubit_count} qubits")
        for energy_file in cur_energy_files:
            logger.info(f"Starting QAOA runs for {energy_file.name}")
            start = time.perf_counter()
            inner_loop(energy_file.as_posix(), logger, constants.OUTPUTS_FOLDER)
            qaoa_time = time.perf_counter() - start

            current_processed += define_processing_estimate({qubit_count: [energy_file]})
            logger.info(f"Completed QAOA runs for {energy_file.name} in {qaoa_time:.2f} seconds - completed {current_processed/total_processing_estimate*100:.3f}% of estimated total processing time\n")

if __name__ == "__main__":
    outer_loop()