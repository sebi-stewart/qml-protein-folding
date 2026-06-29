"""
Helpers to configure a PennyLane QAOA sampler and collect shot-based samples for phase 2 rescoring.
Provides utilities to build a device, generate sampling functions, and deduplicate sampled bitstrings.
"""

import logging
import pennylane as qml

from qaoa.devices import get_cached_device
from qaoa.generators import qaoa_func_generator
from qaoa.h_mixer import ring_xy_mixer_layer
from qaoa.hamiltonians import extract_ising_items
from qaoa.objects import init_basic_params, BasicParams

qubit_to_shot_map = {
    5: 10, 6: 10, 7: 100, 8: 10, 9: 10, 10: 100, 11: 100, 12: 100, 13: 100, 14: 100, 18: 100, 22: 400
}

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

def take_qaoa_shots(one_body, two_body, best_params, phase2_rescoring_logger: logging.Logger, hidden_logger: logging.Logger):
    sample_func, basic_params = get_sample_function_for_phase_2(hidden_logger, one_body, two_body)
    processed_results, unique_bitstrings = get_and_process_shot_results(sample_func, best_params, phase2_rescoring_logger)
    return processed_results, unique_bitstrings, basic_params
