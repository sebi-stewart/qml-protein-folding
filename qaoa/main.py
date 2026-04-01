import logging

from custom_qaoa import get_cached_device
from h_mixer import ring_xy_mixer_layer
from misc import init_basic_params, BasicParams
from qaoa.execution import batched_qaoa
from qaoa.generators import qaoa_func_generator
from qaoa.hamiltonians import extract_ising_items
from constants import IS_LINUX

import pennylane as qml

from qaoa.metrics import calculate_epsilon_success
from qaoa.objects import QAOAParams
from qaoa.scoring import extract_lowest_energy_bitstrings

def main(h_linear, J_quadratic, qaoa_layers):
    basic_params: BasicParams = init_basic_params(h_linear)

    logger = logging.getLogger("qaoa.main")
    coeffs, observables, num_qubits = extract_ising_items(h_linear, J_quadratic, logger)
    cost_hamiltonian = qml.dot(coeffs, observables)

    device_type = 'lightning.gpu' if IS_LINUX else 'lightning.qubit'
    dev = get_cached_device(num_qubits, device_type)
    logger.debug(f"Running on {device_type} for {num_qubits} qubits")

    qaoa_params = QAOAParams(layers=qaoa_layers, optimiser_stepsize=0.01, epochs=150)
    max_memory_gb = 10 # ADJUST THIS BASED ON YOUR GPU CAPACITY

    cost_func, sample_func = qaoa_func_generator(dev, cost_hamiltonian, ring_xy_mixer_layer, basic_params)
    batched_probs = batched_qaoa(cost_func, sample_func, qaoa_params, num_qubits, max_memory_gb, logger)

    target_indices, valid_conformations = extract_lowest_energy_bitstrings(
        h_linear, J_quadratic,
        logger, 1e-6, basic_params
    )

    success_metric = calculate_epsilon_success(batched_probs, target_indices, logger)





