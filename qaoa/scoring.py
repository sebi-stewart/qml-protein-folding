import logging
from itertools import product

import numpy as np
from dataclasses import dataclass

from extraction.qubo_creation import build_dense_qubo
from qaoa.objects import BasicParams

import jax.numpy as jnp
import jax


@dataclass
class Conformation:
    idx: int
    bitstring: list[int]
    qubo_energy: np.float64

@jax.jit
def _evaluate_all_qubo_energies(X_matrix, h_dense, J_dense):
    """
    Executes on GPU/CPU via XLA.
    X_matrix: shape (M, N) - The valid bitstrings
    h_dense: shape (N,)
    J_dense: shape (N, N)
    """
    # 1-body energies: Matrix-vector multiplication (M, N) @ (N,) -> (M,)
    e_1body = jnp.dot(X_matrix, h_dense)

    # 2-body energies: sum(X * (X @ J)) row-wise -> (M,)
    e_2body = jnp.sum(X_matrix * jnp.dot(X_matrix, J_dense), axis=1)

    return e_1body + e_2body

def _get_valid_bitstrings_matrix(params: BasicParams, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    num_qubits = params.num_qubits
    wire_offsets = params.wire_offsets
    seq_positions = params.seq_positions
    rotamer_counts = params.rotamer_counts

    # Generate all combinations of rotamer choices for each residue
    rotamer_choices = [range(rotamer_counts[seq]) for seq in seq_positions]

    valid_bitstrings = []
    for rotamer_combo in product(*rotamer_choices):
        bitstring = [0] * num_qubits
        for seq_idx, seq in enumerate(seq_positions):
            bitstring[wire_offsets[seq] + rotamer_combo[seq_idx]] = 1
        valid_bitstrings.append(bitstring)

    X_matrix = np.array(valid_bitstrings, dtype=np.float64)

    # Vectorized conversion to base-10 indices
    powers_of_two = 1 << np.arange(num_qubits)[::-1]
    indices = X_matrix.dot(powers_of_two).astype(np.int32)

    logger.debug(f"Total Valid Conformations Matrix Shape: {X_matrix.shape}")
    return X_matrix, indices


def extract_lowest_energy_bitstrings(h_linear, J_quadratic, logger, epsilon, params: BasicParams):
    X_matrix, indices = _get_valid_bitstrings_matrix(params, logger)
    h_dense, J_dense = build_dense_qubo(h_linear, J_quadratic, params.num_qubits, params.wire_offsets)

    energies = _evaluate_all_qubo_energies(
        jnp.array(X_matrix),
        jnp.array(h_dense),
        jnp.array(J_dense)
    )

    min_energy = jnp.min(energies)
    valid_mask = jnp.abs(energies - min_energy) <= epsilon
    winning_row_indices = jnp.where(valid_mask)[0]

    target_indices = jnp.array(indices[winning_row_indices], dtype=np.int32)

    winning_conformations = [
        Conformation(
            idx=int(indices[row_idx]),
            bitstring=X_matrix[row_idx].astype(int).tolist(),
            qubo_energy=np.float64(energies[row_idx]),
        ) for row_idx in winning_row_indices
    ]

    return target_indices, winning_conformations