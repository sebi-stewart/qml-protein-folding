import jax
import jax.numpy as jnp
import numpy as np

from qaoa.scoring import Conformation


@jax.jit
def calculate_epsilon_success(batched_probs, target_indices):
    """
    Executes on the A100.
    batched_probs: shape (30, 2^N) - The full QAOA probability distribution
    target_indices: shape (K,) - The integer indices of the biologically valid states
    """
    # Extract only the columns corresponding to the valid biological states
    # Resulting shape: (30, K)
    valid_state_probs = batched_probs[:, target_indices]

    # Sum the probabilities across the K valid states for each of the 30 seeds
    # Resulting shape: (30,)
    p_success_batched = jnp.sum(valid_state_probs, axis=1)

    return p_success_batched

def calculate_epsilon_success_no_jit(batched_probs, target_indices):
    """
    Executes on the A100.
    batched_probs: shape (30, 2^N) - The full QAOA probability distribution
    target_indices: shape (K,) - The integer indices of the biologically valid states
    """
    # Extract only the columns corresponding to the valid biological states
    # Resulting shape: (30, K)
    valid_state_probs = batched_probs[:, target_indices]

    # Sum the probabilities across the K valid states for each of the 30 seeds
    # Resulting shape: (30,)
    p_success_batched = jnp.sum(valid_state_probs, axis=1)

    return p_success_batched


def extract_metrics_for_serialization(final_probs, target_indices, valid_conformations: list[Conformation]):
    # Ensure final_probs is 2D (num_seeds, 2**N) for consistent handling
    if final_probs.ndim == 1:
        final_probs = final_probs[np.newaxis, :]

    num_seeds = final_probs.shape[0]

    # 1. Extract only the probabilities of the valid conformations
    # Resulting shape: (num_seeds, len(target_indices))
    target_probs = final_probs[:, target_indices]

    # 2. Determine the probability of each conformation per seed
    # Create a mapping payload rather than mutating the original Bio-object
    conformation_prob_map = []
    for i in range(len(target_indices)):
        conf_map = {
            "probabilities": [float(target_probs[seed_idx, i]) for seed_idx in range(num_seeds)],
            "conformation_data": valid_conformations[i]  # Reference static data
        }
        conformation_prob_map.append(conf_map)

    # 3. Find the best target index exactly (highest probability)
    # Average across all seeds to find the globally favored target state
    mean_target_probs = np.mean(target_probs, axis=0)
    best_target_idx_relative = np.argmax(mean_target_probs)
    best_target_index_absolute = target_indices[best_target_idx_relative]

    return target_probs, conformation_prob_map, best_target_index_absolute