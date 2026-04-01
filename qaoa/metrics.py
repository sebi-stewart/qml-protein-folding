import jax
import jax.numpy as jnp


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

