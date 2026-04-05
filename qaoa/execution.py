import logging
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

def _retrieve_batch_size(num_qubits: int, max_memory_gb: float, original_num_seeds: int):
    bytes_per_state = (2 ** num_qubits) * 16
    bytes_per_seed = 3 * bytes_per_state * 1.2
    b_max = math.floor((max_memory_gb * (1024 ** 3)) / bytes_per_seed)
    if b_max < 1:
        raise MemoryError(f"Insufficient memory ({max_memory_gb}GB) to run even a single seed for {num_qubits} qubits.")

    return min(b_max, original_num_seeds)

def batched_qaoa(cost_function, sample_function, qaoa_params, seed_versions, num_qubits: int, max_memory_gb: float, logger: logging.Logger, previous_params=None):
    """
    Executes all seeds simultaneously
    Strictly free of PyRosetta objects and file I/O.
    """
    original_num_seeds = len(seed_versions)
    batch_size = _retrieve_batch_size(num_qubits, max_memory_gb, original_num_seeds)
    num_batches = math.ceil(original_num_seeds / batch_size)
    padded_total_seeds = num_batches * batch_size

    logger.debug(f"[VRAM Profiler] {num_qubits} Qubits | Batch Size: {batch_size} | Executing {num_batches} Chunks (Total Padded: {padded_total_seeds})")

    master_key = jax.random.PRNGKey(seed_versions[0])
    keys = jax.random.split(master_key, padded_total_seeds)

    # --- WARM START LOGIC: Handle padding and noise globally before the batch loop ---
    # if previous_params is not None:
    #     prev_layers = previous_params.shape[2]
    #     new_layers = qaoa_params.layers - prev_layers
    #
    #     if new_layers > 0:
    #         # 1. Pad the existing parameters with zeros for the new layers
    #         # Shape transitions from (batch, 2, prev_layers) to (batch, 2, prev_layers + new_layers)
    #         padded_params = jnp.pad(previous_params, ((0, 0), (0, 0), (0, new_layers)))
    #
    #         # 2. Generate tiny noise for the new layers to break symmetry
    #         noise_key = jax.random.PRNGKey(seed_versions[0] + qaoa_params.layers)
    #         noise = jax.random.normal(noise_key, shape=(padded_total_seeds, 2, new_layers)) * 1e-4
    #
    #         # 3. Add noise ONLY to the newly added layers
    #         padded_params = padded_params.at[:, :, prev_layers:].add(noise)
    #     else:
    #         padded_params = previous_params

    # 2. Initialize a batched parameter tensor of shape (30, 2, layers)
    def init_params(key):
        return jax.random.uniform(key, shape=(2, qaoa_params.layers), minval=-jnp.pi, maxval=jnp.pi)

    optimizer = optax.adam(learning_rate=qaoa_params.optimiser_stepsize)

    def single_update_step(params, opt_state_inner):
        cost_val, grads = jax.value_and_grad(cost_function)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state_inner, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, cost_val


    # 5. JIT-Compile the VMAP'd update step
    # in_axes=(0, 0) means "vectorize over the first dimension of both inputs"
    @jax.jit
    def batched_update_step(params_batch, opt_state_batch):
        return jax.vmap(single_update_step, in_axes=(0, 0))(params_batch, opt_state_batch)

    @jax.jit
    def batched_sample(params_batch):
        return jax.vmap(sample_function)(params_batch)


    all_final_probs = []
    all_cost_histories = []
    all_optimized_params = []  # Store the updated parameters to return

    for b in range(num_batches):
        logger.debug(f"[BATCH Tracker] Batch {b} / {num_batches}")
        # Extract the PRNG keys for this specific chunk
        chunk_keys = keys[b * batch_size: (b + 1) * batch_size]

        # Initialize parameters and optimizer state for the chunk
        # --- INITIALIZATION DECISION: Cold Start vs Warm Start ---
        # if previous_params is None:
        current_params = jax.vmap(init_params)(chunk_keys)
        # else:
        #     # Slice the pre-padded warm-started parameters for this specific batch chunk
        #     current_params = padded_params[b * batch_size: (b + 1) * batch_size]

        current_opt_state = jax.vmap(optimizer.init)(current_params)

        batch_cost_history = []

        # Epoch Loop for this chunk
        for epoch in range(qaoa_params.epochs):
            current_params, current_opt_state, batched_costs = batched_update_step(current_params, current_opt_state)
            batch_cost_history.append(batched_costs)
            if epoch % 10 == 0:
                logger.debug(f"\t[EPOCH Tracker] Epoch  {epoch} | Cost: {np.mean(batched_costs):.4f}")

        # Convert batch cost history to array: (batch_size, epochs)
        batch_cost_history = jnp.stack(batch_cost_history, axis=1)
        all_cost_histories.append(batch_cost_history)

        # Sample the final probabilities
        chunk_probs = batched_sample(current_params)
        all_final_probs.append(chunk_probs)

        # Save optimized parameters for this chunk
        all_optimized_params.append(current_params)

    # 5. Concatenate and Discard Padded Dummies
    # jnp.vstack merges the chunks: (num_batches, batch_size, 2^N) -> (padded_total_seeds, 2^N)
    combined_probs = jnp.vstack(all_final_probs)
    combined_costs = jnp.vstack(all_cost_histories)
    combined_params = jnp.vstack(all_optimized_params)

    return combined_probs[:original_num_seeds], combined_costs[:original_num_seeds], combined_params[:original_num_seeds]
