import logging
import math

import jax
import jax.numpy as jnp
import optax

EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 1e-4


def _param_dtype():
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

def _retrieve_batch_size(num_qubits: int, max_memory_gb: float, original_num_seeds: int):
    bytes_per_state = (2 ** num_qubits) * 16
    bytes_per_seed = 3 * bytes_per_state * 1.2
    b_max = math.floor((max_memory_gb * (1024 ** 3)) / bytes_per_seed)
    if b_max < 1:
        raise MemoryError(f"Insufficient memory ({max_memory_gb}GB) to run even a single seed for {num_qubits} qubits.")

    compiler_cap = 6

    return min(compiler_cap, b_max, original_num_seeds)

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
    padded_params = None

    # --- WARM START LOGIC: Handle padding and noise globally before the batch loop ---
    if previous_params is not None:
        prev_layers = previous_params.shape[2]
        new_layers = qaoa_params.layers - prev_layers

        if new_layers > 0:
            # 1. Pad the existing parameters with zeros for the new layers
            # Shape transitions from (batch, 2, prev_layers) to (batch, 2, prev_layers + new_layers)
            padded_params = jnp.pad(previous_params, ((0, 0), (0, 0), (0, new_layers)))

            # 2. Generate tiny noise for the new layers to break symmetry
            noise_key = jax.random.PRNGKey(seed_versions[0] + qaoa_params.layers)
            noise = jax.random.normal(noise_key, shape=(padded_total_seeds, 2, new_layers)) * 1e-4

            # 3. Add noise ONLY to the newly added layers
            padded_params = padded_params.at[:, :, prev_layers:].add(noise)
        else:
            padded_params = previous_params

    # 2. Initialize a batched parameter tensor of shape (30, 2, layers)
    dtype = _param_dtype()

    def init_params(key):
        return jax.random.uniform(
            key,
            shape=(2, qaoa_params.layers),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=dtype,
        )

    optimizer = optax.adam(learning_rate=qaoa_params.optimiser_stepsize)

    def single_update_step(params, opt_state_inner, is_converged):
        cost_val, grads = jax.value_and_grad(cost_function)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state_inner, params)
        new_params = optax.apply_updates(params, updates)

        # Conditionally freeze parameters and optimizer state if converged
        final_params = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_converged, old, new),
            params,
            new_params
        )

        return final_params, new_opt_state, cost_val


    # VMAP a single seed update over the seed dimension.
    @jax.jit
    def batched_update_step(params_batch, opt_state_batch, converged_mask):
        return jax.vmap(single_update_step, in_axes=(0, 0, 0))(
            params_batch,
            opt_state_batch,
            converged_mask
        )

    @jax.jit
    def batched_sample(params_batch):
        return jax.vmap(sample_function)(params_batch)


    all_final_probs = []
    all_cost_histories = []
    all_optimized_params = []  # Store the updated parameters to return

    for b in range(num_batches):
        logger.debug(f"[BATCH Tracker] Batch {b+1} / {num_batches}")
        # Extract the PRNG keys for this specific chunk
        chunk_keys = keys[b * batch_size: (b + 1) * batch_size]

        # Initialize parameters and optimizer state for the chunk
        # --- INITIALIZATION DECISION: Cold Start vs Warm Start ---
        if previous_params is None:
            current_params = jax.vmap(init_params)(chunk_keys)
        else:
            # Slice the pre-padded warm-started parameters for this specific batch chunk
            current_params = padded_params[b * batch_size: (b + 1) * batch_size]

        current_opt_state = jax.vmap(optimizer.init)(current_params)

        batch_cost_history = []
        best_costs = jnp.full(batch_size, jnp.inf)
        patience_counters = jnp.zeros(batch_size, dtype=jnp.int32)
        converged_mask = jnp.zeros(batch_size, dtype=jnp.bool_)

        # Epoch Loop for this chunk
        for epoch in range(qaoa_params.epochs):
            current_params, current_opt_state, batched_costs = batched_update_step(
                current_params,
                current_opt_state,
                converged_mask
            )
            batch_cost_history.append(batched_costs)
            if epoch % 10 == 0:
                logger.debug(
                    f"\t[EPOCH Tracker] Epoch  {epoch} | Cost: {jnp.mean(batched_costs):.4f} | Stopped Seeds: {jnp.sum(converged_mask)} / {batch_size}")

            improved = (best_costs - batched_costs) > EARLY_STOPPING_DELTA

            best_costs = jnp.where(improved, batched_costs, best_costs)
            patience_counters = jnp.where(improved, 0, patience_counters + 1)
            converged_mask = patience_counters >= EARLY_STOPPING_PATIENCE
            if jnp.all(converged_mask):
                logger.debug(
                    f"\t[EPOCH Tracker] Early stopping triggered at epoch {epoch} for all seeds in this batch.")
                break

        # Convert batch cost history to array: (batch_size, epochs)
        logger.info(f"\t[EPOCH Tracker] Completed Epochs for Batch {b + 1} - Final Cost: {jnp.mean(batched_costs):.4f}")
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


def sequential_qaoa(cost_function, sample_function, qaoa_params, seed_versions, num_qubits: int, max_memory_gb: float, logger: logging.Logger, previous_params=None):
    """
    Executes seeds sequentially while keeping the same output format as batched_qaoa.
    Uses explicit epoch loops per seed (first sequential implementation style).
    """
    del num_qubits, max_memory_gb  # Unused in sequential mode, kept for API compatibility.

    original_num_seeds = len(seed_versions)
    logger.debug(f"[Sequential Tracker] Executing {original_num_seeds} Seeds Sequentially")
    padded_params = None
    dtype = _param_dtype()

    def init_params(key):
        return jax.random.uniform(
            key,
            shape=(2, qaoa_params.layers),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=dtype,
        )

    optimizer = optax.adam(learning_rate=qaoa_params.optimiser_stepsize)

    @jax.jit
    def update_step(params, opt_state_inner):
        cost_val, grads = jax.value_and_grad(cost_function)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state_inner, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, cost_val

    @jax.jit
    def sample_step(params):
        return sample_function(params)

    all_final_probs = []
    all_cost_histories = []
    all_optimized_params = []

    if previous_params is not None:
        prev_layers = previous_params.shape[2]
        new_layers = qaoa_params.layers - prev_layers

        if new_layers > 0:
            padded_params = jnp.pad(previous_params, ((0, 0), (0, 0), (0, new_layers)))
            noise_key = jax.random.PRNGKey(seed_versions[0] + qaoa_params.layers)
            noise = jax.random.normal(noise_key, shape=(original_num_seeds, 2, new_layers)) * 1e-4
            padded_params = padded_params.at[:, :, prev_layers:].add(noise)
        else:
            padded_params = previous_params

    for i, seed in enumerate(seed_versions):
        logger.debug(f"[Sequential Tracker] Seed {i + 1} / {original_num_seeds}")

        if previous_params is None:
            seed_key = jax.random.PRNGKey(seed)
            current_params = init_params(seed_key)
        else:
            current_params = padded_params[i]

        current_opt_state = optimizer.init(current_params)
        best_cost = jnp.array(jnp.inf, dtype=dtype)
        patience_counter = 0
        converged = False

        seed_costs = []
        for _ in range(qaoa_params.epochs):
            if not converged:
                current_params, current_opt_state, cost_val = update_step(current_params, current_opt_state)
                improved = float(best_cost - cost_val) > EARLY_STOPPING_DELTA
                best_cost = cost_val if improved else best_cost
                patience_counter = 0 if improved else patience_counter + 1
                converged = patience_counter >= EARLY_STOPPING_PATIENCE
                seed_costs.append(cost_val)
            else:
                seed_costs.append(best_cost)

        cost_history = jnp.asarray(seed_costs)
        final_probs = sample_step(current_params)
        logger.info(f"\t[Sequential Tracker] Completed Seed {i + 1} - Final Cost: {cost_history[-1]:.4f}")

        all_final_probs.append(final_probs)
        all_cost_histories.append(cost_history)
        all_optimized_params.append(current_params)

    combined_probs = jnp.stack(all_final_probs, axis=0)
    combined_costs = jnp.stack(all_cost_histories, axis=0)
    combined_params = jnp.stack(all_optimized_params, axis=0)

    return combined_probs, combined_costs, combined_params

