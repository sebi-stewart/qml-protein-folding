import logging

import jax
import jax.numpy as jnp
import optax

EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 1e-4

def batched_qaoa(cost_function, sample_function, qaoa_params, seed_versions, num_qubits: int, max_memory_gb: float, logger: logging.Logger, previous_params=None):
    """
    Executes seeds sequentially (no batching).
    Strictly free of PyRosetta objects and file I/O.
    Returns results in the same format as the batched version.
    """
    original_num_seeds = len(seed_versions)
    logger.debug(f"[Sequential Execution] Running {original_num_seeds} seeds sequentially")

    master_key = jax.random.PRNGKey(seed_versions[0])
    keys = jax.random.split(master_key, original_num_seeds)

    def init_params(key):
        return jax.random.uniform(key, shape=(2, qaoa_params.layers), minval=-jnp.pi, maxval=jnp.pi)

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

    all_final_probs = []
    all_cost_histories = []
    all_optimized_params = []
    max_epochs_seen = 0

    # Sequential loop through seeds
    for seed_idx in range(original_num_seeds):
        logger.debug(f"[Seed Tracker] Seed {seed_idx+1} / {original_num_seeds}")

        # --- INITIALIZATION DECISION: Cold Start vs Warm Start ---
        if previous_params is None:
            current_params = init_params(keys[seed_idx])
        else:
            # Warm start: use previous parameters with optional padding and noise
            if seed_idx < previous_params.shape[0]:
                prev_layers = previous_params.shape[2]
                new_layers = qaoa_params.layers - prev_layers

                if new_layers > 0:
                    # Pad the existing parameters with zeros for the new layers
                    current_params = jnp.pad(previous_params[seed_idx], ((0, 0), (0, new_layers)))

                    # Generate tiny noise for the new layers to break symmetry
                    noise_key = jax.random.PRNGKey(seed_versions[seed_idx] + qaoa_params.layers)
                    noise = jax.random.normal(noise_key, shape=(2, new_layers)) * 1e-4

                    # Add noise ONLY to the newly added layers
                    current_params = current_params.at[:, prev_layers:].add(noise)
                else:
                    current_params = previous_params[seed_idx]
            else:
                current_params = init_params(keys[seed_idx])

        current_opt_state = optimizer.init(current_params)

        cost_history = []
        best_cost = jnp.inf
        patience_counter = 0
        is_converged = False
        cost_val = jnp.inf  # Initialize for tracking final cost

        # Epoch loop for this seed
        for epoch in range(qaoa_params.epochs):
            current_params, current_opt_state, cost_val = single_update_step(
                current_params,
                current_opt_state,
                is_converged
            )
            cost_history.append(float(cost_val))

            if epoch % 10 == 0:
                logger.debug(f"\t[Epoch Tracker] Epoch {epoch} | Cost: {float(cost_val):.4f} | Converged: {is_converged}")

            # Check for improvement
            if (best_cost - float(cost_val)) > EARLY_STOPPING_DELTA:
                best_cost = float(cost_val)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.debug(f"\t[Epoch Tracker] Early stopping triggered at epoch {epoch}")
                is_converged = True
                break

        logger.info(f"\t[Epoch Tracker] Completed Epochs for Seed {seed_idx+1} - Final Cost: {float(cost_val):.4f}")
        
        # Track maximum epochs for padding
        max_epochs_seen = max(max_epochs_seen, len(cost_history))
        cost_history = jnp.array(cost_history)
        all_cost_histories.append(cost_history)

        # Sample the final probabilities
        probs = sample_function(current_params)
        all_final_probs.append(probs)

        # Save optimized parameters for this seed
        all_optimized_params.append(current_params)

    # Pad all cost histories to the same length (max_epochs_seen) for consistent array shape
    padded_cost_histories = []
    for cost_hist in all_cost_histories:
        if len(cost_hist) < max_epochs_seen:
            # Pad with the last value to maintain shape consistency
            padded = jnp.pad(cost_hist, (0, max_epochs_seen - len(cost_hist)), mode='edge')
        else:
            padded = cost_hist
        padded_cost_histories.append(padded)

    # Stack results to match original output format
    combined_probs = jnp.stack(all_final_probs, axis=0)  # (num_seeds, 2^N)
    combined_costs = jnp.stack(padded_cost_histories, axis=0)  # (num_seeds, max_epochs)
    combined_params = jnp.stack(all_optimized_params, axis=0)  # (num_seeds, 2, layers)

    return combined_probs, combined_costs, combined_params
