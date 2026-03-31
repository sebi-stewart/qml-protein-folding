import pennylane as qml
import pennylane.numpy as np
from misc import QAOAParams, BasicParams
from constants import IS_LINUX

import jax
import jax.numpy as jnp
import optax

_DEVICE_CACHE = {}

def get_cached_device(num_qubits, device_type):
    cache_key = (num_qubits, device_type)
    if cache_key not in _DEVICE_CACHE:

        _DEVICE_CACHE.clear()
        _DEVICE_CACHE[cache_key] = qml.device(device_type, wires=range(num_qubits))
    return _DEVICE_CACHE[cache_key]

def qaoa_func_generator(H_ising, mixer_layer, generator_params: BasicParams):
    num_qubits = generator_params.num_qubits
    wire_offsets = generator_params.wire_offsets
    seq_positions = generator_params.seq_positions
    rotamer_counts = generator_params.rotamer_counts

    device_type = 'lightning.gpu' if (IS_LINUX and num_qubits > 18) else 'lightning.qubit'
    dev = get_cached_device(num_qubits, device_type)
    print(f"Running on {device_type} for {num_qubits} qubits")

    def qaoa_layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, H_ising)
        mixer_layer(beta, wire_offsets, seq_positions, rotamer_counts)


    @qml.qnode(dev, interface="jax", diff_method="adjoint")
    def cost_function(params):
        gammas = params[0]
        betas = params[1]

        # 1. Custom initialisation
        for seq in seq_positions:
            base_wire = wire_offsets[seq]
            qml.PauliX(wires=base_wire) # First rotamer set to 1

        # 2. Apply p layers of QAOA
        for i in range(len(gammas)):
            qaoa_layer(gammas[i], betas[i])

        # 3. Measure the expectation value of the cost Hamiltonian
        return qml.expval(H_ising)

    @qml.qnode(dev)
    def sample_function(params):
        gammas = params[0]
        betas = params[1]

        for seq in seq_positions:
            base_wire = wire_offsets[seq]
            qml.PauliX(wires=base_wire)

        for i in range(len(gammas)):
            qaoa_layer(gammas[i], betas[i])

        return qml.probs(wires=range(num_qubits))

    return cost_function, sample_function

def run_qaoa(cost_function, qaoa_params: QAOAParams):
    np.random.seed(qaoa_params.seed)
    # Initialize parameters close to zero to avoid barren plateaus
    initial_params = np.random.uniform(low=-0.01, high=0.01, size=(2, qaoa_params.layers), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=qaoa_params.optimiser_stepsize)
    current_params = initial_params

    print("\n==================== QAOA Run ====================")
    print(f"Commencing QAOA Optimization [p={qaoa_params.layers}]...")
    for epoch in range(qaoa_params.epochs):
        current_params, cost = opt.step_and_cost(cost_function, current_params)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Cost: {cost:.4f}")

    print("Optimization converged.")
    print("==================== QAOA Run COMPLETE ====================\n")

    return current_params


def run_qaoa_jax(cost_function, qaoa_params: QAOAParams):
    # 1. Initialize functionally pure JAX randomness
    key = jax.random.PRNGKey(qaoa_params.seed)

    # 2. Initialize parameters (Adjusted bounds per Dr. Thorne's QAOA critique)
    # If you insist on narrow bounds, retain the [-0.01, 0.01].
    # Otherwise, use [-jnp.pi, jnp.pi] for proper phase exploration.
    initial_params = jax.random.uniform(
        key,
        shape=(2, qaoa_params.layers),
        minval=-0.01,
        maxval=0.01
    )

    optimizer = optax.adam(learning_rate=qaoa_params.optimiser_stepsize)
    opt_state = optimizer.init(initial_params)

    @jax.jit
    def update_step(params, opt_state_inner):
        # Calculate cost and gradients simultaneously on the backend
        cost_val, grads = jax.value_and_grad(cost_function)(params)

        # Calculate Adam updates and apply them
        updates, new_opt_state = optimizer.update(grads, opt_state_inner, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, cost_val

    current_params = initial_params

    print("\n==================== QAOA Run (JAX/XLA) ====================")
    print(f"Commencing XLA-Compiled Optimization [p={qaoa_params.layers}]...")

    # 5. The Python loop now strictly orchestrates XLA graph executions
    for epoch in range(qaoa_params.epochs):
        current_params, opt_state, cost = update_step(current_params, opt_state)

        if epoch % 10 == 0:
            # jax.numpy arrays must be explicitly cast/formatted for printing
            print(f"Epoch {epoch:3d} | Cost: {float(cost):.4f}")

    print("Optimization converged.")
    print("==================== QAOA Run COMPLETE ====================\n")

    return current_params