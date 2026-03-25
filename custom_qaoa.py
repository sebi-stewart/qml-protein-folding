import pennylane as qml
import pennylane.numpy as np
from misc import QAOAParams, BasicParams
from constants import IS_LINUX


def qaoa_func_generator(H_ising, mixer_layer, generator_params: BasicParams):
    num_qubits = generator_params.num_qubits
    wire_offsets = generator_params.wire_offsets
    seq_positions = generator_params.seq_positions
    rotamer_counts = generator_params.rotamer_counts

    dev = qml.device('lightning.gpu' if IS_LINUX else 'lightning.qubit', wires=range(num_qubits))

    def qaoa_layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, H_ising)
        mixer_layer(beta, wire_offsets, seq_positions, rotamer_counts)


    @qml.qnode(dev, diff_method="adjoint")
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
