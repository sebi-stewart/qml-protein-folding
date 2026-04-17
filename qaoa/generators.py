import numpy as np
import pennylane as qml
from qaoa.objects import BasicParams

def qaoa_func_generator(dev, H_ising, mixer_layer, generator_params: BasicParams):
    num_qubits = generator_params.num_qubits
    wire_offsets = generator_params.wire_offsets
    seq_positions = generator_params.seq_positions
    rotamer_counts = generator_params.rotamer_counts

    def qaoa_layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, H_ising)
        mixer_layer(beta, wire_offsets, seq_positions, rotamer_counts)

    def qaoa_init_layer():
        for seq in seq_positions:
            k = rotamer_counts[seq]

            state_vector = np.zeros(2**k)
            for i in range(k):
                one_hot_int = 2 ** (k - 1 - i)
                state_vector[one_hot_int] = 1.0 / np.sqrt(k)

            base_wire = wire_offsets[seq]
            bundle_wires = list(range(base_wire, base_wire + k))

            qml.StatePrep(state_vector, wires=bundle_wires)

    @qml.qnode(dev, interface="jax", diff_method="adjoint")
    def cost_function(params):
        gammas = params[0]
        betas = params[1]

        qaoa_init_layer()
        for i in range(len(gammas)):
            qaoa_layer(gammas[i], betas[i])

        # 3. Measure the expectation value of the cost Hamiltonian
        return qml.expval(H_ising)

    @qml.qnode(dev, interface="jax", diff_method=None)
    def sample_function(params):
        gammas = params[0]
        betas = params[1]

        qaoa_init_layer()
        for i in range(len(gammas)):
            qaoa_layer(gammas[i], betas[i])

        return qml.probs(wires=range(num_qubits))

    return cost_function, sample_function

def qaoa_final_sample_func_generator(dev, H_ising, mixer_layer, generator_params: BasicParams):
    num_qubits = generator_params.num_qubits
    wire_offsets = generator_params.wire_offsets
    seq_positions = generator_params.seq_positions
    rotamer_counts = generator_params.rotamer_counts

    def qaoa_layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, H_ising)
        mixer_layer(beta, wire_offsets, seq_positions, rotamer_counts)

    def qaoa_init_layer():
        for seq in seq_positions:
            k = rotamer_counts[seq]

            state_vector = np.zeros(2**k)
            for i in range(k):
                one_hot_int = 2 ** (k - 1 - i)
                state_vector[one_hot_int] = 1.0 / np.sqrt(k)

            base_wire = wire_offsets[seq]
            bundle_wires = list(range(base_wire, base_wire + k))

            qml.StatePrep(state_vector, wires=bundle_wires)

    @qml.qnode(dev, interface="jax", diff_method=None)
    def sample_function(params):
        gammas = params[0]
        betas = params[1]

        qaoa_init_layer()
        for i in range(len(gammas)):
            qaoa_layer(gammas[i], betas[i])

        return qml.sample(wires=range(num_qubits))

    return sample_function

