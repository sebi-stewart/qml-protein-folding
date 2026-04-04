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