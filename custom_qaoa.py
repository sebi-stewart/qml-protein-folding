import pennylane as qml

def qaoa_func_generator(H_ising, mixer_layer, generator_params):
    num_qubits = generator_params["num_qubits"]
    wire_offsets = generator_params['wire_offsets']
    seq_positions = generator_params['seq_positions']
    rotamer_counts = generator_params['rotamer_counts']
    use_gpu = generator_params["use_gpu"]

    dev = qml.device('lightning.gpu' if use_gpu else 'lightning.qubit', wires=range(num_qubits))

    def qaoa_layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, H_ising)
        mixer_layer(beta, wire_offsets, seq_positions, rotamer_counts)


    @qml.qnode(dev)
    def cost_function(params):
        # params is a 2D array of shape (2, p) where p is the number of QAOA layers
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
        for seq in seq_positions:
            base_wire = wire_offsets[seq]
            qml.PauliX(wires=base_wire)

        for i in range(len(params[0])):
            qaoa_layer(params[0][i], params[1][i])

        return qml.probs(wires=range(num_qubits))

    return cost_function, sample_function
