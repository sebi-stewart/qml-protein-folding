import pickle
from qaoa.objects import init_basic_params

ENERGIES_SMALL = "energies/small"
ENERGIES_LARGE = "energies/large"
ENERGIES_TOO_LARGE = "energies/too_large"

ENERGIES_ALT_FOLDER = "Phase2_energies"
ALT_ENERGIES_FOLDER_COLLECTION = [f"{ENERGIES_ALT_FOLDER}/{i}" for i in range(1, 23)]

def _choose_save_folder(one_body):
    basic_params = init_basic_params(one_body)
    if basic_params.num_qubits <= 15:
        return ENERGIES_SMALL, basic_params.num_qubits
    elif basic_params.num_qubits <= 22:
        return ENERGIES_LARGE, basic_params.num_qubits
    else:
        return ENERGIES_TOO_LARGE, basic_params.num_qubits

def save_results(one_body, two_body, logger, artifact_path):

    file_folder, qubit_count = _choose_save_folder(one_body)
    file_location = f"{file_folder}/{artifact_path}"
    with open(file_location, 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump({
            'one_body': one_body,
            'two_body': two_body
        }, f)
    logger.info(f"Saved extracted tensors to {file_location} - contains {qubit_count} qubits")
    return file_location

def _choose_save_folder_by_qubit(one_body):
    basic_params = init_basic_params(one_body)
    if basic_params.num_qubits > 22:
        return ENERGIES_TOO_LARGE, basic_params.num_qubits
    return f"{ENERGIES_ALT_FOLDER}/{basic_params.num_qubits}", basic_params.num_qubits

def save_results_alternate(one_body, two_body, logger, artifact_path):

    file_folder, qubit_count = _choose_save_folder_by_qubit(one_body)
    file_location = f"{file_folder}/{artifact_path}"
    with open(file_location, 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump({
            'one_body': one_body,
            'two_body': two_body
        }, f)
    logger.info(f"Saved extracted tensors to {file_location} - contains {qubit_count} qubits")
    return file_location