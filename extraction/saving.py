"""
Helpers to persist extracted energy tensors and manage the energy-mapping folder structure.
This module chooses appropriate subfolders by qubit count and writes JSON energy mapping artifacts.
"""

import json
import pathlib

import utils.make_paths_absolute # Inportant for file paths # noqa

from qaoa.objects import init_basic_params

# Set up the folder structure for saving energy mappings, set them up here to avoid type hints when setting to None
SETUP_RUN = False
ENERGIES_FOLDER = pathlib.Path("intermediates/energy_mappings")
ENERGIES_TOO_LARGE = ENERGIES_FOLDER.joinpath("too_large")

def setup_folders(energies_folder: str = "intermediates/energy_mappings"):
    global SETUP_RUN, ENERGIES_FOLDER, ENERGIES_TOO_LARGE

    ENERGIES_FOLDER = pathlib.Path(energies_folder)
    ENERGIES_TOO_LARGE = ENERGIES_FOLDER.joinpath("too_large")

    pathlib.Path(ENERGIES_TOO_LARGE).mkdir(exist_ok=True, parents=True)
    for i in range(1, 23):
        ENERGIES_FOLDER.joinpath(str(i)).mkdir(exist_ok=True, parents=True)

    SETUP_RUN = True


def _choose_save_folder_by_qubit(one_body) -> tuple[pathlib.Path, int]:
    assert SETUP_RUN, "Call setup_folders() before using this function"

    basic_params = init_basic_params(one_body)
    if basic_params.num_qubits > 22:
        return ENERGIES_TOO_LARGE, basic_params.num_qubits
    return ENERGIES_FOLDER.joinpath(str(basic_params.num_qubits)), basic_params.num_qubits

def _serialize(one_body, two_body):
    # one_body: dict[int, dict[int, float]]
    # two_body: dict[tuple[int,int], dict[tuple[int,int], float]]
    return {
        'one_body': {str(k): {str(rk): rv for rk, rv in v.items()} for k, v in one_body.items()},
        'two_body': {f"{i},{j}": {f"{ri},{rj}": e for (ri, rj), e in interactions.items()}
                     for (i, j), interactions in two_body.items()},
    }

def save_results_alternate(one_body, two_body, logger, artifact_path):
    artifact_path = pathlib.Path(artifact_path).with_suffix('.json')
    file_folder, qubit_count = _choose_save_folder_by_qubit(one_body)
    if qubit_count == 0:
        logger.warning(f"Skipping saving results for {artifact_path} - contains 0 qubits")
        return None
    output_file = file_folder.joinpath(artifact_path)
    with open(output_file, 'w') as f:
        json.dump(_serialize(one_body, two_body), f)
    logger.info(f"Saved extracted tensors to {output_file} - contains {qubit_count} qubits")
    return output_file