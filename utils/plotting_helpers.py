import pathlib
from utils import loading_energy_mappings

def file_to_qubit_mappings(energy_file_folders: str):
    """
    Generate a map from energy file paths to their corresponding qubit counts based on the folder structure.
    :param energy_file_folders:
    :return:
    """

    energy_files = list(pathlib.Path(energy_file_folders).rglob("*.json"))
    file_to_qubit_count_mapping = {}
    for file in energy_files:
        file_stem = str(file.stem)
        str_qubit_count = file.as_posix().split("/")[-2]
        try:
            qubit_count = int(str_qubit_count)
            file_to_qubit_count_mapping[file_stem] = qubit_count
        except ValueError:
            print(f"Warning: Could not convert '{str_qubit_count}' to an integer for file {file_stem}. Skipping this file.")
    return file_to_qubit_count_mapping