import os, sys
sys.path.append(os.getcwd())  # Ensures import work fine when running from the root directory of the project
import utils.make_paths_absolute # Important for file paths # noqa

import pathlib

def find_limit_energy_files(qubit_counts: list[int], source_folder="intermediates/energy_mappings", limit_files_per_qubit=10, start_at_file_idx=0):
    all_energy_files = {num_qubits: list(pathlib.Path(f"{source_folder}/{num_qubits}").glob("*.json")) for num_qubits in qubit_counts}
    print(f"Found energy files for qubit counts: ", all_energy_files)

    # Remove duplicate files across qubit counts, ie. if the file analyses the same residue subsection with the same qubit count, but different rotamer counts it should only be processed once.
    # We keep the first one that appears, since they will have the same one-body and two-body energies, and thus the same QAOA performance.
    for num_qubits, files in all_energy_files.items():
        seen_files = set()

        unique_files = []
        for file in files:
            file_name_wo_extension = str(file).split("/")[-1].split(".")[0]
            file_without_rotamer_count = "_".join(file_name_wo_extension.split("_")[:-1]) # removes the rotamer count from the file name, which is the last element after splitting by "_"

            if file_without_rotamer_count not in seen_files:
                unique_files.append(file)
                seen_files.add(file_without_rotamer_count)
        all_energy_files[num_qubits] = unique_files

    print(f"After removing duplicates, found energy files for qubit counts: ", all_energy_files)
    energy_files = {num_qubits: [] for num_qubits in qubit_counts}
    for num_qubits, files in all_energy_files.items():
        if start_at_file_idx > len(files): continue

        if start_at_file_idx + limit_files_per_qubit > len(files): energy_files[num_qubits] = files[start_at_file_idx:]
        else: energy_files[num_qubits] = files[start_at_file_idx:start_at_file_idx+limit_files_per_qubit]

    return energy_files


def define_processing_estimate(energy_files: dict[int, list[pathlib.Path]]) -> float:
    MULTIPLICATIVE_FACTOR = 20
    EXPONENTIAL_FACTOR = 1.41
    ADDITIVE_FACTOR = 50

    return sum(len(files) * (ADDITIVE_FACTOR + MULTIPLICATIVE_FACTOR * (EXPONENTIAL_FACTOR ** num_qubits)) for
               num_qubits, files in energy_files.items())
