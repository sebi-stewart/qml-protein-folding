import argparse
import os, sys
sys.path.append(os.getcwd())  # Ensures import work fine when running from the root directory of the project
import utils.make_paths_absolute # Important for file paths # noqa

from run import constants, outer_loop, update_constants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAOA runner for protein folding optimization")
    
    # Input File Vars
    parser.add_argument('--source-folder', type=str, default=constants.SOURCE_FOLDER, help=f'Folder containing energy mapping files (default: {constants.SOURCE_FOLDER})')
    parser.add_argument('--start-at-file-idx', type=int, default=constants.START_AT_FILE_IDX, help=f'Starting file index for multi-instance processing (default: {constants.START_AT_FILE_IDX})')
    parser.add_argument('--limit-files-per-qubit', type=int, default=constants.LIMIT_FILES_PER_QUBIT, help=f'Limit files per qubit count (default: {constants.LIMIT_FILES_PER_QUBIT})')
    parser.add_argument('--qubit-counts-to-analyse', type=int, nargs='+', default=constants.QUBIT_COUNTS_TO_ANALYSE, help=f'Qubit counts to analyse (default: {constants.QUBIT_COUNTS_TO_ANALYSE})')
    
    # QAOA Vars
    parser.add_argument('--base-epochs', type=int, default=constants.BASE_EPOCHS, help=f'Base number of epochs for optimization (default: {constants.BASE_EPOCHS})')
    parser.add_argument('--base-stepsize', type=float, default=constants.BASE_STEPSIZE, help=f'Base step size for optimizer (default: {constants.BASE_STEPSIZE})')
    parser.add_argument('--qaoa-layers', type=int, nargs='+', default=constants.QAOA_LAYERS, help=f'QAOA layers to test (default: {constants.QAOA_LAYERS})')
    parser.add_argument('--stepsize-reduction-factor', type=int, default=constants.STEPSIZE_REDUCTION_FACTOR, help=f'Factor to reduce step size by for subsequent layers (default: {constants.STEPSIZE_REDUCTION_FACTOR})')
    parser.add_argument('--high-to-low-qubit-order', type=bool, default=constants.HIGH_TO_LOW_QUBIT_ORDER, help=f'Process qubits from high to low if True, else low to high (default: {constants.HIGH_TO_LOW_QUBIT_ORDER})')
    parser.add_argument('--use-gpu', type=bool, default=constants.USE_GPU, help=f'Use GPU for computation (default: {constants.USE_GPU})')
    parser.add_argument('--qaoa-seed-count', type=int, default=constants.QAOA_SEED_COUNT, help=f'Number of random seeds for QAOA (default: {constants.QAOA_SEED_COUNT})')
    parser.add_argument('--qaoa-execution-mode', type=str, choices=['batched', 'sequential'], default=constants.QAOA_EXECUTION_MODE, help=f'QAOA execution mode (default: {constants.QAOA_EXECUTION_MODE})')
    parser.add_argument('--qaoa-max-memory-gb', type=float, default=constants.QAOA_MAX_MEMORY_GB, help=f'Maximum memory in GB for QAOA execution (default: {constants.QAOA_MAX_MEMORY_GB})')
    
    # Output Vars
    parser.add_argument('--outputs-folder', type=str, default=constants.OUTPUTS_FOLDER, help=f'Output folder for results (default: {constants.OUTPUTS_FOLDER})')
    parser.add_argument('--logs-folder', type=str, default=constants.LOGS_FOLDER, help=f'Output folder for logs (default: {constants.LOGS_FOLDER})')
    parser.add_argument('--logs-file', type=str, default=constants.LOGS_FILE, help=f'Name for log file (default: {constants.LOGS_FILE})')
    
    args = parser.parse_args()
    update_constants(**vars(args))
    outer_loop()
