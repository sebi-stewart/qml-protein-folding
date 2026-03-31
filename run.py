from dataclasses import dataclass
import contextlib
import time
import gc

import pandas as pd
import numpy as np
import random

from pyrosetta import Pose

from energy_calculation import calculate_and_compare_energies
from rotamer_extraction import TrackedResidue, extract_top_n_rotamers
from custom_qaoa import qaoa_func_generator, run_qaoa, run_qaoa_jax, batched_qaoa_execution
from h_mixer import custom_xy_mixer_layer, ring_xy_mixer_layer

from validation import validate_conformations, Conformation

from misc import init_basic_params, default_qaoa_params, BasicParams, QAOAParams
from h_ising_creation import extract_and_reduce_tensors, build_ising_hamiltonian

@dataclass
class RunConfig:
    qaoaParams: QAOAParams
    log_file: str

@dataclass
class LargeRunConfig:
    start: int
    end: int
    n: int

def run(h_linear: dict[int, dict[int, float]], J_quadratic: dict, global_offset: float,
        H_ising, benchmark_pose: Pose, scorefxn: object, residue_library: dict[int, TrackedResidue],
        basic_params: BasicParams, qaoa_params: QAOAParams) -> list[Conformation]:

    cost_function, sample_function = qaoa_func_generator(H_ising, ring_xy_mixer_layer, basic_params)

    # Run the Quantum Approximate Optimisation Algorithm and sample the final parameters
    final_params = run_qaoa_jax(cost_function, qaoa_params)
    probabilities = sample_function(final_params)

    # Extract the top 100 most probably conformations and check that exactly 1 rotamer for each residue is selected
    valid_conformations = validate_conformations(probabilities, basic_params)

    # Calculate both the quantum and pyrosetta energies for comparison
    try:
        calculate_and_compare_energies(valid_conformations,
                                       h_linear, J_quadratic, global_offset,
                                       benchmark_pose, scorefxn, residue_library,
                                       basic_params)
    except AssertionError as e:
        print("ERROR ERROR ERROR", e)

    return valid_conformations

def extract_rank_matches(valid_conformations: list[Conformation]):
    energies: list[dict[str, np.float64 | int | None]] = [
        {"quantum_energy": conf.quantum_energy, "probability": conf.probability} for conf in valid_conformations
    ]

    energies.sort(key=lambda conf: conf['probability'], reverse=True)
    for i, conf in enumerate(energies):
        conf['probs_idx'] = i

    energies.sort(key=lambda conf: conf['quantum_energy'])
    for i, conf in enumerate(energies):
        conf['quant_idx'] = i

    return [abs(conf['probs_idx'] - conf['quant_idx']) for _, conf in enumerate(energies)]



def run_one_residue_combo(large_run_config: LargeRunConfig, benchmark_pose: Pose, log_prefix: str, log_dir: str, df_dir: str):
    print(
        "\n================================================================================================================\n")
    print(f"====================Starting new run {large_run_config}====================")
    residue_library, ig, rot_sets, scorefxn = extract_top_n_rotamers(benchmark_pose,
                                                                     n=large_run_config.n,
                                                                     active_start=large_run_config.start,
                                                                     active_end=large_run_config.end)

    df_file = f"n_{large_run_config.n}_{large_run_config.start}-{large_run_config.end}"
    log_postfix = df_file

    # Generating QUBO (Quadratic Unconstrained Binary Optimisation) Model, and then reduce it
    h_linear, J_quadratic, global_offset = extract_and_reduce_tensors(residue_library, ig)
    basic_params: BasicParams = init_basic_params(h_linear)
    base_qaoa_params: QAOAParams = default_qaoa_params()

    # Generate the actual observable and running functions we will use in the QAOA Algorithm
    cost_hamiltonian, hamiltonian_size = build_ising_hamiltonian(h_linear, J_quadratic)
    if hamiltonian_size > 22:
        print("Hamiltonian Will Exceed Memory Available, skipping")
        return

    p_runs = [1, 2, 4, 8, 12]
    seed_versions = list(range(30))

    run_configs = [
        RunConfig(
            QAOAParams(p, base_qaoa_params.seed,
                       base_qaoa_params.optimiser_stepsize,
                       base_qaoa_params.epochs)
            , f"{log_prefix}run_layers={p}_{log_postfix}.log") for p in p_runs
    ]

    run_records = []

    for config in run_configs:
        layers = config.qaoaParams.layers
        print(f"===== Starting batched GPU run (Layers: {layers}) =====")
        start = time.perf_counter()

        # 1. Generate QNodes (Make sure they use interface="jax")
        cost_function, sample_function = qaoa_func_generator(cost_hamiltonian, ring_xy_mixer_layer, basic_params)

        # 2. RUN ALL 30 SEEDS ON THE GPU SIMULTANEOUSLY
        batched_probabilities = batched_qaoa_execution(
            cost_function,
            sample_function,
            config.qaoaParams,
            seed_versions
        )

        # 3. Classical Post-Processing and Logging
        with open(f"{log_dir}/{config.log_file}", "w") as log_file:
            for i, seed in enumerate(seed_versions):
                seed_probs = batched_probabilities[i]
                valid_conformations = validate_conformations(seed_probs, basic_params)

                try:
                    calculate_and_compare_energies(valid_conformations,
                                                   h_linear, J_quadratic, global_offset,
                                                   benchmark_pose, scorefxn, residue_library,
                                                   basic_params)
                except AssertionError as e:
                    log_file.write(f"ERROR ERROR ERROR {e}")

                prob_rank_match = extract_rank_matches(valid_conformations)

                run_records.append({
                    'layers': layers,
                    'seed': seed,
                    'rank_matches': prob_rank_match
                })

                log_file.write(f"Seed {seed} completed. Rank Match: {prob_rank_match}\n")
        gc.collect()

        end = time.perf_counter()
        time_taken = end - start
        print(
            f"===== Run Complete time taken = {time_taken:5.3f} seconds | {time_taken / len(seed_versions):5.3f} per run =====\n")

    print(f"====================Large Run Complete Saving to DF====================")

    final_df = pd.DataFrame(run_records)
    final_df.to_pickle(f"{df_dir}/{df_file}_final.pkl")

    gc.collect()