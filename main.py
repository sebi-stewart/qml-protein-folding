import contextlib
import time

import pandas as pd
import pyrosetta

from benchmark import full_bpti_benchmark
from misc import init_basic_params, default_qaoa_params, BasicParams, QAOAParams
from rotamer_extraction import extract_top_n_rotamers
from h_ising_creation import extract_and_reduce_tensors
from initialisation import initialize_rosetta
from run import run, RunConfig, extract_rank_matches
import sys

old_stdout = sys.stdout

log_dir = "logs"


if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    # Pyrosetta Relevant Code
    benchmark_pose = full_bpti_benchmark()
    residue_library, ig, rot_sets, scorefxn = extract_top_n_rotamers(benchmark_pose)

    # Generating QUBO (Quadratic Unconstrained Binary Optimisation) Model, and then reduce it
    h_linear, J_quadratic, global_offset = extract_and_reduce_tensors(residue_library, ig)
    basic_params: BasicParams = init_basic_params(h_linear)
    base_qaoa_params: QAOAParams = default_qaoa_params()

    p_runs = [1, 2, 3, 4, 8, 12, 16, 32]
    seed_versions = [1, 2, 3, 5, 7, 42]

    run_configs = [
        RunConfig(
            QAOAParams(p, base_qaoa_params.seed,
                       base_qaoa_params.optimiser_stepsize,
                       base_qaoa_params.epochs)
            , f"run_layers={p}_s.log") for p in p_runs
    ]

    results: dict[int, list[int]] = {}
    run_records = []

    for config in run_configs:
        layers = config.qaoaParams.layers
        print(f"===== Starting run (Layers: {layers}) - {len(seed_versions)} seeds =====")

        start = time.perf_counter()

        with open(f"{log_dir}/{config.log_file}", "w") as log_file:
            with contextlib.redirect_stdout(log_file):
                for seed in seed_versions:
                    config.qaoaParams.seed = seed

                    # Execute the quantum pipeline
                    valid_conformations = run(
                        h_linear, J_quadratic, global_offset,
                        benchmark_pose, scorefxn, residue_library,
                        basic_params, config.qaoaParams
                    )

                    # Extract metrics
                    prob_rank_match = extract_rank_matches(valid_conformations)

                    # SAVE RAW DATA: Append every single run as an independent record
                    run_records.append({
                        'layers': layers,
                        'seed': seed,
                        'rank_matches': prob_rank_match
                    })

        end = time.perf_counter()
        time_taken = end - start
        print(f"===== Run Complete time taken = {time_taken:5.3f} seconds | {time_taken / len(seed_versions):5.3f} per run =====\n")


    print("\n\n=============== All Runs Complete Saving to DF ===============\n\n")

    df = pd.DataFrame(run_records)
    df.to_pickle("results/run2_raw.pkl")