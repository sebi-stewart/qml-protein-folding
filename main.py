import pyrosetta

from benchmark import full_bpti_benchmark
from initialisation import initialize_rosetta
from logging_setup import setup_logging
from run import LargeRunConfig, run_one_residue_combo

from pathlib import Path

log_dir_base = "overnight_runs"
df_dir = "overnight_results"

for path in (log_dir_base, df_dir):
  Path(f"./{path}").mkdir(parents=True, exist_ok=True)

setup_logging(log_dir_base)

# df_file = "run_5_ring_mixer"

large_runs = [
    # 5 residues long
    LargeRunConfig(19, 23, 4),
    LargeRunConfig(19, 23, 5),
    LargeRunConfig(20, 24, 4),
    LargeRunConfig(20, 24, 5),
    LargeRunConfig(21, 25, 4),
    LargeRunConfig(21, 25, 5),

    # 6 Residues Long
    LargeRunConfig(19, 24, 4),
    LargeRunConfig(19, 24, 5),
    LargeRunConfig(20, 25, 4),
    LargeRunConfig(20, 25, 5),
    LargeRunConfig(21, 26, 4),
    LargeRunConfig(21, 26, 5),

    # 7 Residues Long
    LargeRunConfig(18, 24, 4),
    LargeRunConfig(18, 24, 5),
    LargeRunConfig(19, 25, 4),
    LargeRunConfig(19, 25, 5),
    LargeRunConfig(20, 26, 4),
    LargeRunConfig(20, 26, 5),

    # 8 Residues Long
    LargeRunConfig(17, 24, 4),
    LargeRunConfig(17, 24, 5),
    LargeRunConfig(18, 25, 4),
    LargeRunConfig(18, 25, 5),
    LargeRunConfig(20, 27, 4),
    LargeRunConfig(20, 27, 5),
]

explorable_state_space = 0
for large_config in large_runs:
    explorable_state_space += (large_config.n - 1)**(large_config.end - large_config.start + 1)
current_explored_state_space = 0


if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    # Pyrosetta Relevant Code
    benchmark_pose = full_bpti_benchmark()


    for large_config in large_runs:
        run_one_residue_combo(large_config, benchmark_pose, df_dir)

        current_explored_state_space += (large_config.n - 1)**(large_config.end - large_config.start + 1)
        print(f" =============================== {current_explored_state_space/explorable_state_space:3.4f}% COMPLETE ({current_explored_state_space}/{explorable_state_space}) =============================== ")
