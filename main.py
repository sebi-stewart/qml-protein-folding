import pyrosetta

from benchmark import full_bpti_benchmark
from initialisation import initialize_rosetta
from run import LargeRunConfig, run_one_residue_combo


log_dir = "overnight_runs"
log_prefix = ""
df_dir = "overnight_results"
# df_file = "run_5_ring_mixer"

large_runs = [
    LargeRunConfig(19, 23, 4),
    LargeRunConfig(19, 23, 5),
    LargeRunConfig(20, 24, 4),
    LargeRunConfig(20, 24, 5),
    LargeRunConfig(21, 25, 4),
    LargeRunConfig(21, 25, 5),
    LargeRunConfig(20, 25, 4),
    LargeRunConfig(19, 24, 4)
]


if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    # Pyrosetta Relevant Code
    benchmark_pose = full_bpti_benchmark()

    for large_config in large_runs:
        run_one_residue_combo(large_config, benchmark_pose, log_prefix, log_dir, df_dir)
