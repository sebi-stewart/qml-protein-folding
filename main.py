import pyrosetta

from benchmark import full_bpti_benchmark
from initialisation import initialize_rosetta
from run import LargeRunConfig, run_one_residue_combo

import logging
from pathlib import Path
import sys

log_dir_base = "overnight_runs"
df_dir = "overnight_results"

for path in (log_dir_base, df_dir):
  Path(f"./{path}").mkdir(parents=True, exist_ok=True)

log_dir = Path.cwd().joinpath(log_dir_base)
log_path = log_dir.joinpath("myapp.log")

# 1. Reset root logger and set it to WARNING to silence JAX/libraries
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers[:]:
        root.removeHandler(handler)
root.setLevel(logging.WARNING)

# 2. Create a dedicated logger for your application code
logger = logging.getLogger("qaoa_pf")
logger.setLevel(logging.DEBUG)
logger.propagate = False # Prevent messages from reaching the noisy root logger

# 3. Setup File Handler (DEBUG and higher to file)
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
file_handler.setFormatter(file_formatter)

# 4. Setup Console Handler (INFO and higher to stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console_handler.setFormatter(console_formatter)

# 5. Add handlers to your specific logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Logging initialized. Application logs are isolated from JAX.")
logger.debug("This debug message will only appear in the file.")

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
