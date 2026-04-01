import logging
from dataclasses import dataclass

import pyrosetta

from initialisation import initialize_rosetta

from extraction.qubo_creation import extract_and_reduce_tensors
from extraction.rotamers import extract_top_n_rotamers

@dataclass
class TestInstance:
    pose_func: callable
    test_name: str
    residue_start: int
    residue_end: int
    rotamer_count: int

def run_pyrosetta_obj_extraction(pose_func, logger: logging.Logger, n=4, active_start=20, active_end=24):
    pose = pose_func()
    residue_library, ig, rot_sets, scorefxn = extract_top_n_rotamers(
        pose,
        logger=logger,
        n=n,
        active_start=active_start,
        active_end=active_end
    )

    return pose, residue_library, ig, rot_sets, scorefxn

def from_energies_to_tensors(residue_library, ig):
    # Placeholder for the actual tensor extraction logic
    h_flex_linear, J_flex_quadratic, global_offset = extract_and_reduce_tensors(residue_library, ig)
    return h_flex_linear, J_flex_quadratic, global_offset

def main(inst: TestInstance, test_name: str, ):
    logger = logging.getLogger(f"extraction.{test_name}")

    pose, residue_library, ig, rot_sets, scorefxn = run_pyrosetta_obj_extraction(
        inst.pose_func,
        logger=logger,
        n=inst.rotamer_count,
        active_start=inst.residue_start,
        active_end=inst.residue_end)

    h_flex_linear, J_flex_quadratic, global_offset = from_energies_to_tensors(residue_library, ig)


if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

