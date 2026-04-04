import logging
from dataclasses import dataclass
import pickle
import pathlib

import pyrosetta

from initialisation import initialize_rosetta

from extraction.qubo_creation import extract_and_reduce_tensors
from extraction.rotamers import extract_top_n_rotamers, load_5PTI_pose
from logging_setup import setup_logging


@dataclass
class TestInstance:
    pose_func: callable
    test_name: str
    residue_start: int
    residue_end: int
    rotamer_count: int

class TestInstanceFactory:
    @staticmethod
    def create_test_instance(protein: str, start: int, end: int, rot_count: int) -> TestInstance:
        test_name = f"{protein}_{start}_{end}_{rot_count}"

        if protein == "5PTI": pose_func = load_5PTI_pose
        else: raise ValueError(f"Unknown protein: {protein}")

        return TestInstance(
            pose_func=pose_func,
            test_name=test_name,
            residue_start=start,
            residue_end=end,
            rotamer_count=rot_count
        )

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


def save_results(one_body, two_body, logger, artifact_path):
    # Placeholder for the actual saving logic
    with open(f"energies/{artifact_path}", 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump({
            'one_body': one_body,
            'two_body': two_body
        }, f)
    logger.info(f"Saved extracted tensors to {artifact_path}")


def main(inst: TestInstance):
    test_name = inst.test_name
    logger = logging.getLogger(f"extraction.{test_name}")

    pose, residue_library, ig, rot_sets, scorefxn = run_pyrosetta_obj_extraction(
        inst.pose_func,
        logger=logger,
        n=inst.rotamer_count,
        active_start=inst.residue_start,
        active_end=inst.residue_end)

    one_body, two_body, global_offset = from_energies_to_tensors(residue_library, ig)

    save_results(one_body, two_body, logger, f"{test_name}.pkl")

if __name__ == '__main__':
    setup_logging("new_runs_qaoa")

    initialize_rosetta(pyrosetta, extra_flags="-mute all")
    factory = TestInstanceFactory()

    pathlib.Path("energies").mkdir(exist_ok=True)

    main(factory.create_test_instance("5PTI", 20, 24, 4))

