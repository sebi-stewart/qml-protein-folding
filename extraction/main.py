import logging
from dataclasses import dataclass
import pathlib
from collections.abc import Callable

import pyrosetta

from extraction.initialisation import initialize_rosetta

from extraction.qubo_creation import extract_and_reduce_tensors
from extraction.rotamers import extract_top_n_rotamers, load_5PTI_pose
from extraction.saving import save_results, ENERGIES_SMALL, ENERGIES_LARGE, ENERGIES_TOO_LARGE, save_results_alternate, \
    ALT_ENERGIES_FOLDER_COLLECTION, ENERGIES_ALT_FOLDER
from logging_setup import setup_logging


@dataclass
class ExtractionTestInstance:
    pose_func: Callable[[], pyrosetta.Pose]
    test_name: str
    residue_start: int
    residue_end: int
    rotamer_count: int

class TestInstanceFactory:
    @staticmethod
    def create_test_instance(protein: str, start: int, end: int, rot_count: int) -> ExtractionTestInstance:
        test_name = f"{protein}_{start}_{end}_{rot_count}"

        if protein == "5PTI": pose_func = load_5PTI_pose
        else: raise ValueError(f"Unknown protein: {protein}")

        return ExtractionTestInstance(
            pose_func=pose_func,
            test_name=test_name,
            residue_start=start,
            residue_end=end,
            rotamer_count=rot_count
        )

    @staticmethod
    def create_test_instance_from_func(pose_func: Callable[[], pyrosetta.Pose], test_name: str, start: int, end: int, rot_count: int) -> ExtractionTestInstance:
        return ExtractionTestInstance(
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

def main(inst: ExtractionTestInstance):
    test_name = inst.test_name
    logger = logging.getLogger(f"qaoa.{test_name}")

    pose, residue_library, ig, rot_sets, scorefxn = run_pyrosetta_obj_extraction(
        inst.pose_func,
        logger=logger,
        n=inst.rotamer_count,
        active_start=inst.residue_start,
        active_end=inst.residue_end)

    one_body, two_body, global_offset = from_energies_to_tensors(residue_library, ig)

    return save_results_alternate(one_body, two_body, logger, f"{test_name}.pkl")

def _setup_folders():
    pathlib.Path(ENERGIES_SMALL).mkdir(exist_ok=True, parents=True)
    pathlib.Path(ENERGIES_LARGE).mkdir(exist_ok=True, parents=True)
    pathlib.Path(ENERGIES_TOO_LARGE).mkdir(exist_ok=True, parents=True)

    pathlib.Path(ENERGIES_ALT_FOLDER).mkdir(exist_ok=True, parents=True)
    for folder in ALT_ENERGIES_FOLDER_COLLECTION:
        pathlib.Path(folder).mkdir(exist_ok=True, parents=True)

def setup_extraction():
    setup_logging("new_runs_qaoa")
    _setup_folders()
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    return TestInstanceFactory()

if __name__ == '__main__':
    logger = logging.getLogger("qaoa.main")
    fac = setup_extraction()

    test_instances = []
    logger.info("Creating test instances...")
    for residue_length in range(4, 9):
        for start_pos in range(10, 25):
            # test_instances.append(fac.create_test_instance("5PTI", start_pos, start_pos + residue_length - 1, rot_count=3))
            # test_instances.append(fac.create_test_instance("5PTI", start_pos, start_pos + residue_length - 1, rot_count=4))
            test_instances.append(fac.create_test_instance("5PTI", start_pos, start_pos + residue_length - 1, rot_count=5))
            test_instances.append(fac.create_test_instance("5PTI", start_pos, start_pos + residue_length - 1, rot_count=6))
            test_instances.append(fac.create_test_instance("5PTI", start_pos, start_pos + residue_length - 1, rot_count=7))
    logger.info(f"Created {len(test_instances)} test instances.")


    for inst in test_instances:
        file_location = main(inst)
        logger.info(f"Completed extraction for {inst.test_name} - saved to {file_location}")


