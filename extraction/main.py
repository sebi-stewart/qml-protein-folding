import logging
from dataclasses import dataclass
import pathlib
from collections.abc import Callable

import pyrosetta

from extraction.initialisation import initialize_rosetta

from extraction.qubo_creation import extract_and_reduce_tensors
from extraction.rotamers import extract_top_n_rotamers, load_5PTI_pose
from extraction.saving import save_results, ENERGIES_SMALL, ENERGIES_LARGE, ENERGIES_TOO_LARGE
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

    return save_results(one_body, two_body, logger, f"{test_name}.pkl")

def _setup_folders():
    pathlib.Path(ENERGIES_SMALL).mkdir(exist_ok=True, parents=True)
    pathlib.Path(ENERGIES_LARGE).mkdir(exist_ok=True, parents=True)

def setup_extraction():
    setup_logging("new_runs_qaoa")
    _setup_folders()
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    return TestInstanceFactory()

if __name__ == '__main__':
    logger = logging.getLogger("qaoa.main")
    fac = setup_extraction()

    test_instances = [
        fac.create_test_instance("5PTI", 18, 22, 4),
        fac.create_test_instance("5PTI", 18, 22, 5),
        fac.create_test_instance("5PTI", 19, 23, 4),
        fac.create_test_instance("5PTI", 19, 23, 5),
        fac.create_test_instance("5PTI", 20, 24, 4),
        fac.create_test_instance("5PTI", 20, 24, 5),
        fac.create_test_instance("5PTI", 21, 25, 4),
        fac.create_test_instance("5PTI", 21, 25, 5),
        fac.create_test_instance("5PTI", 22, 26, 4),
        fac.create_test_instance("5PTI", 22, 26, 5),

        fac.create_test_instance("5PTI", 18, 23, 4),
        fac.create_test_instance("5PTI", 18, 23, 5),
        fac.create_test_instance("5PTI", 19, 24, 4),
        fac.create_test_instance("5PTI", 19, 24, 5),
        fac.create_test_instance("5PTI", 20, 25, 4),
        fac.create_test_instance("5PTI", 20, 25, 5),
        fac.create_test_instance("5PTI", 21, 26, 4),
        fac.create_test_instance("5PTI", 21, 26, 5),
        fac.create_test_instance("5PTI", 22, 27, 4),
        fac.create_test_instance("5PTI", 22, 27, 5),

        fac.create_test_instance("5PTI", 18, 24, 4),
        fac.create_test_instance("5PTI", 18, 24, 5),
        fac.create_test_instance("5PTI", 19, 25, 4),
        fac.create_test_instance("5PTI", 19, 25, 5),
        fac.create_test_instance("5PTI", 20, 26, 4),
        fac.create_test_instance("5PTI", 20, 26, 5),
        fac.create_test_instance("5PTI", 21, 27, 4),
        fac.create_test_instance("5PTI", 21, 27, 5),
        fac.create_test_instance("5PTI", 22, 28, 4),
        fac.create_test_instance("5PTI", 22, 28, 5),
    ]


    for inst in test_instances:
        file_location = main(inst)
        logger.info(f"Completed extraction for {inst.test_name} - saved to {file_location}")

    small_files = list(pathlib.Path(ENERGIES_SMALL).glob("*.pkl"))
    large_files = list(pathlib.Path(ENERGIES_LARGE).glob("*.pkl"))
    too_large = list(pathlib.Path(ENERGIES_TOO_LARGE).glob("*.pkl"))

    logger.info(f"Extraction complete. {len(small_files)} small files and {len(large_files)} large files saved.")
    logger.warning(f"{len(too_large)} files were categorized as too large for the current saving scheme and may require special handling.")

