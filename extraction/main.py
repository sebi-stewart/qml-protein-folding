import os, sys
sys.path.append(os.getcwd())  # Ensures import work fine when running from the root directory of the project
import utils.make_paths_absolute # Important for file paths

import logging
from dataclasses import dataclass
from collections.abc import Callable

import pyrosetta

from extraction.initialisation import initialize_rosetta

from extraction.qubo_creation import extract_and_reduce_tensors
from extraction.rotamers import extract_top_n_rotamers, load_5PTI_pose
from extraction.saving import save_results_alternate, setup_folders
from utils.logging_setup import setup_logging

import argparse


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
        test_name = f"{test_name}_{start}_{end}_{rot_count}"
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

    return save_results_alternate(one_body, two_body, logger, f"{test_name}.json")


def setup_extraction(log_dir: str, output_dir: str, log_file: str) -> TestInstanceFactory:
    setup_logging(log_dir, log_file)
    setup_folders(output_dir)
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    return TestInstanceFactory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, default="AF-5PTI", help="Name of the test instance")
    parser.add_argument("--input_pdb", type=str, default="inputs/structural_files/AF-P00974-F1-model_v6.pdb", help="Path to the input PDB file")
    parser.add_argument("--output_dir", type=str, default="intermediates/energy_mappings", help="Directory to save the extraction results")
    parser.add_argument("--log_dir", type=str, default="outputs/logs", help="Directory to save the logs")
    parser.add_argument("--log_file", type=str, default="extraction_main", help="Name of the log file")

    # The following arguments are for the residue segment and rotamer extraction
    parser.add_argument("--min_residue_length", type=int, default=3, help="Minimum length of the residue segment")
    parser.add_argument("--max_residue_length", type=int, default=4, help="Maximum length of the residue segment - exclusive upper bound")

    parser.add_argument("--min_start_pos", type=int, default=4, help="Minimum starting residue position (range: 4-5 for AF-5PTI)")
    parser.add_argument("--max_start_pos", type=int, default=5, help="Maximum starting residue position (range: 4-5 for AF-5PTI) - exclusive upper bound")

    parser.add_argument("--min_rot_count", type=int, default=2, help="Minimum number of rotamers to extract")
    parser.add_argument("--max_rot_count", type=int, default=3, help="Maximum number of rotamers to extract - exclusive upper bound")
    args = parser.parse_args()

    logger = logging.getLogger("qaoa.main")
    fac = setup_extraction(args.log_dir, args.output_dir, args.log_file)
    input_pdb = args.input_pdb

    min_residue_length = args.min_residue_length
    max_residue_length = args.max_residue_length
    min_start_pos = args.min_start_pos
    max_start_pos = args.max_start_pos
    min_rot_count = args.min_rot_count
    max_rot_count = args.max_rot_count

    print(f"Running extraction for test instances with residue lengths {min_residue_length}-{max_residue_length-1}, start positions {min_start_pos}-{max_start_pos-1}, and rotamer counts {min_rot_count}-{max_rot_count-1}.")

    test_instances = []
    logger.info("Creating test instances...")
    for residue_length in range(min_residue_length, max_residue_length):
        for start_pos in range(min_start_pos, max_start_pos):
            for rot_count in range(min_rot_count, max_rot_count):
                inst = fac.create_test_instance_from_func(
                    pose_func=lambda : pyrosetta.pose_from_pdb(input_pdb),
                    test_name=args.test_name,
                    start=start_pos,
                    end=start_pos + residue_length - 1,
                    rot_count=rot_count
                )
                test_instances.append(inst)
    logger.info(f"Created {len(test_instances)} test instances.")


    for inst in test_instances:
        file_location = main(inst)
        logger.info(f"Completed extraction for {inst.test_name} - saved to {file_location}")


