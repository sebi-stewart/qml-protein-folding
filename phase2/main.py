import logging

from extraction import
import pyrosetta

from extraction.initialisation import initialize_rosetta
from extraction.main import TestInstanceFactory, ExtractionTestInstance
from logging_setup import setup_logging

import numpy as np



def extract_qaoa_relevant_data(logger: logging.Logger, qaoa_file_path: str):
    assert qaoa_file_path.endswith(".npz"), "Expected a .npz file containing the QAOA results"
    data = np.load(qaoa_file_path, allow_pickle=True)

    qaoa_final_sample_func_generator

if __name__ == "__main__":
    setup_logging("rescoring_phase2", "5PTI")
    initialize_rosetta(pyrosetta, extra_flags="-mute all")
    fac = TestInstanceFactory()

    inst = fac.create_test_instance_from_func(
        pose_func=lambda : pyrosetta.pose_from_pdb("data/AF-P00974-F1-model_v6.pdb"),
        test_name="AF-5PTI",
        start=20,
        end=24,
        rot_count=4
    )





