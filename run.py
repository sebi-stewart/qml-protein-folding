from dataclasses import dataclass

import numpy as np
from pyrosetta import Pose

from energy_calculation import calculate_and_compare_energies
from misc import BasicParams, QAOAParams
from rotamer_extraction import TrackedResidue
from custom_qaoa import qaoa_func_generator, run_qaoa
from h_mixer import custom_xy_mixer_layer, ring_xy_mixer_layer

from validation import validate_conformations, Conformation

@dataclass
class RunConfig:
    qaoaParams: QAOAParams
    log_file: str

def run(h_linear: dict[int, dict[int, float]], J_quadratic: dict, global_offset: float,
        H_ising, benchmark_pose: Pose, scorefxn: object, residue_library: dict[int, TrackedResidue],
        basic_params: BasicParams, qaoa_params: QAOAParams) -> list[Conformation]:

    cost_function, sample_function = qaoa_func_generator(H_ising, ring_xy_mixer_layer, basic_params)

    # Run the Quantum Approximate Optimisation Algorithm and sample the final parameters
    final_params = run_qaoa(cost_function, qaoa_params)
    probabilities = sample_function(final_params)

    # Extract the top 100 most probably conformations and check that exactly 1 rotamer for each residue is selected
    valid_conformations = validate_conformations(probabilities, basic_params)

    # Calculate both the quantum and pyrosetta energies for comparison
    calculate_and_compare_energies(valid_conformations,
                                   h_linear, J_quadratic, global_offset,
                                   benchmark_pose, scorefxn, residue_library,
                                   basic_params)

    return valid_conformations

def extract_rank_matches(valid_conformations: list[Conformation]):
    energies: list[dict[str, np.float64 | int | None]] = [
        {"quantum_energy": conf.quantum_energy, "probability": conf.probability} for conf in valid_conformations
    ]

    energies.sort(key=lambda conf: conf['probability'], reverse=True)
    for i, conf in enumerate(energies):
        conf['probs_idx'] = i

    energies.sort(key=lambda conf: conf['quantum_energy'])
    for i, conf in enumerate(energies):
        conf['quant_idx'] = i

    return [abs(conf['probs_idx'] - conf['quant_idx']) for _, conf in enumerate(energies)]