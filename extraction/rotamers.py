from dataclasses import dataclass
import logging

import pyrosetta
from pyrosetta import get_fa_scorefxn
from pyrosetta.rosetta.core import pack

@dataclass
class TrackedRotamer:
    one_body_energy: float
    original_pyrosetta_index: int
    residue: object

@dataclass
class TrackedResidue:
    moltenres_idx: int # ID within movable/active structure
    seqpos: int # ID within the wider structure
    rotamers: list[TrackedRotamer]

def load_5PTI_pose() -> pyrosetta.Pose | None:
    full_pose = pyrosetta.toolbox.pose_from_rcsb("5PTI")

    assert full_pose.total_residue() == 58, "Unexpected PDB length."
    return full_pose


def extract_top_n_rotamers(base_pose: pyrosetta.Pose, logger: logging.Logger, n=4, active_start=20, active_end=24) -> tuple[dict[int, TrackedResidue], pack.interaction_graph.InteractionGraphFactory, pack.rotamer_set.RotamerSets, object]:
    """
    Extracts the top N lowest-energy rotamers for each packable residue using a precomputed Interaction Graph.
    """
    logger.info("==================== Rotamer Energy Extraction ====================")
    residue_library = {}

    # fa --> full atom
    # scorefxn --> Score function
    # we will modify this to only take into account hyrogen interactions and
    logger.debug("Creating score function")
    scorefxn = get_fa_scorefxn()
    scorefxn(base_pose)

    logger.debug("Creating Repacking Task - Core Rotamer Optimisation Protocol")

    task = create_packing_task(base_pose, active_start, active_end)
    task.or_precompute_ig(True)

    scorefxn.setup_for_packing(base_pose, task.repacking_residues(), task.designing_residues())

    packer_neighbour_graph = pack.create_packer_graph(base_pose, scorefxn, task)

    rot_sets = pack.rotamer_set.RotamerSets()
    rot_sets.set_task(task)
    rot_sets.build_rotamers(base_pose, scorefxn, packer_neighbour_graph)
    rot_sets.prepare_sets_for_packing(base_pose, scorefxn)

    logger.debug("Computing One-Body and Two-Body Energies")
    ig = pack.interaction_graph.InteractionGraphFactory.create_and_initialize_two_body_interaction_graph(
        task, rot_sets, base_pose, scorefxn, packer_neighbour_graph
    )

    logger.debug("Iterating through molten residues - determining the top rotamer positions for each amino acid")
    for moltenres_id in range(1, rot_sets.nmoltenres() + 1):
        seqpos = rot_sets.moltenres_2_resid(moltenres_id)
        logger.debug(f"Moltenres ID: {moltenres_id}, SeqPos ID: {seqpos}")

        # Get all rotamers for the chosen residue
        n_rots = rot_sets.rotamer_set_for_moltenresidue(moltenres_id).num_rotamers()
        scored_rotamers = []

        # extract one body energies for each rotamer
        for rot_index in range(1, n_rots + 1):
            energy = ig.get_one_body_energy_for_node_state(moltenres_id, rot_index)

            rotamer_res = rot_sets.rotamer_set_for_moltenresidue(moltenres_id).rotamer(rot_index)
            scored_rotamers.append((energy, rot_index, rotamer_res))

        # Keep the lowest N energy rotamer states, throw out the rest
        scored_rotamers.sort(key=lambda x: x[0])
        top_n_rotamers = []
        for energy, rot_index, rotamer_res in scored_rotamers[:n]:
            tracked_rotamer = TrackedRotamer(
                one_body_energy=energy,
                original_pyrosetta_index=rot_index,
                residue=rotamer_res
            )
            top_n_rotamers.append(tracked_rotamer)

        residue_library[seqpos] = TrackedResidue(moltenres_id, seqpos, top_n_rotamers)

    logger.info("==================== Rotamer Energy Extraction Complete ====================")

    return residue_library, ig, rot_sets, scorefxn


def create_packing_task(base_pose, active_start, active_end):
    tf = pack.task.TaskFactory()
    tf.push_back(pack.task.operation.RestrictToRepacking())
    packer_task = tf.create_task_and_apply_taskoperations(base_pose)

    total_residues = base_pose.total_residue()
    for res_idx in range(1, total_residues + 1):
        res_task = packer_task.nonconst_residue_task(res_idx)
        if res_idx < active_start or res_idx > active_end:
            res_task.prevent_repacking()
        else:
            res_task.restrict_to_repacking()
    return packer_task

