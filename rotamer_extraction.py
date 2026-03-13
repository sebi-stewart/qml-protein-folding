from typing import Tuple, Dict, List

import pyrosetta
from pyrosetta.rosetta.core.chemical import DISULFIDE
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack import create_packer_graph, pack_rotamers_setup
from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue
from dataclasses import dataclass

@dataclass
class TrackedRotamer:
    one_body_energy: float
    original_pyrosetta_index: int
    residue: object


def get_score_function():
    scorefxn = pyrosetta.rosetta.core.scoring.ScoreFunction()

    # Internal Side chain energies:
    # fa_dun                                     Internal energy of sidechain rotamers as derived from Dunbrack's statistics (2010 Rotamer Library used in Talaris2013).  Supports any residue type for which a rotamer library is avalable.
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 0.7)

    # Hydrogen bonding
    # hbond_bb_sc                                Sidechain-backbone hydrogen bond energy.
    # hbond_sc                                   Sidechain-sidechain hydrogen bond energy.
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 1.0)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_sc, 1.0)

    # Hyrdrophobic interactions
    # fa_atr                                     Lennard-Jones attractive between atoms in different residues.  Supports canonical and noncanonical residue types.
    # fa_rep                                     Lennard-Jones repulsive between atoms in different residues.  Supports canonical and noncanonical residue types.
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_atr , 1.0)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1.0)


    return scorefxn


def extract_top_n_rotamers(pose, n=4) -> Tuple[Dict[int, List[TrackedRotamer]], InteractionGraphFactory, RotamerSets, object]:
    """
    Extracts the top N lowest-energy rotamers for each packable residue using a precomputed Interaction Graph.
    """
    rotamer_library = {}

    # fa --> full atom
    # scorefxn --> Score function
    # we will modify this to only take into account hyrogen interactions and
    print("Creating score function")
    scorefxn = get_score_function()
    pose = safe_score_pose(scorefxn, pose)

    print("Creating Repacking Task - Core Rotamer Optimisation Protocol")
    task = TaskFactory.create_packer_task(pose)
    task.restrict_to_repacking()
    task.or_precompute_ig(True) 

    pose.update_residue_neighbors()
    packer_neighbour_graph = create_packer_graph(pose, scorefxn, task)

    rot_sets = RotamerSets()
    pyrosetta.rosetta.core.io.pdb.dump_pdb(pose, "pose-pre-packing.pdb")
    pack_rotamers_setup(pose, scorefxn, task, rot_sets)
    pyrosetta.rosetta.core.io.pdb.dump_pdb(pose, "pose-post-packing.pdb")

    print("Computing One-Body and Two-Body Energies")
    ig = InteractionGraphFactory.create_and_initialize_two_body_interaction_graph(
        task, rot_sets, pose, scorefxn, packer_neighbour_graph
    )
    pyrosetta.rosetta.core.io.pdb.dump_pdb(pose, "pose-post-ig.pdb")

    print("Iterating through molten residues - determining the top rotamer positions for each amino acid")
    for moltenres_id in range(1, rot_sets.nmoltenres() + 1):
        seqpos = rot_sets.moltenres_2_resid(moltenres_id)
        assert seqpos == moltenres_id, f"{seqpos} != {moltenres_id}"
        print(seqpos, moltenres_id)

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
        top_n_tracked = []
        for energy, rot_index, rotamer_res in scored_rotamers[:n]:
            tracked_rotamer = TrackedRotamer(
                one_body_energy=energy,
                original_pyrosetta_index=rot_index,
                residue=rotamer_res
            )
            top_n_tracked.append(tracked_rotamer)

        rotamer_library[seqpos] = top_n_tracked
        
    return rotamer_library, ig, rot_sets, scorefxn


def safe_score_pose(scorefxn, pose, max_retries=3):
    attempts = 0

    clean_backup = pose.clone()

    while attempts < max_retries:
        try:
            scorefxn(pose)
            print("Pose scored successfully!")
            return pose

        except RuntimeError as e:
            error_msg = str(e)
            fixed = False
            attempts += 1

            if "FullatomDisulfideEnergyContainer.cc" in error_msg:
                fixed = fix_disulfide_bond(clean_backup)

            if fixed:
                print("Retrying score...\n")
                pose.assign(clean_backup)
            else:
                print("Could not find (or fix) error for", error_msg)
                raise e
    raise RuntimeError("Exceeded maximum retries for fixing the pose.")


def fix_disulfide_bond(pose):
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)

        if not res.name3() == "CYS" or not res.has_variant_type(DISULFIDE):
            continue
        # Connection 3 is the side-chain SG-SG bond. Let's find the partner!
        partner_id = res.connect_map(3).resid()

        # Check if the partner is missing (0) or outside our fragment
        if partner_id == 0 or partner_id > pose.total_residue():
            print(f" -> Dangling disulfide at CYS {i} (pointed to missing partner {partner_id}). Fixing...")
            remove_variant_type_from_pose_residue(pose, DISULFIDE, i)
            return True
        # else:
        #     print(f" -> Intact disulfide found between CYS {i} and CYS {partner_id}. Preserving!")
    return False