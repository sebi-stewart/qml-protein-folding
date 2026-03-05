from typing import Tuple

import pyrosetta
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack import create_packer_graph, pack_rotamers_setup

def extract_top_n_rotamers(pose, n=4) -> Tuple[dict, InteractionGraphFactory, RotamerSets]:
    """
    Extracts the top N lowest-energy rotamers for each packable residue using a precomputed Interaction Graph.
    """
    rotamer_library = {}
    
    scorefxn = pyrosetta.get_fa_scorefxn()
    scorefxn(pose) 
    
    task = TaskFactory.create_packer_task(pose)
    task.restrict_to_repacking()
    task.or_precompute_ig(True) 
    
    pose.update_residue_neighbors()
    packer_neighbour_graph = create_packer_graph(pose, scorefxn, task)
    
    rot_sets = RotamerSets()
    pack_rotamers_setup(pose, scorefxn, task, rot_sets)
    
    ig = InteractionGraphFactory.create_and_initialize_two_body_interaction_graph(
        task, rot_sets, pose, scorefxn, packer_neighbour_graph
    )
    
    for moltenres_id in range(1, rot_sets.nmoltenres() + 1):
        seqpos = rot_sets.moltenres_2_resid(moltenres_id)
        
        n_rots = rot_sets.rotamer_set_for_moltenresidue(moltenres_id).num_rotamers()
        
        scored_rotamers = []
        
        for rot_index in range(1, n_rots + 1):
            energy = ig.get_one_body_energy_for_node_state(moltenres_id, rot_index)
            
            rotamer_res = rot_sets.rotamer_set_for_moltenresidue(moltenres_id).rotamer(rot_index)
            scored_rotamers.append((energy, rot_index, rotamer_res))
            
        scored_rotamers.sort(key=lambda x: x[0])

        top_n_data = [(energy, rot_index, res) for energy, rot_index, res in scored_rotamers[:n]]
        rotamer_library[seqpos] = top_n_data
        
    return rotamer_library, ig, rot_sets