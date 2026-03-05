import pyrosetta

def bpti_ryfyn_benchmark():
    # fetch from the internet
    full_pose = pyrosetta.toolbox.pose_from_rcsb("5PTI")
    
    assert full_pose.total_residue() == 58, "Unexpected PDB length."

    # Remove everything apart from 20-24 --> Now we have a 5 residue length
    full_pose.delete_residue_range_slow(25, full_pose.total_residue())
    full_pose.delete_residue_range_slow(1, 19)

    fragment_pose = full_pose

    print(f"Fragment Sequence: {fragment_pose.sequence()}")
    assert fragment_pose.sequence() == "RYFYN", "Fragment extraction failed."
    print(f"Total Residues: {fragment_pose.total_residue()}")
    
    return fragment_pose
