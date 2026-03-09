import pyrosetta
from pyrosetta import Pose


def bpti_ryfyn_benchmark(start=20, end=24) -> Pose | None:
    assert start < end, f"Start {start} or greater than or equal to end {end}"
    # fetch from the internet
    full_pose = pyrosetta.toolbox.pose_from_rcsb("5PTI")
    
    assert full_pose.total_residue() == 58, "Unexpected PDB length."

    # Remove everything apart from 20-24 --> Now we have a 5 residue length
    full_pose.delete_residue_range_slow(end+1, full_pose.total_residue())
    full_pose.delete_residue_range_slow(1, start-1)

    fragment_pose = full_pose

    print(f"Fragment Sequence: {fragment_pose.sequence()}")
    print(f"Total Residues: {fragment_pose.total_residue()}")
    
    return fragment_pose
