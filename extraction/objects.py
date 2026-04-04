from dataclasses import dataclass


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