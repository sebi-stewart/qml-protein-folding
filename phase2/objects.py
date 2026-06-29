"""
Lightweight container definitions used during phase 2 rescoring.
Holds simple dataclasses to represent rescoring conformations and their metadata.
"""

from dataclasses import dataclass
import pyrosetta
import numpy as np

@dataclass
class RescoringConformation:
    bitstring: list[int] | None
    pose: pyrosetta.Pose = None
    biological_energy: np.float64 = None
    energy_diff: np.float64 = None