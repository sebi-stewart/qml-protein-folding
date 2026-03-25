from constants import *
from dataclasses import dataclass

@dataclass
class BasicParams:
    wire_offsets: dict
    seq_positions: list[int]
    rotamer_counts: dict
    num_qubits: int

def init_basic_params(h_flex_linear) -> BasicParams:
    print("initializing basic params")

    seq_positions = sorted(list(h_flex_linear.keys()))
    wire_offsets = {}
    current_wire = 0
    rotamer_counts = {}
    for seq in seq_positions:
        wire_offsets[seq] = current_wire
        rotamer_counts[seq] = len(h_flex_linear[seq])
        current_wire += len(h_flex_linear[seq])

    num_qubits = sum(rotamer_counts.values())

    return BasicParams(
        wire_offsets,
        seq_positions,
        rotamer_counts,
        num_qubits
    )

@dataclass
class QAOAParams:
    layers: int
    seed: int
    optimiser_stepsize: float
    epochs: int

def default_qaoa_params() -> QAOAParams:
    return QAOAParams(
        QAOA_LAYERS,
        RAND_SEED,
        OPTIMISER_STEPSIZE,
        OPTIMISER_EPOCHS
    )