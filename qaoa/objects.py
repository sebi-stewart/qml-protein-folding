from dataclasses import dataclass

from constants import QAOA_LAYERS, OPTIMISER_STEPSIZE, OPTIMISER_EPOCHS


@dataclass
class QAOAParams:
    layers: int
    optimiser_stepsize: float
    epochs: int

def default_qaoa_params() -> QAOAParams:
    return QAOAParams(
        QAOA_LAYERS,
        OPTIMISER_STEPSIZE,
        OPTIMISER_EPOCHS
    )