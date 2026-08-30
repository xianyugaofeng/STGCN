from .STGCN import STGCN
from .STID import STID
from .GraphWaveNet import GraphWaveNet

MODEL_ZOO = {
    'STGCN': STGCN,
    'STID': STID,
    'GraphWaveNet': GraphWaveNet
}

def get_model(model_name):
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Model {model_name} not found in MODEL_ZOO")
    return MODEL_ZOO[model_name]

__all__ = ['STGCN', 'STID', 'GraphWaveNet','get_model']