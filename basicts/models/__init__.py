from .STGCN import STGCN
from .STID import STID

MODEL_ZOO = {
    'STGCN': STGCN,
    'STID': STID
}

def get_model(model_name):
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Model {model_name} not found in MODEL_ZOO")
    return MODEL_ZOO[model_name]

__all__ = ['STGCN', 'STID', 'get_model']