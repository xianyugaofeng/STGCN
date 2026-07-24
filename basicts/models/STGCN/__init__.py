from .STGCN import STGCN

MODEL_ZOO = {
    'STGCN': STGCN,
}

def get_model(model_name):
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Model {model_name} not found in MODEL_ZOO")
    return MODEL_ZOO[model_name]