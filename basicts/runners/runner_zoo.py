import torch
import numpy as np
from .base_runner import BaseRunner
from .stgcn_runner import STGCNRunner
from .stid_runner import STIDRunner
from .gwnet_runner import GraphWaveNetRunner

RUNNER_ZOO = {
    'BaseRunner': BaseRunner,
    'STGCNRunner': STGCNRunner,
    'STIDRunner': STIDRunner,
    'GraphWaveNetRunner': GraphWaveNetRunner
}

def get_runner(runner_name):
    if runner_name not in RUNNER_ZOO:
        raise ValueError(f"Runner {runner_name} not found in RUNNER_ZOO")
    return RUNNER_ZOO[runner_name]