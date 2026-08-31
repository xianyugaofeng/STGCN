import torch
import numpy as np
from .base_runner import BaseRunner

class STIDRunner(BaseRunner):
    def __init__(self, config, adj_matrix=None):
        super().__init__(config, adj_matrix=adj_matrix)