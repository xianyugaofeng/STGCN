import torch
import numpy as np
from .base_runner import BaseRunner

class STIDRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
