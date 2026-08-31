import torch
import numpy as np
from .base_runner import BaseRunner


class GraphWaveNetRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def sym_adj(adj_matrix):
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        d = adj_matrix.sum(dim=1)
        d_sqrt_inv = torch.sqrt(1 / (d + 1e-8))
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        sym_adj = torch.matmul(torch.matmul(d_sqrt_inv, adj_matrix), d_sqrt_inv)
        return sym_adj

    @staticmethod
    def asym_adj(adj_matrix):
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        d = adj_matrix.sum(dim=1)
        d_inv = 1.0 / (d + 1e-8)
        d_inv = torch.diag(d_inv)
        asym_adj = torch.matmul(d_inv, adj_matrix)
        return asym_adj

    @staticmethod
    def compute_normalized_laplacian(adj_matrix):
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        n = adj_matrix.shape[0]
        d = adj_matrix.sum(dim=1)
        d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        sym_adj = torch.matmul(torch.matmul(d_sqrt_inv, adj_matrix), d_sqrt_inv)
        laplacian = torch.eye(n, device=adj_matrix.device) - sym_adj
        return laplacian
    
    @staticmethod
    def compute_scaled_laplacian(adj_matrix):
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        adj_matrix = torch.maximum(adj_matrix, adj_matrix.T)
        n = adj_matrix.shape[0]
        d = adj_matrix.sum(dim=1)
        d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        sym_adj = d_sqrt_inv @ adj_matrix @ d_sqrt_inv
        laplacian = torch.eye(n, device=adj_matrix.device) - sym_adj
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        lambda_max = eigenvalues[-1].item()
        L_scaled = (2.0 / lambda_max) * laplacian - torch.eye(n, device=adj_matrix.device)
        return L_scaled

    def generate_supports(self):
        adj_matrix = getattr(self, 'adj_matrix', None)
        if adj_matrix is None:
            return []
        
        n = adj_matrix.shape[0]
        
        adj_type = self.config.get('ADJ_TYPE', 'sym_adj')
        if adj_type == 'sym_adj':
            adj = [GraphWaveNetRunner.sym_adj(adj_matrix)]
        elif adj_type == 'transition':
            adj = [GraphWaveNetRunner.asym_adj(adj_matrix)]
        elif adj_type == 'scalap':
            adj = [GraphWaveNetRunner.compute_scaled_laplacian(adj_matrix)]
        elif adj_type == 'normalap':
            adj = [GraphWaveNetRunner.compute_normalized_laplacian(adj_matrix)]
        elif adj_type == 'doubletransition':
            adj = [GraphWaveNetRunner.asym_adj(adj_matrix), GraphWaveNetRunner.asym_adj(adj_matrix.T)]
        elif adj_type == 'identity':
            adj = [torch.eye(adj_matrix.shape[0], dtype=torch.float32, device=self.device)]
        else:
            self.logger.info(f"adj_type '{adj_type}' not defined, using identity")
            adj = [torch.eye(adj_matrix.shape[0], dtype=torch.float32, device=self.device)]
        
        # adj elements are already tensors, just move to device
        supports = [i.to(self.device) if isinstance(i, torch.Tensor) else torch.tensor(i, device=self.device) for i in adj]
        return supports

    def _init_model(self):
        # Override to use GraphWaveNet-specific args (input_dim instead of num_features)
        from basicts.models import get_model
        
        model_name = self.config.get('MODEL_NAME', 'GraphWaveNet')
        model_args = self.config.get('MODEL_ARGS', {}).copy()
        
        # GraphWaveNet specific args from config
        model_args['num_nodes'] = self.config.get('NUM_NODES', 325)
        model_args['input_dim'] = self.config.get('NUM_FEATURES', 2)  # input_dim, not num_features
        model_args['input_length'] = self.config.get('INPUT_LENGTH', 12)
        model_args['output_length'] = self.config.get('OUTPUT_LENGTH', 12)
        
        model = get_model(model_name)(**model_args)
        model = model.to(self.device)
        
        if 'supports' not in model_args:
            model.supports = self.generate_supports()
        
        self.logger.info(f"Model: {self.config.get('MODEL_NAME', 'GraphWaveNet')}")
        self.logger.info(f"Model args: {model_args}")
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        
        self.model = model
        return model