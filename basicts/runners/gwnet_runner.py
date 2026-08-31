import torch
import numpy as np
from .base_runner import BaseRunner


class GraphWaveNetRunner(BaseRunner):
    def __init__(self, config, adj_matrix=None):
        super().__init__(config, adj_matrix=adj_matrix)

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

    @staticmethod
    def _extract_adj_matrices(adj_matrix):
        # pkl格式多样: 单矩阵 / GWNet三元组(sensor_ids, id_to_ind, adj_mx) / 矩阵列表 / dict
        # 统一提取所有二维方阵 忽略ID列表、索引字典等非矩阵元素
        if adj_matrix is None:
            return []
        if isinstance(adj_matrix, np.ndarray):
            return [adj_matrix] if adj_matrix.ndim == 2 else []
        if isinstance(adj_matrix, dict):
            for key in ('adj', 'adj_mx', 'matrix'):
                if key in adj_matrix:
                    return GraphWaveNetRunner._extract_adj_matrices(adj_matrix[key])
            return []
        if isinstance(adj_matrix, (list, tuple)):
            if len(adj_matrix) >= 2 and isinstance(adj_matrix[0], (list, tuple, np.ndarray)) \
                    and isinstance(adj_matrix[1], (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(adj_matrix, dtype=np.float32)
                except (TypeError, ValueError):
                    arr = None
                if arr is not None and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    return [arr]
            matrices = []
            for m in adj_matrix:
                matrices.extend(GraphWaveNetRunner._extract_adj_matrices(m))
            return matrices
        return []

    def generate_supports(self):
        adj_matrix = getattr(self, 'adj_matrix', None)
        if adj_matrix is None:
            return []
        
        # PEMS-BAY的pkl为GWNet格式三元组(sensor_ids, id_to_ind, adj_mx) 统一提取出二维矩阵
        matrices = self._extract_adj_matrices(adj_matrix)
        if not matrices:
            self.logger.info(f"No valid 2-D adjacency in pkl (type={type(adj_matrix)}), fallback to identity")
            n = self.config.get('NUM_NODES', 325)
            return [torch.eye(n, dtype=torch.float32, device=self.device)]
        self.logger.info(f"Extracted {len(matrices)} adjacency matrix from pkl")
        
        adj_type = self.config.get('ADJ_TYPE', 'sym_adj')
        if adj_type == 'sym_adj':
            adj = [GraphWaveNetRunner.sym_adj(m) for m in matrices]
        elif adj_type == 'transition':
            adj = [GraphWaveNetRunner.asym_adj(m) for m in matrices]
        elif adj_type == 'scalap':
            adj = [GraphWaveNetRunner.compute_scaled_laplacian(m) for m in matrices]
        elif adj_type == 'normalap':
            adj = [GraphWaveNetRunner.compute_normalized_laplacian(m) for m in matrices]
        elif adj_type == 'doubletransition':
            adj = [GraphWaveNetRunner.asym_adj(m) for m in matrices] + \
                  [GraphWaveNetRunner.asym_adj(m.T) for m in matrices]
        elif adj_type == 'identity':
            adj = [torch.eye(m.shape[0], dtype=torch.float32, device=self.device) for m in matrices]
        else:
            self.logger.info(f"adj_type '{adj_type}' not defined, using identity")
            adj = [torch.eye(m.shape[0], dtype=torch.float32, device=self.device) for m in matrices]
        
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
        model_args['input_dim'] = self.config.get('NUM_FEATURES', 3)  # input_dim, not num_features
        model_args['input_length'] = self.config.get('INPUT_LENGTH', 12)
        model_args['output_length'] = self.config.get('OUTPUT_LENGTH', 12)
        
        # supports必须在构造模型前生成 GCN的MLP通道数在__init__中按len(supports)确定
        supports = self.generate_supports()
        if supports:
            model_args['supports'] = supports
        
        model = get_model(model_name)(**model_args)
        model = model.to(self.device)
        
        log_args = dict(model_args)
        if supports:
            log_args['supports'] = f"list of {len(supports)} matrices"
        self.logger.info(f"Model: {self.config.get('MODEL_NAME', 'GraphWaveNet')}")
        self.logger.info(f"Model args: {log_args}")
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        
        self.model = model
        return model