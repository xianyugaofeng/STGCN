import torch
import numpy as np
from .base_runner import BaseRunner


class GraphWaveNetRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def sym_adj(adj_matrix):
        # 对称归一化邻接矩阵
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        # 对每一行求和，得到度向量d，d[i]是节点i的度
        d = adj_matrix.sum(dim=1)
        # 度矩阵的-1/2次方，加ε防止除零
        d_sqrt_inv = torch.sqrt(1 / (d + 1e-8))
        # 转为对角矩阵 D^{-1/2}
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        # 对称归一化：D^{-1/2} * A * D^{-1/2}
        sym_adj = torch.matmul(torch.matmul(d_sqrt_inv, adj_matrix), d_sqrt_inv)

        return sym_adj

    @staticmethod
    def asym_adj(adj_matrix):
        # 非对称归一化邻接矩阵
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        # 对邻接矩阵按行求和，得到每个节点的度
        d = adj_matrix.sum(dim=1)
        # 对每个节点的度取-1次方
        d_inv = 1.0 / (d + 1e-8)
        # 构造对角矩阵D^{-1}
        d_inv = torch.diag(d_inv)
        # 非对称归一化: D^{-1} * A
        asym_adj = torch.matmul(d_inv, adj_matrix)

        return asym_adj

    @staticmethod
    def compute_normalized_laplacian(adj_matrix):
        # 归一化拉普拉斯矩阵 L = I - D^{-1/2} * A * D^{-1/2}
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        # 获取节点数
        n = adj_matrix.shape[0]
        # 对邻接矩阵按行求和，得到每个节点的度
        d = adj_matrix.sum(dim=1)
        # 对每个节点的度取-1/2次方
        d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
        # 构造对角矩阵D^{-1/2}
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        # 非对称归一化：D^{-1/2} * A * D^{-1/2}
        sym_adj = torch.matmul(torch.matmul(d_sqrt_inv, adj_matrix), d_sqrt_inv)
        # 归一化拉普拉斯矩阵：I - D^{-1/2} * A * D^{-1/2}
        laplacian = torch.eye(n, device=adj_matrix.device) - sym_adj

        return laplacian
    
    @staticmethod
    def compute_scaled_laplacian(adj_matrix):
        # 计算缩放拉普拉斯矩阵，用于切比雪夫多项式图卷积
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        # 确保邻接矩阵对称，取逐元素最大值
        adj_matrix = torch.maximum(adj_matrix, adj_matrix.T)
        # 获取节点数
        n = adj_matrix.shape[0]
        # 归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
        d = adj_matrix.sum(dim=1)
        d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        sym_adj = d_sqrt_inv @ adj_matrix @ d_sqrt_inv
        laplacian = torch.eye(n, device=adj_matrix.device) - sym_adj
        # 特征值分解获取 lambda_max
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        lambda_max = eigenvalues[-1].item()
        # 缩放到 [-1, 1]
        L_scaled = (2.0 / lambda_max) * laplacian - torch.eye(n, device=adj_matrix.device)

        return L_scaled

    def _get_model_kwargs(self):
        adj_matrix = self.adj_matrix  # 直接是 (N, N) numpy array
        if adj_matrix is None:
            return {}  # 无邻接矩阵，模型用自适应邻接
        
        # 从矩阵索引生成 sensor_ids 和 mapping
        n = adj_matrix.shape[0]
        sensor_ids = list(range(n))
        sensors_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
        
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
            # 构造单位矩阵作为邻接矩阵，每个节点只与自身相连
            adj = [torch.eye(adj_matrix.shape[0], dtype=torch.float32, device=self.device)]
        else:
            self.logger.info(f"adj_type '{adj_type}' not defined, using identity")
            adj = [torch.eye(adj_matrix.shape[0], dtype=torch.float32, device=self.device)]
        
        supports = [torch.tensor(i, device=self.device) for i in adj]
        # 将supports注册为模型的buffer，确保随模型移动到GPU/CPU
        if hasattr(self.model, 'supports'):
            self.model.supports = supports
        else:
            self.model.register_buffer('supports', torch.stack(supports) if len(supports) > 0 else torch.empty(0))
        
        return {}