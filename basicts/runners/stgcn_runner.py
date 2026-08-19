import torch
import numpy as np
from .base_runner import BaseRunner

class STGCNRunner(BaseRunner):
    # STGCN专用的Runner
    # 负责计算和管理拉普拉斯矩阵，这是图卷积网络特有的需求
    # 通过重写_get_model_kwargs hook向模型传递laplacian参数

    def __init__(self, config):
        super().__init__(config)
        self._cached_laplacian = None  # 拉普拉斯矩阵缓存

    def _compute_laplacian(self, adj_matrix):
        # 将原始邻接矩阵转换为对称归一化拉普拉斯矩阵，供图卷积层使用
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()

        d = adj_matrix.sum(dim=1)
        # 对每一行求和，得到度向量d，d[i]是节点i的度
        d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
        # 度矩阵的-1/2次方，加ε防止除零
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        # 转为对角矩阵 D^{-1/2}
        laplacian = torch.matmul(torch.matmul(d_sqrt_inv, adj_matrix), d_sqrt_inv)
        # 对称归一化：D^{-1/2} * A * D^{-1/2}
        return laplacian

    def _get_model_kwargs(self):
        # 重写hook：向STGCN模型传递拉普拉斯矩阵
        # 使用缓存避免每个batch重复计算
        if self.adj_matrix is None:
            return {}
        if self._cached_laplacian is None:
            self._cached_laplacian = self._compute_laplacian(self.adj_matrix).to(self.device)
            self.logger.info(f"Laplacian matrix computed and cached: {self._cached_laplacian.shape}")
        return {'laplacian': self._cached_laplacian}