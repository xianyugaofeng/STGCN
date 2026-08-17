import numpy as np
import torch

class Normalizer:
    # Z-score归一化 零均值单位方差
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        # data: (num_timesteps, num_nodes, num_features)
        self.mean = np.mean(data, axis=(0, 1), keepdims=True)  # (1, 1, num_features)
        # 在时间步和节点两个维度上计算统计量，得到的是每个特征独立的均值和标准差
        self.std = np.std(data, axis=(0, 1), keepdims=True)    # (1, 1, num_features)
        # 这样transform中的减法和除法可以自动广播到原数据的(T, N, F)形状
        self.std[self.std < 1e-8] = 1e-8  # 避免除以0 将小于1e-8的标准差替换为1e-8，避免除以零错误
        print(f"[INFO] Normalizer fitted: mean={self.mean.squeeze()}, std={self.std.squeeze()}")
    
    def transform(self, data):
        return (data - self.mean) / self.std
        # 加减乘除都可以依靠numpy的广播机制完成
    
    def inverse_transform(self, data):
        return data * self.std + self.mean
    
    # Z-score标准化后，每个特征的均值为0，标准差为1

def set_random_seed(seed):
    np.random.seed(seed) # NumPy的随机数生成器
    torch.manual_seed(seed) # 固定PyTorch在CPU上的随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # 固定所有GPU上的随机数生成器