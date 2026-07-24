import numpy as np
import torch

def normalize_data(data, method='minmax'):
    if method == 'minmax':
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        return normalized, min_val, max_val
    elif method == 'zscore':
        mean_val = data.mean(axis=0)
        std_val = data.std(axis=0)
        normalized = (data - mean_val) / (std_val + 1e-8)
        return normalized, mean_val, std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def denormalize_data(data, min_val=None, max_val=None, mean_val=None, std_val=None, method='minmax'):
    if method == 'minmax':
        return data * (max_val - min_val + 1e-8) + min_val
    elif method == 'zscore':
        return data * (std_val + 1e-8) + mean_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def set_random_seed(seed):
    np.random.seed(seed) # NumPy的随机数生成器
    torch.manual_seed(seed) # 固定PyTorch在CPU上的随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # 固定所有GPU上的随机数生成器