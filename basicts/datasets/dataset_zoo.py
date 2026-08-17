import os
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from basicts.utils.data_utils import Normalizer

class PEMSDataset(Dataset):
    # 原始数据(总时长, 节点数, 特征数)按固定长度切分成监督学习样本
    # PEMS Dataset for Traffic Forecasting
    def __init__(self, data, input_length=12, output_length=12, mode='train'):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        self.num_samples = data.shape[0] - input_length - output_length + 1
        # 总时间步T_total中，用长度为input_length+output_length的窗口滑动，能切出的样本数
        # Precompute indices
        self.indices = [(i, i + input_length, i + input_length + output_length) 
                       for i in range(self.num_samples)]
        # 每个元组(start, mid, end) start输入起始位置 mid输入结束位置 end输出结束位置

    def __len__(self):
        return self.num_samples
        # 返回样本总数num_samples 供DataLoader使用
    
    def __getitem__(self, idx):
        start, mid, end = self.indices[idx] # 第idx个样本的输入和目标
        x = self.data[start:mid]  # (input_length, num_nodes, num_features)
        y = self.data[mid:end]    # (output_length, num_nodes, num_features)
        return x, y

def load_pems_data(data_file_path, adj_file_path=None, max_train_samples=None, 
                   max_val_samples=None, max_test_samples=None, smoke_test_mode=False,
                   normalize=True, train_ratio=0.6, val_ratio=0.2):
    # Load PEMS dataset
    print(f"[INFO] Loading data from {data_file_path}")
    
    # Load data
    data = np.load(data_file_path)['data']  # (num_timesteps, num_nodes, num_features)
    
    # Load adjacency matrix if available
    adj_matrix = None
    if adj_file_path and os.path.exists(adj_file_path):
        print(f"[INFO] Loading adjacency matrix from {adj_file_path}")
        with open(adj_file_path, 'rb') as f:
            adj_matrix = pickle.load(f) # 直接通过pickle.load反序列化得到adj_matrix，通常为(N, N)的矩阵
    else:
        print(f"[WARN] Adjacency matrix file not found: {adj_file_path}")
        # CSV兜底已移交各Processor的create_adjacency_from_csv hook处理
    
    # Train/Val/Test split (70%/15%/15%)
    num_timesteps = data.shape[0] # 总时间步数
    train_end = int(num_timesteps * train_ratio) # 前60%作为训练集
    val_end = train_end + int(num_timesteps * val_ratio) # 接着20%作为验证集, 20%为测试集

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Apply smoke test limits
    if smoke_test_mode:
        if max_train_samples:
            train_data = train_data[:max_train_samples + 12 + 12]
            # 为滑动窗口留足余量 保证PEMSDataset能够切出至少100个样本
        if max_val_samples:
            val_data = val_data[:max_val_samples + 12 + 12]
        if max_test_samples:
            test_data = test_data[:max_test_samples + 12 + 12]
        # 形状均为(子集长度, num_nodes, num_features)
    
    # Z-score归一化（仅使用训练集统计量）
    normalizer = None
    if normalize:
        normalizer = Normalizer()
        normalizer.fit(train_data)
        train_data = normalizer.transform(train_data)
        val_data = normalizer.transform(val_data)
        test_data = normalizer.transform(test_data)
        print(f"[INFO] Data normalized using Z-score")

    print(f"[INFO] Data loaded: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
    
    return train_data, val_data, test_data, adj_matrix, normalizer

class STIDDataset(Dataset):
    # STID专用数据集:在(x, y)基础上附加时间特征(time_of_day, day_of_week)
    def __init__(self, data, input_length=12, output_length=12,
                 mode='train', steps_per_day=288, add_time_of_day=True, add_day_of_week=True):
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        self.steps_per_day = steps_per_day
        self.data = self.add_temporal_features(data, add_time_of_day, add_day_of_week, steps_per_day)
        self.num_samples = self.data.shape[0] - input_length - output_length + 1
        self.indices = [(i, i + input_length, i + input_length + output_length)
                        for i in range(self.num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start, mid, end = self.indices[idx]
        x = self.data[start:mid]
        y = self.data[mid:end]
        return x, y
    
    @staticmethod
    def add_temporal_features(data, add_time_of_day, add_day_of_week, steps_per_day=288):
        T, N, C = data.shape
        feature_list = [data]
        if add_time_of_day:
            time_of_day = np.array([(i % steps_per_day / steps_per_day) for i in range(T)])
            time_of_day_tiled = np.tile(time_of_day, [1, N, 1]).transpose(2, 1, 0)
            feature_list.append(time_of_day_tiled)
        
        if add_day_of_week:
            day_of_week = np.array([(i // steps_per_day) % 7 / 7 for i in range(T)])
            day_of_week_tiled = np.tile(day_of_week, [1, N, 1]).transpose(2, 1, 0)
            feature_list.append(day_of_week_tiled)
        
        data_with_features = np.concatenate(feature_list, axis=-1)
        return data_with_features

DATASET_ZOO = {
    'PEMS': PEMSDataset,
    'STGCN': PEMSDataset,
    'STID': STIDDataset
}

def get_dataset(dataset_name):
    if dataset_name not in DATASET_ZOO:
        raise ValueError(f"Data set {dataset_name} not found in DATASET_ZOO. "
                          f"Available: {list(DATASET_ZOO.keys())}")
    return DATASET_ZOO[dataset_name]