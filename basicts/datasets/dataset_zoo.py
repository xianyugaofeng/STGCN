import os
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader

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
            adj_matrix = pickle.load(f) # 直接通过pickle.load反序列化得到adj_matrix，通常为 (N, N)的矩阵
    else:
        print(f"[WARN] Adjacency matrix file not found: {adj_file_path}")
        # Create simple adjacency matrix based on distance thresholds if CSV available
        csv_path = data_file_path.replace('.npz', '.csv')
        if os.path.exists(csv_path):
            print(f"[INFO] Creating adjacency matrix from CSV: {csv_path}")
            adj_matrix = create_adjacency_from_csv(csv_path, data.shape[1])
            # 读取传感器经纬度；计算传感器间距离；利用阈值或高斯核生成邻接矩阵
    
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

def create_adjacency_from_csv(csv_path, num_nodes=None, symmetric=True,
                              default_diag=1.0, threshold=0.1,
                              source_col='from', target_col='to', weight_col='cost'):
    # Create adjacency matrix from sensor distance CSV
    # 距离阈值归一化后，距离小于该值的两个节点视为相邻
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] 无法读取 CSV 文件: {csv_path}, 错误: {e}")
        n = num_nodes if num_nodes is not None else 1
        return np.eye(n)
    
    df.columns = df.columns.str.strip() # 去除首尾空格
    if weight_col not in df.columns:
        for alt in ['weight', 'w', 'cost', 'distance', 'length']:
            if alt in df.columns:
                weight_col = alt
                break
        else:
            # 未找到，添加全1列
            df[weight_col] = 1.0

    if source_col not in df.columns:
        for alt in ['source', 'src', 'from', 'node_from', 'start']:
            if alt in df.columns:
                source_col = alt
                break
        else:
            raise ValueError(f"CSV 中缺少源节点列（尝试过 {source_col} 及常见别名）")

    if target_col not in df.columns:
        for alt in ['target', 'dst', 'to', 'node_to', 'end']:
            if alt in df.columns:
                target_col = alt
                break
        else:
            raise ValueError(f"缺少目标节点列（已尝试 {target_col} 及常见别名）")

    if num_nodes is None:
        raise ValueError(f"邻接矩阵缺失节点参数")
    
    distances = df[weight_col].values.astype(np.float32)
    sigma = 0.5 * np.std(distances)
    print(f"[INFO] 自动设置高斯核 sigma = {sigma:.4f}")
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weights[weights < threshold] = 0.0

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for idx, row in df.iterrows():
        u = int(row[source_col])
        v = int(row[target_col])
        w = weights[idx]
        if u == v or w == 0.0:
            continue
        adj_matrix[u, v] = max(adj_matrix[u, v], w)

    if symmetric:
        # 对称化：取两者较大值
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        
    np.fill_diagonal(adj_matrix, default_diag)
    return adj_matrix

def build_dataloader(config, mode='train'):
   # Build DataLoader based on config
    data_file_path = config.get('DATA_FILE_PATH', 'STGCN_data/PEMS04/PEMS04.npz')
    adj_file_path = config.get('ADJ_FILE_PATH', None)
    
    input_length = config.get('INPUT_LENGTH', 12)
    output_length = config.get('OUTPUT_LENGTH', 12)
    batch_size = config.get(f'{mode.upper()}_BATCH_SIZE', 32)
    num_workers = config.get('NUM_WORKERS', 2)
    smoke_test_mode = config.get('SMOKE_TEST_MODE', False)
    normalize = config.get('NORMALIZE', True)
    train_ratio = config.get('TRAIN_RATIO', 0.6)
    val_ratio = config.get('VAL_RATIO', 0.2)

    if mode == 'train':
        train_data, val_data, test_data, adj_matrix, normalizer = load_pems_data(
            data_file_path, adj_file_path,
            max_train_samples=config.get('MAX_TRAIN_SAMPLES'),
            smoke_test_mode=smoke_test_mode,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        dataset = PEMSDataset(train_data, input_length, output_length, mode='train')
    elif mode == 'val':
        # Reload to get validation data
        train_data, val_data, test_data, adj_matrix, normalizer = load_pems_data(
            data_file_path, adj_file_path,
            max_val_samples=config.get('MAX_VAL_SAMPLES'),
            smoke_test_mode=smoke_test_mode,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        dataset = PEMSDataset(val_data, input_length, output_length, mode='val')
    elif mode == 'test':
        train_data, val_data, test_data, adj_matrix, normalizer = load_pems_data(
            data_file_path, adj_file_path,
            max_test_samples=config.get('MAX_TEST_SAMPLES'),
            smoke_test_mode=smoke_test_mode,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        dataset = PEMSDataset(test_data, input_length, output_length, mode='test')
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'), # 只在训练时打乱数据顺序
        num_workers=num_workers, # 提升数据到GPU的传输效率
        pin_memory=True # 多进程加载数据
    )
    
    return dataloader, adj_matrix, normalizer