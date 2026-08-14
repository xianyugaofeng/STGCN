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

class BaseDataProcessor:
    # 数据处理基类 封装加载/切分/归一化/构建DataLoader的通用流程
    # 不同模型通过子类化并重写build_dataset等hook实现差异化数据处理 实现解耦
    def __init__(self, config):
        self.config = config
        self.data_file_path = config.get('DATA_FILE_PATH', 'STGCN_data/PEMS04/PEMS04.npz')
        self.adj_file_path = config.get('ADJ_FILE_PATH', None)
        self.input_length = config.get('INPUT_LENGTH', 12)
        self.output_length = config.get('OUTPUT_LENGTH', 12)
        self.smoke_test_mode = config.get('SMOKE_TEST_MODE', False)
        self.normalize = config.get('NORMALIZE', True)
        self.train_ratio = config.get('TRAIN_RATIO', 0.6)
        self.val_ratio = config.get('VAL_RATIO', 0.2)
        self.steps_per_day = config.get('STEPS_PER_DAY', 288)  # PEMS默认5分钟采样 一天288步

    def load_raw(self, mode):
        # 复用load_pems_data加载并切分原始数据 返回三段数据+邻接矩阵+归一化器
        max_samples_key = {
            'train': 'MAX_TRAIN_SAMPLES',
            'val': 'MAX_VAL_SAMPLES',
            'test': 'MAX_TEST_SAMPLES'
        }[mode]
        return load_pems_data(
            self.data_file_path, self.adj_file_path,
            max_train_samples=self.config.get('MAX_TRAIN_SAMPLES') if mode == 'train' else None,
            max_val_samples=self.config.get('MAX_VAL_SAMPLES') if mode == 'val' else None,
            max_test_samples=self.config.get('MAX_TEST_SAMPLES') if mode == 'test' else None,
            smoke_test_mode=self.smoke_test_mode,
            normalize=self.normalize,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )

    def build_dataset(self, data, mode):
        # hook方法 子类重写以返回模型专属Dataset
        return PEMSDataset(data, self.input_length, self.output_length, mode=mode)

    def build_dataloader(self, mode):
        # 通用流程:加载原始数据->选取对应分段->构建Dataset->构建DataLoader
        train_data, val_data, test_data, adj_matrix, normalizer = self.load_raw(mode)
        data_map = {'train': train_data, 'val': val_data, 'test': test_data}
        if mode not in data_map:
            raise ValueError(f"Unknown mode: {mode}")
        dataset = self.build_dataset(data_map[mode], mode)
        batch_size = self.config.get(f'{mode.upper()}_BATCH_SIZE', 32)
        num_workers = self.config.get('NUM_WORKERS', 2)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),  # 只在训练时打乱数据顺序
            num_workers=num_workers,    # 提升数据到GPU的传输效率
            pin_memory=True            # 多进程加载数据
        )
        return dataloader, adj_matrix, normalizer


class STGCNProcessor(BaseDataProcessor):
    # STGCN数据处理:标准(x, y)格式 无额外时间特征
    pass


class STIDDataset(Dataset):
    # STID专用数据集:在(x, y)基础上附加时间特征(time_of_day, day_of_week)
    # 返回(x, y, tod, dow) 其中tod/dow为输入窗口每个时间步的整数索引 供nn.Embedding使用
    def __init__(self, data, input_length=12, output_length=12, mode='train', steps_per_day=288):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        self.steps_per_day = steps_per_day
        self.num_samples = data.shape[0] - input_length - output_length + 1
        self.indices = [(i, i + input_length, i + input_length + output_length)
                        for i in range(self.num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start, mid, end = self.indices[idx]
        x = self.data[start:mid]
        y = self.data[mid:end]
        # 由输入窗口的全局时间步索引推导time_of_day与day_of_week
        ts = np.arange(start, mid)
        tod = (ts % self.steps_per_day).astype(np.int64)        # 0..steps_per_day-1
        dow = (ts // self.steps_per_day % 7).astype(np.int64)   # 0..6
        return x, y, tod, dow


class STIDProcessor(BaseDataProcessor):
    # STID数据处理:附加时间特征(ToD/DoW) 配合STID模型的embedding输入
    def build_dataset(self, data, mode):
        return STIDDataset(data, self.input_length, self.output_length,
                           mode=mode, steps_per_day=self.steps_per_day)


# 数据处理器注册表:新增模型只需在此注册对应Processor 无需改动build_dataloader
PROCESSOR_ZOO = {
    'Base': BaseDataProcessor,
    'STGCN': STGCNProcessor,
    'STID': STIDProcessor,
}


def get_processor(name):
    # 按名称获取数据处理器类 未注册时给出可用列表
    if name not in PROCESSOR_ZOO:
        raise ValueError(f"Data processor {name} not found in PROCESSOR_ZOO. "
                          f"Available: {list(PROCESSOR_ZOO.keys())}")
    return PROCESSOR_ZOO[name]


def build_dataloader(config, mode='train'):
    # 通过PROCESSOR_ZOO解耦:用MODEL_NAME指定处理器
    processor_name = config.get(config.get('MODEL_NAME', 'Base'))
    processor_cls = get_processor(processor_name)
    processor = processor_cls(config)
    print(f"[INFO] Using data processor: {processor_name} for mode={mode}")
    return processor.build_dataloader(mode)