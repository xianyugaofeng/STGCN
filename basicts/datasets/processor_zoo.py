import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from basicts.datasets.dataset_zoo import get_dataset, load_pems_data

class BaseDataProcessor:
    # 数据处理基类 封装加载/切分/归一化/构建DataLoader的通用流程
    # 不同模型通过子类化并重写build_dataset等hook实现差异化数据处理 实现解耦
    def __init__(self, config):
        self.config = config
        self.data_file_path = config.get('DATA_FILE_PATH', 'PEMSdata/PEMS04/PEMS04.npz')
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
        train_data, val_data, test_data, adj_matrix, normalizer = load_pems_data(
            self.data_file_path, self.adj_file_path,
            max_train_samples=self.config.get('MAX_TRAIN_SAMPLES') if mode == 'train' else None,
            max_val_samples=self.config.get('MAX_VAL_SAMPLES') if mode == 'val' else None,
            max_test_samples=self.config.get('MAX_TEST_SAMPLES') if mode == 'test' else None,
            smoke_test_mode=self.smoke_test_mode,
            normalize=self.normalize,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )
        # 无pickle邻接矩阵时，由各Processor的hook负责从CSV兜底生成
        if adj_matrix is None:
            csv_path = self.data_file_path.replace('.npz', '.csv')
            if os.path.exists(csv_path):
                print(f"[INFO] Creating adjacency matrix from CSV: {csv_path}")
                adj_matrix = self.create_adjacency_from_csv(csv_path, train_data.shape[1])
        return train_data, val_data, test_data, adj_matrix, normalizer

    def create_adjacency_from_csv(self, csv_path, num_nodes=None, symmetric=True,
                                  default_diag=1.0, threshold=0.1,
                                  source_col='from', target_col='to', weight_col='cost'):
        # hook方法由传感器距离CSV生成邻接矩阵 默认实现为高斯核子类可覆写
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN]无法读取CSV文件: {csv_path}, 错误: {e}")
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

        # 利用高斯核计算 将原始的距离转换为相似度/连接强度
        distances = df[weight_col].values.astype(np.float32)
        sigma = 0.5 * np.std(distances)
        print(f"[INFO] 自动设置高斯核 sigma = {sigma:.4f}")
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights[weights < threshold] = 0.0

        # 向量化操作创建邻接矩阵
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        u_arr = df[source_col].values.astype(np.int64)
        v_arr = df[target_col].values.astype(np.int64)
        mask = (u_arr != v_arr) & (weights > 0)
        np.maximum.at(adj_matrix, (u_arr[mask], v_arr[mask]), weights[mask])

        if symmetric:
            # 对称化：取两者较大值
            adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

        np.fill_diagonal(adj_matrix, default_diag)
        return adj_matrix

    def build_dataset(self, data, mode):
        # hook方法 子类重写以返回模型专属Dataset
        model_name = self.config.get('MODEL_NAME')
        return get_dataset(model_name)(data, self.input_length, self.output_length, mode=mode)

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
class STIDProcessor(BaseDataProcessor):
    # STID数据处理:附加时间特征(ToD/DoW) 配合STID模型的embedding输入
    def build_dataset(self, data, mode):
        add_time_of_day = self.config.get('ADD_TIME_OF_DAY', True)
        add_day_of_week = self.config.get('ADD_DAY_OF_WEEK', True)
        return get_dataset('STID')(data, self.input_length, self.output_length,
                           mode=mode, steps_per_day=self.steps_per_day, add_time_of_day=add_time_of_day, add_day_of_week=add_day_of_week)

    def create_adjacency_from_csv(self, csv_path, num_nodes=None, **kwargs):
        # STID为MLP模型 无需图结构 覆写hook不生成邻接矩阵
        return None

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
    processor_name = config.get('MODEL_NAME', 'Base')
    processor_cls = get_processor(processor_name)
    processor = processor_cls(config)
    print(f"[INFO] Using data processor: {processor_name} for mode={mode}")
    return processor.build_dataloader(mode)