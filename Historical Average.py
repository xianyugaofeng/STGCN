import os
import sys
import numpy as np
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicts.datasets.dataset_zoo import load_pems_data
from basicts.metrics.metric_zoo import compute_metrics

# 常见参数
TIME_STEPS_PER_DAY = 288
HISTORY_WEEKS = 4
WEEKLY_PERIOD = TIME_STEPS_PER_DAY * 7  # 一周的步数 (5分钟间隔: 288 * 7 = 2016)

def create_sliding_window_samples(data, input_length, output_length):
    # 创建滑动窗口样本，与PEMSDataset相同的方式
    num_samples = data.shape[0] - input_length - output_length + 1
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = data[i:i+input_length]
        y = data[i+input_length:i+input_length+output_length]
        x_list.append(x)
        y_list.append(y)
    return np.array(x_list), np.array(y_list)

class HistoricalAverageModel:
    # Historical Average Baseline Model for Traffic Forecasting
    # 基于历史同期平均值进行预测，利用交通数据的周周期性

    def __init__(self, history_weeks=4, period_steps=WEEKLY_PERIOD):
        self.period_steps = period_steps  # 周期步数（一周）
        self.history_weeks = history_weeks

    def fit(self, train_data):
        # HA模型存储训练数据用于历史查询
        self.train_data = train_data
        self.train_size = train_data.shape[0]
        self.num_nodes = train_data.shape[1]
        self.num_features = train_data.shape[2] if len(train_data.shape) > 2 else 1
        print(f"[INFO] HA模型初始化完成，使用过去{self.history_weeks}周的数据进行历史平均")

    def predict(self, test_data_start_index, num_samples, input_length, output_length):
        # 对测试集进行多步预测
        # test_data_start_index: 测试集在原始数据中的绝对起始索引
        # returns: (num_samples, output_length, num_nodes, num_features)
        predictions = np.zeros((num_samples, output_length, self.num_nodes, self.num_features))
        print(f"[INFO] 正在进行Historical Average预测...")

        # 预计算所有唯一目标时间的HA预测值（向量化优化）
        target_start = test_data_start_index + input_length
        target_end = test_data_start_index + num_samples + input_length + output_length - 1
        all_target_times = np.arange(target_start, target_end)
        ha_lookup = np.zeros((len(all_target_times), self.num_nodes, self.num_features))

        for i, t in enumerate(all_target_times):
            history_indices = []
            for w in range(1, self.history_weeks + 1):
                hist_idx = t - w * self.period_steps
                if 0 <= hist_idx < self.train_size:
                    history_indices.append(hist_idx)
            if len(history_indices) > 0:
                ha_lookup[i] = np.mean(self.train_data[history_indices], axis=0)
            else:
                ha_lookup[i] = self.train_data[-1]

        # 从查找表中提取每个样本的预测
        for s in range(num_samples):
            for j in range(output_length):
                predictions[s, j] = ha_lookup[s + j]

        print(f"[INFO] HA预测完成 预测样本数: {num_samples}")
        return predictions

def evaluate_historical_average(config):
    # 评估Historical Average baseline模型
    print("=" * 70)
    print("Historical Average Baseline Evaluation for Traffic Forecasting")
    print("=" * 70)

    data_file_path = config.get('DATA_FILE_PATH', 'STGCN_data/PEMS04/PEMS04.npz')
    adj_file_path = config.get('ADJ_FILE_PATH', None)
    input_length = config.get('INPUT_LENGTH', 12)
    output_length = config.get('OUTPUT_LENGTH', 12)
    history_weeks = config.get('HISTORY_WEEKS', HISTORY_WEEKS)

    print(f"\n[INFO] Loading data from {data_file_path}")
    train_data, val_data, test_data, adj_matrix, normalizer = load_pems_data(
        data_file_path, adj_file_path,
        max_train_samples=config.get('MAX_TRAIN_SAMPLES'),
        max_val_samples=config.get('MAX_VAL_SAMPLES'),
        max_test_samples=config.get('MAX_TEST_SAMPLES'),
        normalize=True
    )
    print(f"[INFO] Data shapes: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")

    print(f"[INFO] Creating sliding window samples...")
    test_x, test_y = create_sliding_window_samples(test_data, input_length, output_length)
    print(f"[INFO] Test samples: x={test_x.shape}, y={test_y.shape}")

    model = HistoricalAverageModel(history_weeks=history_weeks)
    model.fit(train_data)

    # 测试集在原始数据中的绝对起始索引
    test_data_start_index = len(train_data) + len(val_data)
    num_samples = test_x.shape[0]
    predictions = model.predict(test_data_start_index, num_samples, input_length, output_length)

    # 反归一化到原始尺度
    if normalizer is not None:
        predictions = normalizer.inverse_transform(predictions)
        test_y = normalizer.inverse_transform(test_y)
        print(f"[INFO] Data inverse transformed to original scale")

    pred_tensor = torch.from_numpy(predictions).float()
    target_tensor = torch.from_numpy(test_y).float()
    # 随后将NumPy数组转换为PyTorch张量，方便利用PyTorch的算子计算损失或指标
    metric_names = config.get('METRICS', ['MAE', 'RMSE', 'MAPE'])
    metrics = compute_metrics(pred_tensor, target_tensor, metric_names)

    print("\n" + "-" * 70)
    print(f"{'Model':<30} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
    print("-" * 70)
    print(f"{'Historical Average':<30} {metrics['MAE']:>12.4f} {metrics['RMSE']:>12.4f} {metrics['MAPE']:>10.2f}%")
    print("-" * 70)

    # 对比STGCN模型结果
    stgcn_results = None
    stgcn_metrics_path = config.get('STGCN_METRICS_PATH', 'outputs/smoke_STGCN_PEMS04/best_val_metrics.json')
    if os.path.exists(stgcn_metrics_path):
        print(f"\n[INFO] STGCN Model Best Validation Metrics (from {stgcn_metrics_path}):")
        with open(stgcn_metrics_path, 'r', encoding='utf-8') as f:
            best_val_metrics = json.load(f)
        stgcn_results = {
            'MAE': best_val_metrics['MAE'],
            'RMSE': best_val_metrics['RMSE'],
            'MAPE': best_val_metrics['MAPE']
        }
        print(f"{'STGCN':<30} {stgcn_results['MAE']:>12.4f} {stgcn_results['RMSE']:>12.4f} {stgcn_results['MAPE']:>10.2f}%")
    else:
        print(f"\n[WARN] STGCN metrics file not found: {stgcn_metrics_path}")

    # 保存结果
    output_dir = config.get('LOG_DIR', 'outputs/STGCN_PEMS04')
    os.makedirs(output_dir, exist_ok=True)
    baseline_results = {
        'model': 'Historical Average',
        'config': {'history_weeks': history_weeks, 'period_steps': WEEKLY_PERIOD,
                   'input_length': input_length, 'output_length': output_length},
        'metrics': metrics,
        'stgcn': stgcn_results,
        'data_info': {
            'train_shape': list(train_data.shape), 'val_shape': list(val_data.shape),
            'test_shape': list(test_data.shape), 'input_length': input_length,
            'output_length': output_length, 'num_nodes': train_data.shape[1],
            'num_features': train_data.shape[2] if len(train_data.shape) > 2 else 1
        }
    }
    output_path = os.path.join(output_dir, 'ha_baseline_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Results saved to {output_path}")
    return metrics

if __name__ == '__main__':
    config = {
        'DATA_FILE_PATH': 'STGCN_data/PEMS04/PEMS04.npz',
        'ADJ_FILE_PATH': 'STGCN_data/PEMS04/adj_PEMS04.pkl',
        'INPUT_LENGTH': 12, 'OUTPUT_LENGTH': 12,
        'METRICS': ['MAE', 'RMSE', 'MAPE'],
        'LOG_DIR': 'outputs/STGCN_PEMS04',
        'STGCN_METRICS_PATH': 'outputs/smoke_STGCN_PEMS04/best_val_metrics.json',
        'HISTORY_WEEKS': 4,
        'MAX_TRAIN_SAMPLES': None, 'MAX_VAL_SAMPLES': None, 'MAX_TEST_SAMPLES': None
    }
    evaluate_historical_average(config)