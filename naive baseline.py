import os
import sys
import numpy as np
import torch
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicts.datasets.dataset_zoo import load_pems_data
from basicts.metrics.metric_zoo import compute_metrics

class BaselineModels:
    # Naive Baseline Models for Traffic Forecasting
    
    @staticmethod
    def last_value(x):
        # 使用输入序列的最后一个时间步作为预测
        # x: (batch, input_length, num_nodes, num_features)
        # returns: (batch, output_length, num_nodes, num_features)

        # 取最后一个时间步
        last_step = x[:, -1:, :, :]  # (batch, 1, num_nodes, num_features)
        # 重复 output_length 次
        output_length = x.shape[1]  # 使用input_length作为output_length
        return last_step.repeat(1, output_length, 1, 1)
    
    @staticmethod
    def mean_value(x):
        # 使用输入序列所有时间步的平均值作为预测
        # x: (batch, input_length, num_nodes, num_features)
        # returns: (batch, output_length, num_nodes, num_features)
        
        # 沿时间轴求平均
        mean_step = x.mean(dim=1, keepdim=True)  # (batch, 1, num_nodes, num_features)
        output_length = x.shape[1]
        return mean_step.repeat(1, output_length, 1, 1)
    
    @staticmethod
    def median_value(x):
        # 使用输入序列所有时间步的中位数作为预测
        median_step = torch.median(x, dim=1, keepdim=True).values
        output_length = x.shape[1]
        return median_step.repeat(1, output_length, 1, 1)
    
    @staticmethod
    def moving_average(x, window=3):
        # 使用滑动窗口平均值作为预测
        input_length = x.shape[1]
        start_idx = max(0, input_length - window)
        ma_step = x[:, start_idx:, :, :].mean(dim=1, keepdim=True)
        # 取输入序列最后window步的特征平均值
        output_length = input_length
        return ma_step.repeat(1, output_length, 1, 1)
    
    @staticmethod
    def exponential_smoothing(x, alpha=0.3):
        # 指数平滑预测
        input_length = x.shape[1]
        # 从后往前计算指数平滑
        smoothed = x[:, -1, :, :].unsqueeze(1)  # 初始值为最后一个时间步 用unsqueeze(1)恢复时间维度
        for i in range(input_length - 2, -1, -1): # 从倒数第二个时间步往前遍历
            smoothed = alpha * x[:, i:i+1, :, :] + (1 - alpha) * smoothed
        output_length = input_length
        return smoothed.repeat(1, output_length, 1, 1)

def create_sliding_window_samples(data, input_length, output_length):
    # 创建滑动窗口样本 与PEMSDataset相同的方式
    num_samples = data.shape[0] - input_length - output_length + 1
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = data[i:i+input_length]
        y = data[i+input_length:i+input_length+output_length]
        x_list.append(x)
        y_list.append(y)
    return np.array(x_list), np.array(y_list)
    # returns: x (num_samples, input_length, ...)
    #          y (num_sampels, output_length, ...)

def evaluate_baseline(config):
    # 评估所有baseline模型
    print("=" * 70)
    print("Naive Baseline Evaluation for Traffic Forecasting")
    print("=" * 70)
    
    # 加载数据（使用与模型相同的方式）
    data_file_path = config.get('DATA_FILE_PATH', 'STGCN_data/PEMS04/PEMS04.npz')
    adj_file_path = config.get('ADJ_FILE_PATH', None)
    
    input_length = config.get('INPUT_LENGTH', 12)
    output_length = config.get('OUTPUT_LENGTH', 12)
    
    print(f"\n[INFO] Loading data from {data_file_path}")
    train_data, val_data, test_data, adj_matrix = load_pems_data(
        data_file_path, adj_file_path,
        max_train_samples=config.get('MAX_TRAIN_SAMPLES'),
        max_val_samples=config.get('MAX_VAL_SAMPLES'),
        max_test_samples=config.get('MAX_TEST_SAMPLES')
    )
    
    print(f"[INFO] Data shapes: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
    
    # 创建测试样本
    print(f"[INFO] Creating sliding window samples...")
    test_x, test_y = create_sliding_window_samples(test_data, input_length, output_length)
    print(f"[INFO] Test samples: x={test_x.shape}, y={test_y.shape}")
    
    # 转换为torch张量
    test_x_tensor = torch.from_numpy(test_x).float()
    test_y_tensor = torch.from_numpy(test_y).float()
    
    # 定义baseline模型
    baselines = {
        'Last Value': BaselineModels.last_value,
        'Mean Value': BaselineModels.mean_value,
        'Median Value': BaselineModels.median_value,
        'Moving Average (3)': lambda x: BaselineModels.moving_average(x, window=3),
        'Moving Average (6)': lambda x: BaselineModels.moving_average(x, window=6),
        'Moving Average (12)': lambda x: BaselineModels.moving_average(x, window=12),
        'Exponential Smoothing': BaselineModels.exponential_smoothing,
    }
    
    # 评估每个baseline
    results = {}
    metric_names = config.get('METRICS', ['MAE', 'RMSE', 'MAPE'])
    
    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
    print("-" * 70)
    
    for name, model_func in baselines.items():
        pred = model_func(test_x_tensor)
        metrics = compute_metrics(pred, test_y_tensor, metric_names)
        results[name] = metrics
        
        print(f"{name:<25} {metrics['MAE']:>12.4f} {metrics['RMSE']:>12.4f} {metrics['MAPE']:>10.2f}%")
    
    print("-" * 70)
    
    # 对比STGCN模型结果
    print("\n[INFO] STGCN Model Best Validation Metrics (from training):")
    with open('outputs/STGCN_PEMS04/best_val_metrics.json', 'r', encoding='utf-8') as f:
        best_val_metrics = json.load(f)

    stgcn_results = {
        'MAE': best_val_metrics['MAE'],
        'RMSE': best_val_metrics['RMSE'],
        'MAPE': best_val_metrics['MAPE']  # 这个MAPE异常大，可能是计算问题
    }
    print(f"{'STGCN':<25} {stgcn_results['MAE']:>12.4f} {stgcn_results['RMSE']:>12.4f} {stgcn_results['MAPE']:>10.2f}%")
    
    # 找出最佳baseline
    best_model = min(results.keys(), key=lambda k: results[k]['MAE'])
    print(f"\n[INFO] Best Baseline: {best_model}")
    print(f"[INFO] Best Baseline Metrics: MAE={results[best_model]['MAE']:.4f}, "
          f"RMSE={results[best_model]['RMSE']:.4f}, MAPE={results[best_model]['MAPE']:.2f}%")
    
    # 保存结果
    output_dir = config.get('LOG_DIR', 'outputs/STGCN_PEMS04')
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_results = {
        'config': config,
        'baselines': results,
        'stgcn': stgcn_results,
        'best_baseline': best_model,
        'data_info': {
            'train_shape': train_data.shape,
            'val_shape': val_data.shape,
            'test_shape': test_data.shape,
            'input_length': input_length,
            'output_length': output_length,
            'num_nodes': train_data.shape[1],
            'num_features': train_data.shape[2]
        }
    }
    
    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"\n[INFO] Results saved to {os.path.join(output_dir, 'baseline_results.json')}")
    
    return results

if __name__ == '__main__':
    # 使用默认配置或从命令行读取
    config = {
        'DATA_FILE_PATH': 'STGCN_data/PEMS04/PEMS04.npz',
        'ADJ_FILE_PATH': 'STGCN_data/PEMS04/adj_PEMS04.pkl',
        'INPUT_LENGTH': 12,
        'OUTPUT_LENGTH': 12,
        'METRICS': ['MAE', 'RMSE', 'MAPE'],
        'LOG_DIR': 'outputs/STGCN_PEMS04',
        'MAX_TRAIN_SAMPLES': None,
        'MAX_VAL_SAMPLES': None,
        'MAX_TEST_SAMPLES': None
    }
    
    evaluate_baseline(config)