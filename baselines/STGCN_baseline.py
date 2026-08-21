import os
import sys
import numpy as np
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicts.datasets.dataset_zoo import load_pems_data
from basicts.metrics.metric_zoo import compute_metrics

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
    print("STGCN Baseline Evaluation for Traffic Forecasting")
    print("=" * 70)
    
    # 加载数据（使用与模型相同的方式）
    data_file_path = config.get('DATA_FILE_PATH')
    adj_file_path = config.get('ADJ_FILE_PATH', None)
    
    input_length = config.get('INPUT_LENGTH', 12)
    output_length = config.get('OUTPUT_LENGTH', 12)
    train_ratio = config.get('TRAIN_RATIO', 0.6)
    val_ratio = config.get('VAL_RATIO', 0.2)
    normalize = config.get('NORMALIZE', True)

    print(f"\n[INFO] Loading data from {data_file_path}")
    train_data, val_data, test_data, adj_matrix, normalizer, offsets = load_pems_data(
        data_file_path, adj_file_path,
        max_train_samples=config.get('MAX_TRAIN_SAMPLES'),
        max_val_samples=config.get('MAX_VAL_SAMPLES'),
        max_test_samples=config.get('MAX_TEST_SAMPLES'),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=True,
        return_offsets=True
    )

    print(f"[INFO] Data shapes: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
    
    # 创建测试样本
    print(f"[INFO] Creating sliding window samples...")
    test_x, test_y = create_sliding_window_samples(test_data, input_length, output_length)
    print(f"[INFO] Test samples: x={test_x.shape}, y={test_y.shape}")
    # 转换为torch张量
    if normalizer is not None:
        test_x_tensor = torch.from_numpy(normalizer.inverse_transform(test_x)).float()
        test_y_tensor = torch.from_numpy(normalizer.inverse_transform(test_y)).float()
        print(f"[INFO] Data inverse transformed to original scale")
    
    results = {}
    metric_names = config.get('METRICS', ['MAE', 'RMSE', 'MAPE'])
    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
    print("-" * 70)

    # 对比STGCN模型结果
    print("\n[INFO] STGCN Model Best Validation Metrics (from training):")
    with open('outputs/smoke_STGCN_PEMS04/best_val_metrics.json', 'r', encoding='utf-8') as f:
        best_val_metrics = json.load(f)

    stgcn_results = {
        'MAE': best_val_metrics['MAE'],
        'RMSE': best_val_metrics['RMSE'],
        'MAPE': best_val_metrics['MAPE']  # 这个MAPE异常大，可能是计算问题
    }
    print(f"{'STGCN':<25} {stgcn_results['MAE']:>12.4f} {stgcn_results['RMSE']:>12.4f} {stgcn_results['MAPE']:>10.2f}%")

    # 对比STID模型结果
    print("\n[INFO] STID Model Best Validation Metrics (from training):")
    with open('outputs/smoke_STID_PEMS04/best_val_metrics.json', 'r', encoding='utf-8') as f:
        best_val_metrics = json.load(f)

    stid_results = {
        'MAE': best_val_metrics['MAE'],
        'RMSE': best_val_metrics['RMSE'],
        'MAPE': best_val_metrics['MAPE']  # 这个MAPE异常大，可能是计算问题
    }
    print(f"{'STID':<25} {stid_results['MAE']:>12.4f} {stid_results['RMSE']:>12.4f} {stid_results['MAPE']:>10.2f}%")
    # 保存结果
    output_dir = config.get('LOG_DIR', 'outputs/STID_PEMS04')
    os.makedirs(output_dir, exist_ok=True)

    baseline_results = {
        'config': config,
        'baselines': stgcn_results,
        'stid': stid_results,
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
    cfg_path = 'configs\STID_PEMS04.json'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    evaluate_baseline(config)