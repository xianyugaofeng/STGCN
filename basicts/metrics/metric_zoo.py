import torch
import numpy as np

def calculate_mae(pred, target):
    # Mean Absolute Error"""
    return torch.abs(pred - target).mean().item()

def calculate_rmse(pred, target):
    # Root Mean Squared Error"""
    return torch.sqrt(torch.square(pred - target).mean()).item()

def calculate_mape(pred, target, eps=1e-8):
    # Mean Absolute Percentage Error
    # 只对流量特征（第一个特征通道）计算MAPE
    # PEMS数据特征顺序：[流量, 占有率, 速度]
    # 占有率范围0-0.77，速度范围3-85，只有流量适合计算MAPE
    
    if pred.dim() == 4 and pred.size(-1) > 1:
        # (batch, output_length, num_nodes, num_features) → 取流量特征
        pred_flow = pred[:, :, :, 0]
        target_flow = target[:, :, :, 0]
    else:
        pred_flow = pred
        target_flow = target
    
    # 过滤流量小于50的值（避免小流量导致MAPE异常）
    mask = (target_flow != 0).float()
    if mask.sum() == 0:
        return 0.0
    
    mape = torch.abs((pred_flow - target_flow) / (target_flow + eps)) * mask
    return (mape.sum() / mask.sum() * 100).item()

def calculate_mape_flow(pred, target, eps=1e-8, threshold=50.0):
    # 只对流量特征（第一个特征通道）计算MAPE
    # 交通预测中通常只关注流量指标
    if pred.dim() == 4 and pred.size(-1) > 1:
        # (batch, output_length, num_nodes, num_features)
        pred_flow = pred[:, :, :, 0]
        target_flow = target[:, :, :, 0]
    else:
        pred_flow = pred
        target_flow = target
    
    mask = (target_flow > threshold).float()
    if mask.sum() == 0:
        return 0.0
    mape = torch.abs((pred_flow - target_flow) / (target_flow + eps)) * mask
    return (mape.sum() / mask.sum() * 100).item()

METRIC_ZOO = {
    'MAE': calculate_mae,
    'RMSE': calculate_rmse,
    'MAPE': calculate_mape_flow,  # 使用流量特征专用MAPE
}

def get_metrics(metric_names):
    # Get metric functions by names
    return [METRIC_ZOO[name] for name in metric_names]

def compute_metrics(pred, target, metric_names):
    # Compute multiple metrics
    results = {}
    for name in metric_names:
        if name in METRIC_ZOO:
            results[name] = METRIC_ZOO[name](pred, target)
    return results