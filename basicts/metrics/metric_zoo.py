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
    mask = (target != 0).float()
    mape = torch.abs((pred - target) / (target + eps)) * mask
    return (mape.sum() / mask.sum() * 100).item()

METRIC_ZOO = {
    'MAE': calculate_mae,
    'RMSE': calculate_rmse,
    'MAPE': calculate_mape,
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