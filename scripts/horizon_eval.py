"""
Horizon Evaluation Script
Evaluates prediction error (MAE/MSE) vs prediction horizon (τ) on test set
using rolling origins. For each horizon τ, computes average error across all origins.
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pickle
import json

# Add project root to path
sys.path.insert(0, r'D:\data\STGCN')

from basicts.datasets.dataset_zoo import load_pems_data, get_dataset
from basicts.models import get_model
from basicts.metrics.metric_zoo import compute_metrics
from basicts.utils.data_utils import Normalizer


def load_model_and_config(model_path, config=None):
    """Load model checkpoint and config."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if config is None:
        config = checkpoint.get('config', {})
    
    model_name = config.get('MODEL_NAME', 'STGCN')
    model_args = config.get('MODEL_ARGS', {})
    
    # Ensure required args are present
    model_args['num_nodes'] = config.get('NUM_NODES', 307)
    model_args['num_features'] = config.get('NUM_FEATURES', 3)
    model_args['input_length'] = config.get('INPUT_LENGTH', 12)
    model_args['output_length'] = config.get('OUTPUT_LENGTH', 12)
    
    model = get_model(model_name)(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    normalizer = None
    if 'normalizer_state' in checkpoint:
        normalizer = Normalizer()
        normalizer.__dict__.update(checkpoint['normalizer_state'])
    
    return model, config, normalizer


def prepare_data(config, normalizer=None):
    """Load and prepare test data."""
    data_file = config.get('DATA_FILE_PATH', 'PEMSdata/PEMS04/PEMS04.npz')
    adj_file = config.get('ADJ_FILE_PATH', 'PEMSdata/PEMS04/adj_PEMS04.pkl')
    input_length = config.get('INPUT_LENGTH', 12)
    output_length = config.get('OUTPUT_LENGTH', 12)
    train_ratio = config.get('TRAIN_RATIO', 0.6)
    val_ratio = config.get('VAL_RATIO', 0.2)
    normalize = config.get('NORMALIZE', True)
    
    # Load full data to get test split
    data = np.load(data_file)['data']  # (T, N, C)
    num_timesteps = data.shape[0]
    num_nodes = data.shape[1]
    train_end = int(num_timesteps * train_ratio)
    val_end = train_end + int(num_timesteps * val_ratio)
    test_data = data[val_end:]
    
    # Load adj matrix
    adj_matrix = None
    if adj_file and os.path.exists(adj_file):
        with open(adj_file, 'rb') as f:
            adj_matrix = pickle.load(f)
        print(f"[INFO] Loaded adjacency matrix from {adj_file}")
    else:
        print(f"[WARN] Adjacency matrix file not found: {adj_file}")
        # Try to generate from CSV like processor_zoo does
        csv_path = data_file.replace('.npz', '.csv')
        if os.path.exists(csv_path):
            print(f"[INFO] Creating adjacency matrix from CSV: {csv_path}")
            adj_matrix = create_adjacency_from_csv(csv_path, num_nodes)
        else:
            print(f"[WARN] CSV file not found: {csv_path}")
    
    # Fit normalizer on train data if not provided
    if normalizer is None and normalize:
        train_data = data[:train_end]
        normalizer = Normalizer()
        normalizer.fit(train_data)
    
    if normalizer is not None:
        test_data = normalizer.transform(test_data)
    
    # Store test offset for STID
    config['TEST_OFFSET'] = val_end
    
    return test_data, adj_matrix, normalizer


def create_adjacency_from_csv(csv_path, num_nodes=None, symmetric=True,
                              default_diag=1.0, threshold=0.1,
                              source_col='from', target_col='to', weight_col='cost'):
    """Generate adjacency matrix from CSV (copied from processor_zoo)."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read CSV file: {csv_path}, error: {e}")
        n = num_nodes if num_nodes is not None else 1
        return np.eye(n)

    df.columns = df.columns.str.strip()
    if weight_col not in df.columns:
        for alt in ['weight', 'w', 'cost', 'distance', 'length']:
            if alt in df.columns:
                weight_col = alt
                break
        else:
            df[weight_col] = 1.0

    if source_col not in df.columns:
        for alt in ['source', 'src', 'from', 'node_from', 'start']:
            if alt in df.columns:
                source_col = alt
                break
        else:
            raise ValueError(f"CSV missing source column")

    if target_col not in df.columns:
        for alt in ['target', 'dst', 'to', 'node_to', 'end']:
            if alt in df.columns:
                target_col = alt
                break
        else:
            raise ValueError(f"CSV missing target column")

    if num_nodes is None:
        raise ValueError(f"Adjacency matrix missing num_nodes parameter")

    distances = df[weight_col].values.astype(np.float32)
    sigma = 0.5 * np.std(distances)
    print(f"[INFO] Gaussian kernel sigma = {sigma:.4f}")
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weights[weights < threshold] = 0.0

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    u_arr = df[source_col].values.astype(np.int64)
    v_arr = df[target_col].values.astype(np.int64)
    mask = (u_arr != v_arr) & (weights > 0)
    np.maximum.at(adj_matrix, (u_arr[mask], v_arr[mask]), weights[mask])

    if symmetric:
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    np.fill_diagonal(adj_matrix, default_diag)
    return adj_matrix


def get_model_kwargs(model_name, adj_matrix, config):
    """Get model-specific kwargs for forward pass."""
    device = config.get('DEVICE', 'cuda:0')
    if torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')
    
    if model_name == 'STGCN':
        # STGCN model can work without adj_matrix (uses learned laplacian)
        # Only pass laplacian if adj_matrix is available and valid
        if adj_matrix is not None:
            # Compute symmetrically normalized Laplacian like STGCN model does
            if isinstance(adj_matrix, np.ndarray):
                adj_matrix = torch.from_numpy(adj_matrix).float()
            # Symmetrize
            adj_matrix = (adj_matrix + adj_matrix.t()) / 2
            # Degree matrix
            d = adj_matrix.sum(dim=1)
            # Check for zero-degree nodes
            if (d == 0).any():
                print("Warning: Zero-degree nodes detected, using learned laplacian instead")
                return {}
            d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
            d_sqrt_inv = torch.diag(d_sqrt_inv)
            laplacian = torch.matmul(torch.matmul(d_sqrt_inv, adj_matrix), d_sqrt_inv)
            # Add small diagonal to ensure positive definite
            laplacian = laplacian + 1e-6 * torch.eye(laplacian.size(0), device=laplacian.device)
            laplacian = laplacian.to(device)
            return {'laplacian': laplacian}
        # If no adj_matrix, model will use its learned laplacian
        return {}
    elif model_name == 'STID':
        # STID doesn't need adj matrix in forward
        return {}
    return {}


def rolling_origin_evaluation(model, test_data, config, model_kwargs, 
                              output_length=12, stride=1):
    """
    Rolling origin evaluation on test set.
    
    Args:
        model: trained model
        test_data: test data array (T_test, N, C)
        config: configuration dict
        model_kwargs: additional kwargs for model forward
        output_length: prediction horizon H
        stride: step size between rolling origins
    
    Returns:
        horizon_errors: dict with keys 'MAE', 'MSE' containing arrays of shape (H,)
    """
    device = next(model.parameters()).device
    input_length = config.get('INPUT_LENGTH', 12)
    num_features = config.get('NUM_FEATURES', 3)
    
    # Create dataset
    dataset_class = get_dataset(config.get('MODEL_NAME', 'STGCN'))
    if config.get('MODEL_NAME') == 'STID':
        dataset = dataset_class(
            test_data, input_length=input_length, output_length=output_length,
            mode='test', steps_per_day=288,
            add_time_of_day=config.get('ADD_TIME_OF_DAY', True),
            add_day_of_week=config.get('ADD_DAY_OF_WEEK', True),
            global_start=config.get('TEST_OFFSET', 0)
        )
    else:
        dataset = dataset_class(test_data, input_length=input_length, output_length=output_length)
    
    # Collect all predictions and targets for each horizon
    all_preds = []  # list of (num_origins, N, C) for each horizon
    all_targets = []
    
    num_origins = len(dataset)
    
    with torch.no_grad():
        for origin_idx in range(0, num_origins, stride):
            x, y = dataset[origin_idx]  # x: (L, N, C), y: (H, N, C)
            x = torch.from_numpy(x).unsqueeze(0).float().to(device)  # (1, L, N, C)
            y = torch.from_numpy(y).unsqueeze(0).float().to(device)  # (1, H, N, C)
            
            pred = model(x, **model_kwargs)  # (1, H, N, C)
            
            # Store for each horizon step
            for tau in range(output_length):
                if tau >= len(all_preds):
                    all_preds.append([])
                    all_targets.append([])
                all_preds[tau].append(pred[0, tau].cpu().numpy())  # (N, C)
                all_targets[tau].append(y[0, tau].cpu().numpy())   # (N, C)
    
    # Compute metrics for each horizon
    horizon_mae = []
    horizon_mse = []
    
    for tau in range(output_length):
        if len(all_preds[tau]) == 0:
            horizon_mae.append(np.nan)
            horizon_mse.append(np.nan)
            continue
            
        preds_tau = torch.from_numpy(np.stack(all_preds[tau]))  # (num_origins, N, C)
        targets_tau = torch.from_numpy(np.stack(all_targets[tau]))
        
        # Compute MSE and MAE
        mse = torch.mean((preds_tau - targets_tau) ** 2).item()
        mae = torch.mean(torch.abs(preds_tau - targets_tau)).item()
        
        horizon_mae.append(mae)
        horizon_mse.append(mse)
    
    return {
        'MAE': np.array(horizon_mae),
        'MSE': np.array(horizon_mse),
    }


def plot_horizon_errors(horizon_errors, model_name, dataset_name, output_dir):
    """Plot MAE and MSE vs horizon."""
    output_length = len(horizon_errors['MAE'])
    horizons = np.arange(1, output_length + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MAE plot
    ax1.plot(horizons, horizon_errors['MAE'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Prediction Horizon (τ)', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title(f'{model_name} on {dataset_name} - MAE vs Horizon', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(horizons)
    
    # MSE plot
    ax2.plot(horizons, horizon_errors['MSE'], 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Prediction Horizon (τ)', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title(f'{model_name} on {dataset_name} - MSE vs Horizon', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(horizons)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'horizon_{model_name}_{dataset_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save data
    np.savez(os.path.join(output_dir, f'horizon_{model_name}_{dataset_name}.npz'),
             horizons=horizons, MAE=horizon_errors['MAE'], MSE=horizon_errors['MSE'])
    
    print(f"Plot saved to {save_path}")
    return save_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/horizon_plots')
    parser.add_argument('--stride', type=int, default=1, help='Stride between rolling origins')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model, config, normalizer = load_model_and_config(args.model_path)
    
    print("Preparing test data...")
    test_data, adj_matrix, normalizer = prepare_data(config, normalizer)
    
    print("Running rolling origin evaluation...")
    model_kwargs = get_model_kwargs(config.get('MODEL_NAME', 'STGCN'), adj_matrix, config)
    device = config.get('DEVICE', 'cuda:0')
    if torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')
    model = model.to(device)
    
    output_length = config.get('OUTPUT_LENGTH', 12)
    horizon_errors = rolling_origin_evaluation(
        model, test_data, config, model_kwargs, 
        output_length=output_length, stride=args.stride
    )
    
    print(f"\nHorizon MAE: {horizon_errors['MAE']}")
    print(f"Horizon MSE: {horizon_errors['MSE']}")
    
    model_name = config.get('MODEL_NAME', 'STGCN')
    dataset_name = config.get('DATASET_NAME', 'PEMS04')
    
    print(f"\nPlotting results...")
    plot_horizon_errors(horizon_errors, model_name, dataset_name, args.output_dir)
    
    # Print table
    print("\nHorizon | MAE       | MSE")
    print("--------|-----------|-----------")
    for tau in range(output_length):
        print(f"{tau+1:7d} | {horizon_errors['MAE'][tau]:9.4f} | {horizon_errors['MSE'][tau]:9.4f}")


if __name__ == '__main__':
    main()