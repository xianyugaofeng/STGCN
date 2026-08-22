import re
import json

def extract_last_run_config_and_metrics(log_path):
    """Extract config and best metrics from the last run in log file."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all "Config: {" occurrences and extract the JSON block
    # The config starts with "Config: {" and ends with a matching "}" on its own line
    # re.finditer(pattern, string)返回一个迭代器，包含所有非重叠匹配的Match对象
    # Config: 匹配字面量Config:
    # m.start() 返回当前匹配项在content中的起始索引
    config_starts = [m.start() for m in re.finditer(r'Config: \{', content)]
    
    if not config_starts:
        return None
    
    # 最后一个位置的起始索引
    last_start = config_starts[-1]
    
    # 确定整个配置块的结束位置
    # 默认结束位置设为起始位置

    brace_count = 0
    end_pos = last_start
    for i, ch in enumerate(content[last_start:], last_start):
        # last_start是enumerate的start参数，表示索引计数的起始值
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                end_pos = i + 1
                break
    
    config_str = content[last_start:end_pos]
    # Remove "Config: " prefix
    config_str = config_str.replace('Config: ', '', 1)
    
    try:
        config = json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error for {log_path}: {e}")
        return None
    
    # Extract key hyperparameters
    lr = config.get('TRAIN_LEARNING_RATE', 'N/A')
    batch_size = config.get('TRAIN_BATCH_SIZE', 'N/A')
    epochs = config.get('TRAIN_NUM_EPOCHS', 'N/A')
    
    # Model-specific params
    model_args = config.get('MODEL_ARGS', {})
    model_name = config.get('MODEL_NAME', '')
    
    if model_name == 'STID':
        # STID: embed_dim and num_layer
        hidden_dim = model_args.get('embed_dim', 'N/A')
        mlp_layers = model_args.get('num_layer', 'N/A')
    else:
        hidden_dim = 'N/A'
        mlp_layers = 'N/A'
    
    # Extract best validation metrics (last occurrence)
    metrics_pattern = r"Best validation metrics: \{'MAE': ([\d.]+), 'RMSE': ([\d.]+), 'MAPE': ([\d.]+)"
    metrics_matches = re.findall(metrics_pattern, content)
    
    if metrics_matches:
        last_metrics = metrics_matches[-1]
        mae = float(last_metrics[0])
        rmse = float(last_metrics[1])
        mape = float(last_metrics[2])
    else:
        mae = rmse = mape = 'N/A'
    
    return {
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'hidden_dim': hidden_dim,
        'mlp_layers': mlp_layers,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
    }

def generate_latex_table(results, output_path):
    """Generate complete LaTeX document from results dictionary."""
    latex = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{caption}
\\usepackage{adjustbox}

\\begin{document}

\\begin{table}[htbp]
\\centering
\\caption{Unified comparison of STID models on PEMS04 and PEMS08 datasets. Best validation metrics reported.}
\\label{tab:unified_results}
\\small
\\begin{adjustbox}{max width=\\textwidth}
\\begin{tabular}{lcccccccc}
\\toprule
\\textbf{Model} & \\textbf{LR} & \\textbf{BS} & \\textbf{Epochs} & \\textbf{Hidden} & \\textbf{MLP} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{MAPE (\\%)} \\\\
\\midrule
"""
    for model_name, metrics in results.items():
        latex += (f"{model_name} & {metrics['lr']} & {metrics['batch_size']} & "
                  f"{metrics['epochs']} & {metrics['hidden_dim']} & {metrics['mlp_layers']} & "
                  f"{metrics['MAE']:.3f} & {metrics['RMSE']:.3f} & {metrics['MAPE']:.3f} \\\\\n")
    
    latex += """\\bottomrule
\\end{tabular}
\\end{adjustbox}
\\end{table}

\\end{document}
"""
    with open(output_path, 'w') as f:
        f.write(latex)
    return latex

# Log file paths
log_files = {
    'STID (PEMS04, smoke)': r'outputs\smoke_STID_PEMS04\log.txt',
    'STID (PEMS08, smoke)': r'outputs\smoke_STID_PEMS08\log.txt',
    'STID (PEMS04, full)': r'outputs\STID_PEMS04\log.txt',
    'STID (PEMS08, full)': r'outputs\STID_PEMS08\log.txt',
}

# Extract metrics and configs
results = {}
for model_name, log_path in log_files.items():
    data = extract_last_run_config_and_metrics(log_path)
    if data:
        results[model_name] = data
        print(f"{model_name}: LR={data['lr']}, BS={data['batch_size']}, Epochs={data['epochs']}, "
              f"Hidden={data['hidden_dim']}, MLP={data['mlp_layers']}, "
              f"MAE={data['MAE']:.3f}, RMSE={data['RMSE']:.3f}, MAPE={data['MAPE']:.3f}")
    else:
        print(f"Failed to extract from {log_path}")

# Generate LaTeX table
output_tex = r'outputs\STID_results.tex'
latex_code = generate_latex_table(results, output_tex)
print(f"\nLaTeX table saved to {output_tex}")
print("\nGenerated LaTeX code:")
print(latex_code)