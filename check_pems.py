import numpy as np
import os
from datetime import datetime, timedelta

def check_dataset(dataset_name, data_dir="datasets"):
    # 检查PEMS数据集基础信息 区分真实0值与异常缺失
    file_path = f'PEMSdata/{data_dir}/{dataset_name}.npz'
    if not os.path.exists(file_path): # 若该文件不存在，直接返回一条提示字符串
        return f"{dataset_name}文件不存在: {file_path}"

    data = np.load(file_path)
    traffic = data['data']

    # 关键维度统计
    T, N, C = traffic.shape
    # 三维交通流量数据 时间步 × 节点数 × 特征数
    # 时间步数--样本数 空间节点数--传感器数量 特征通道数--流量,速度，占有率
    if dataset_name == "PEMS04":
        start_date = datetime(2018, 1, 1)
    elif dataset_name == 'PEMS03':
        start_date = datetime(2018, 9, 1)
    elif dataset_name == "PEMS07":
        start_date = datetime(2017, 5, 1)
    elif dataset_name == "PEMS08":
        start_date = datetime(2016, 7, 1)

    end_date = start_date + timedelta(minutes=(T - 1)* 5)

    start_time = start_date.strftime('%Y-%m-%d')
    end_time = end_date.strftime('%Y-%m-%d')
    # 时间戳原始单位是纳秒
    # 得到数据集的起始日期和结束日期

    return {
        "dataset": dataset_name,
        "nodes": N,
        "timesteps": T,
        "time_range": f"{start_time} to {end_time}",
        "dimensions": f"{T}x{N}x{C}"
    }

# 执行检查
results = []
for name in ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]:
    res = check_dataset(name, name)
    if isinstance(res, dict):
        # 如果文件存在且成功检查，res将是一个字典
        # 包含dataset、nodes、timesteps、time_range、missing_ratio、dimensions、note等字段
        results.append(res)
    else:
        # 如果文件不存在，res将是一个字符串
        print(res)

# 生成概览表格
if results:
    print("\n===== PEMS数据集概览 =====")
    print(f"{'数据集':<6} {'节点数':<6} {'时间步':<8} {'时间范围':<20} {'维度':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<6} {r['nodes']:<6} {r['timesteps']:<8} {r['time_range']:<30} {r['dimensions']:<15}")

