import numpy as np
import os
from datetime import datetime, timedelta

def check_dataset(dataset_name, data_dir="datasets"):
    # 检查PEMS数据集基础信息 区分真实0值与异常缺失
    file_path = f'STGCN_data/{data_dir}/{dataset_name}.npz'
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

    # 缺失值检测逻辑 交通数据特殊处理
    # 流量/速度=0 可能是真实值如深夜无车流，仅当连续>30分钟为0才视为异常
    # 占有率=0 通常为真实值，不计入缺失

    is_missing = np.zeros_like(traffic, dtype=bool) # 初始化缺失标记数组
    delay_missing = np.zeros_like(traffic, dtype=bool) # 初始化候补缺失标记数组
    # 创建一个与traffic形状完全相同T×N×C的布尔数组，初始全为False
    # 后续将把判定为异常缺失的位置标记为True

    # 提取关键特征(确保占有率存在)
    flow = traffic[...,0]
    feature_to_check = [0]  # 流量在所有数据集都存在

    if C >= 3: # 只有当流量和速度均存在时，占有率才会存在
        feature_to_check.append(1)
        speed = traffic[..., 1]
        feature_to_check.append(2)
        occ = traffic[...,2]
        # 检测传感器故障的两种关键矛盾
        # 流量=0但是占有率>0.5% -->传感器故障
        flow_fault = (flow == 0) & (occ > 0.005)
        # 速度=0但是占有率>0.5% -->传感器故障
        speed_fault = (speed == 0) & (occ > 0.005)
        # 只有数据集包含速度/占用率时才检查
        # 标记候补缺失 流量/速度/占有率都存在
        delay_missing[..., 0] = flow_fault
        delay_missing[..., 1] = speed_fault
        # 排除占有率=0的合理空闲状态

        for i in range(N): # 外层循环i遍历所有节点
            for c in feature_to_check: # 内层循环c仅检查流量(0)和速度(1)
                # np.where(条件) 返回满足条件的索引元组
                # np.diff(索引数组) 计算的是索引差值
                if c == 2:
                    continue

                for t in range(2,T): # 检查连续缺失序列 仅处理流量/速度特征
                    if np.all(delay_missing[t-2:t+1, i, c]):
                        # 连续3个时间步均为候选缺失 → 确认为真实缺失
                        is_missing[t-2:t+1, i, c] = True

    if C >= 3:
        valid_empty = (flow == 0) & (speed == 0) & (occ == 0)
        # 仅当满足空闲条件时，取消流量/速度的缺失标记
        is_missing[..., 0] &= ~valid_empty
        is_missing[..., 1] &= ~valid_empty

    for i in range(N): # 外层循环i遍历所有节点
        for c in feature_to_check: # 内层循环c仅检查流量(0)和速度(1)
            if c == 2:
                continue
            # np.where(条件) 返回满足条件的索引元组
            # np.diff(索引数组) 计算的是索引差值
            # 获取当前节点-特征的时间序列
            for t in range(6, T):
                if np.all(traffic[t - 6:t + 1, i, c] == 0):
                    # 连续3个时间步均为候选缺失 → 确认为真实缺失
                    is_missing[t - 6:t + 1, i, c] = True

    if C >= 2:
        missing_ratio = np.mean(is_missing[..., :2]) * 100
    else:
        missing_ratio = np.mean(is_missing[..., 0]) * 100
    # np.mean(is_missing)计算True的比例

    return {
        "dataset": dataset_name,
        "nodes": N,
        "timesteps": T,
        "time_range": f"{start_time} to {end_time}",
        "missing_ratio": f"{missing_ratio:.4f}%",
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
    print(f"{'数据集':<6} {'节点数':<6} {'时间步':<8} {'时间范围':<20} {'维度':<15} {'缺失率':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<6} {r['nodes']:<6} {r['timesteps']:<8} {r['time_range']:<20} {r['dimensions']:<15} {r['missing_ratio']:<8}")

