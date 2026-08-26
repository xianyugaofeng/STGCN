import numpy as np
import h5py
import pandas as pd
# 查看PEMS-BAY数据特征分布
with h5py.File('PEMSdata/PEMS-BAY/PEMS-BAY.h5', 'r') as f:
    # 递归打印文件完整结构
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Group:   {name}")
    
    print("=== HDF5 文件结构 ===")
    f.visititems(print_structure)
    
    # 同时打印根级别的 keys
    print("\n=== 根级别 keys ===")
    print(list(f.keys()))

"""代码运行结果
(D:\data\CondaEnvs\torch-gpu5060) PS D:\data\STGCN> python .\check.py
=== HDF5 文件结构 ===
  Group:   speed
  Dataset: speed/axis0, shape=(325,), dtype=int64
  Dataset: speed/axis1, shape=(52116,), dtype=int64
  Dataset: speed/block0_items, shape=(325,), dtype=int64
  Dataset: speed/block0_values, shape=(52116, 325), dtype=float64

=== 根级别 keys ===
['speed']
"""

"""
print(f"\n特征0（流量）范围: [{data[:,:,0].min():.2f}, {data[:,:,0].max():.2f}], 均值: {data[:,:,0].mean():.2f}")
print(f"特征1（速度）范围: [{data[:,:,1].min():.2f}, {data[:,:,1].max():.2f}], 均值: {data[:,:,1].mean():.2f}")
print(f"特征2（占有率）范围: [{data[:,:,2].min():.2f}, {data[:,:,2].max():.2f}], 均值: {data[:,:,2].mean():.2f}")

# 统计接近0的值
print(f"\n特征0小于50的比例: {(data[:,:,0] < 50).sum() / data[:,:,0].size * 100:.2f}%")
print(f"特征1小于50的比例: {(data[:,:,1] < 50).sum() / data[:,:,1].size * 100:.2f}%")
print(f"特征2小于50的比例: {(data[:,:,2] < 50).sum() / data[:,:,2].size * 100:.2f}%")
"""