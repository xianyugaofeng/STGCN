import numpy as np

# 查看PEMS04数据特征分布
data = np.load('PEMSdata/PEMS04/PEMS04.npz')['data']
print(f"数据形状: {data.shape}")
print(f"\n特征0（流量）范围: [{data[:,:,0].min():.2f}, {data[:,:,0].max():.2f}], 均值: {data[:,:,0].mean():.2f}")
print(f"特征1（速度）范围: [{data[:,:,1].min():.2f}, {data[:,:,1].max():.2f}], 均值: {data[:,:,1].mean():.2f}")
print(f"特征2（占有率）范围: [{data[:,:,2].min():.2f}, {data[:,:,2].max():.2f}], 均值: {data[:,:,2].mean():.2f}")

# 统计接近0的值
print(f"\n特征0小于50的比例: {(data[:,:,0] < 50).sum() / data[:,:,0].size * 100:.2f}%")
print(f"特征1小于50的比例: {(data[:,:,1] < 50).sum() / data[:,:,1].size * 100:.2f}%")
print(f"特征2小于50的比例: {(data[:,:,2] < 50).sum() / data[:,:,2].size * 100:.2f}%")