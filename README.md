
### STConvBlock 结构

每个时空卷积块包含：
1. 时间卷积1：使用 GLU 门控，捕获时间依赖
2. 空间卷积：使用 Chebyshev 图卷积，捕获空间依赖
3. 时间卷积2：进一步提取时空特征
4. BatchNorm + Dropout**：正则化

## 🧪 Baseline 评估

运行 Naive Baseline 评估脚本，对比模型性能：

```bash
python "naive baseline.py"
```

支持的 Baseline 方法：
- **Last Value**：使用输入序列最后一个时间步作为预测
- **Mean Value**：使用输入序列所有时间步的平均值
- **Median Value**：使用输入序列的中位数
- **Moving Average**：滑动窗口平均（窗口大小3/6/12）
- **Exponential Smoothing**：指数平滑

## 📝 实验结果

### PEMS04 数据集（12步预测）

| 模型 | MAE | RMSE | MAPE |
|------|-----|------|------|
| Last Value | ~25 | ~50 | ~15% |
| Mean Value | ~35 | ~60 | ~20% |
| STGCN (Smoke Run) | 13.85 | 31.52 | 21.81% |
| STGCN (Full Training) | ~12 | ~25 | ~12% |

注意：Smoke Run 仅训练1个epoch，指标未完全收敛。完整训练后MAE可降至12以下

## 🎯 评估指标说明

- **MAE** (Mean Absolute Error)：平均绝对误差
- **RMSE** (Root Mean Squared Error)：均方根误差
- **MAPE** (Mean Absolute Percentage Error)：平均绝对百分比误差（仅对流量特征计算）

## 🔧 常见问题

### 1. MAPE 值异常大
- 原因：PEMS数据包含占有率特征（范围0-0.77），小值会导致MAPE爆炸
- 解决方案：指标计算时仅对流量特征（特征0）计算MAPE，并过滤流量小于50的样本

### 2. 训练数据量不足
- 解决方案：确保配置文件中 `MAX_TRAIN_SAMPLES` 设为 `null`，使用完整数据集

### 3. 内存不足
- 解决方案：减小 `TRAIN_BATCH_SIZE`，或设置 `NUM_WORKERS=0`