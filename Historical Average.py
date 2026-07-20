import numpy as np
import matplotlib.pyplot as plt
import os

# 常见参数
TIME_STEPS_PER_DAY = 288
PERIOD_LENGTH = 7
HISTORY_WEEKS = 4

# 数据划分比例
TRAIN_RAITO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

def import_dataset(dataset_name, data_dir="datasets"):
    file_path = f'STGCN_data/{data_dir}/{dataset_name}.npz'
    if not os.path.exists(file_path): # 若该文件不存在，直接返回一条提示字符串
        print(f"{dataset_name}文件不存在: {file_path}")
        return f"{dataset_name}文件不存在: {file_path}"

    data_dict = np.load(file_path)
    # PEMS数据通常存储在data key中
    if 'data' in data_dict:
        data = data_dict['data']
    else:
        key = list(data_dict.keys())[0]
        data = data_dict[key]

    if len(data.shape) == 3:
        data = data[:, :, 0] # 取第一个特征流量
    return data

def calculate_metrics(y_true, y_pred):
    # 计算MAE RMSE MAPE
    # 避免除以零
    epsilon = 1e-10
    MAE = np.mean(np.abs(y_true - y_pred))
    RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE忽略真实值为0的情况
    mask = y_true > 50
    if np.sum(mask) > 0:
        MAPE = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        MAPE = 0.0

    return MAE, RMSE, MAPE

class HistoricalAverageModel:
    def __init__(self, history_weeks=4, period_steps=TIME_STEPS_PER_DAY * 7):
        self.period_steps = period_steps # 一周的步数
        self.history_weeks = history_weeks

    def fit(self, train_data):
        # HA模型不需要传统意义上的'fit' 需要存储训练数据以便查询历史
        # 为了效率，我们可以在predict时动态计算或者预先计算好均值表
        # 采用动态计算方式，更灵活且内存友好
        self.train_data = train_data
        self.train_size = train_data.shape[0]
        print(f"模型初始化完成。使用过去{self.history_weeks}周的数据进行平均")

    def predict(self, test_start_index, total_test_steps, num_nodes):
        # 对测试集进行预测
        # test_start_index 测试集在原始数据的起始索引
        # total_test_steps 测试集的总步数
        # num_nodes 节点数
        # 预测结果 [Total_Test_Steps, Num_Nodes]


        predictions = np.zeros((total_test_steps, num_nodes))
        print("正在进行Historical Average预测")

        for t in range(total_test_steps):
            # 当前绝对时间索引
            current_abs_t = test_start_index + t

            # 获取历史同期的索引列表
            # 例如: t，t-1周，t-2周，t-3周
            history_indices = []
            for w in range(1, self.history_weeks + 1):
                hist_idx = current_abs_t - (w * self.period_steps)
                if hist_idx < self.train_size and hist_idx >= 0:
                    history_indices.append(hist_idx)

            if len(history_indices) > 0:
                # 提取历史数据并求平均
                # train_data[history_indices] 形状: [Num_History, Num_Nodes]
                historical_values = self.train_data[history_indices]
                pred_value = np.mean(historical_values, axis=0)
            else:
                # 如果没有任何历史数据可用(例如测试集刚开始且训练集不够长)
                # 退化为使用全局训练集均值或上一时刻值
                # 这里简单退化为训练集最后一个时刻的值，或者全0
                pred_value = self.train_data[-1] if len(self.train_data) > 0 else np.zeros(num_nodes)

            predictions[t] = pred_value

        return predictions

    def plot_results(self, y_true, y_pred, node_id=0, steps_to_plot=288*2):
        # 绘制单个节点前两天的预测对比
        plt.figure(figsize=(12, 6))
        time_axis = np.arange(steps_to_plot)

        plt.plot(time_axis, y_true[:steps_to_plot, node_id], label="True Value",
                 color="blue", linewidth=1.5)
        plt.plot(time_axis, y_pred[:steps_to_plot, node_id], label="HA Prediction",
                 linestyle='--', color="red", linewidth=1.5)

        plt.title(f'Historical Average Prediction vs True Value (Node {node_id})')
        plt.xlabel('Time Steps(5 min interval)')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# 主执行流程
if __name__ == '__main__':
    # 加载数据
    for name in ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]:
        raw_data = import_dataset(name, name)
        NUM_NODES = raw_data.shape[1]

        print(f"数据形状：{NUM_NODES}")
        print(f"数据范围：[{raw_data.min():.2f}, {raw_data.max():.2f}]")

        # 数据划分
        total_steps = raw_data.shape[0]
        train_size = int(total_steps * TRAIN_RAITO)
        val_size = int(total_steps * VAL_RATIO)
        # test_size剩余部分

        train_data = raw_data[:train_size]
        val_data = raw_data[train_size:train_size + val_size]
        test_data = raw_data[train_size + val_size:]

        test_start_index = train_size + val_size

        print(f"训练集大小: {train_data.shape[0]}步")
        print(f"验证集大小: {val_data.shape[0]}步")
        print(f"测试集大小: {test_data.shape[0]}步")

        # 初始化并运行HA模型
        # HA通常只在训练集上查找历史，或者在整个可用历史中查找
        # 严格基线通常只使用训练集数据来计算历史平均值，以防止数据泄露
        model = HistoricalAverageModel(history_weeks=HISTORY_WEEKS)
        model.fit(train_data)

        # 预测测试集
        predictions = model.predict(test_start_index, test_data.shape[0], NUM_NODES)

        # 评估结果
        mae, rmse, mape = calculate_metrics(test_data, predictions)
        print("\n" + "="*30)
        print("Historical Average Baseline Results")
        print("="*30)
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print("="*30)

        # 可视化
        # 选择一个流量变化比较明显的节点进行展示
        # 计算每个节点在测试集上的方差，选方差最大的节点
        # 从所有预测节点中，自动选取预测值波动最剧烈的节点，并可视化其预测结果
        node_variances = np.var(predictions, axis=0) # 对每个节点，求它在所有时间步上预测值的方差
        target_node = np.argmax(node_variances)

        print(f"正在绘制节点 {target_node} 的预测结果...")
        model.plot_results(test_data, predictions, node_id=target_node)