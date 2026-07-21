import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChebConv(nn.Module):
    # Chebyshev Graph Convolution
    def __init__(self, in_channels, out_Channels, K, bias=True):
        super(ChebConv, self).__init__()
        self.K = K # 保存多项式的阶数，供前向传播时使用
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self._cached_lambda_max = None
        if bias: # bias布尔值，是否添加可学习的偏置项
            self.bias = nn.Paramter(torch.Tensor(out_channels))
        else:
            self.register_paramter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # 在模型构造时对权重和偏置进行合理的初始化
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # self.weight 形状(K, in_channels, out_channels)进行kaiming均匀初始化
        if self.bias is not None:
            # 如果偏置项存在（且不是 None）然后就对其进行初始化
            fan_in = self.in_channels
            # 计算张量中输入单元数（fan_in）和输出单元数（fan_out）的一个工具函数
            # 该函数返回(fan_in, fan_out)元组
            bound = 1.0 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            # uniform_将偏置初始化为[-bound, bound]内的均匀分布
    
    def forward(self, x, laplacian):
        # x: (batch, in_channels, num_nodes)
        # laplacian: (num_nodes, num_nodes)
        # returns: (batch, out_channels. num_nodes)
        batch_size, in_channels, num_nodes = x.size()
        # 先计算多项式矩阵
        lambda_max = get_lambda_max_eigh(laplacian)

        L_scaled = (2.0/lambda_max) * laplacian - torch.eye(num_nodes, device=x.device)
        laplacian_power = [torch.eye(num_nodes, device=x.device), L_scaled]
        for k in range(2, self.K):
            laplacian_power.append(2 * torch.mm(L_scaled, laplacian_power[-1]) - laplacian_power[-2])
        
        outputs = []
        for k in range(self.K):
            # (batch, in_channels, num_nodes) @ (num_nodes, num_nodes) -> (batch, in_channels, num_nodes)
            # 将多项式矩阵作用在特征上，相当于沿图的结构进行k跳邻居的信息聚合
            tmp = torch.matmul(x, laplacian_power[k])
            # (batch, in_channels, num_nodes) @ (in_channels, out_channels) -> (batch, out_channels, num_nodes)
            # 将tmp的维度转置，使每个节点变成一个in_channels维的行向量
            # 所有的num_node和batch都被自动广播处理
            tmp = torch.matmul(tmp.transpose(1, 2), self.weight[k]).transpose(1, 2)
            outputs.append(tmp) # 收集k阶的结果
        
        output = torch.stack(outputs, dim=-1).sum(dim=-1)
        # stack在最后一维堆叠  沿K维求和，将所有阶的贡献加起来
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).unsqueeze(-1)
            # 通过unsqueeze变成(1, out_channels, 1)以便广播
            # 加到output上，为每个输出通道增加一个可学习的偏置
        return output

    def get_lambda_max_eigh(laplacian):
        # 对称矩阵(N,N)
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        # 计算一个实对称矩阵的所有特征值
        # 取排序后的最后一个元素，即最大的特征值
        return eigenvalues[-1].item() # 升序排列 将仅含一个元素的张量转换为Python标量

