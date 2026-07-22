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

class TemporalConv(nn.Module):
    # Temporal Convolution with GLU
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=(kernel_size, 1), 
                              padding=((kernel_size-1)//2 , 0)) 
        # (N, out_channels*2, T, V)
        # 只在时间维度上做卷积 在空间维度上不进行混合
        # 只在时间维度两端填充，保证输出的时间长度与输入相同；空间维度不填充
        self.gate = nn.GLU(dim=1) # 沿通道维度(dim=1)将数据分为两半

    def forward(self, x):
        x = self.conv(x) #(N, in_channels, T, V) ->(N, out_channels * 2, T, V)
        x = self.gate(x) # output = A ⊙ σ(B) -> (N, out_channels, T, V)
        return x
    
class STConvBlock(nn.Module):
    # Spatio-Temporal Convolution Block
    def __init__(self, in_channels, hidden_channels, out_channels, Kt, Ks, 
                 dropout=0.0, graph_conv_type='cheb_conv'):
        super(STConvBlock, self).__init__()
        self.temporal_conv1 = TemporalConv(in_channels, hidden_channels, Kt)
        if graph_conv_type == 'cheb_conv':
            self.spatial_conv = ChebConv(hidden_channels, hidden_channels, Ks)
        else:
            raise ValueError(f"Unsupported graph_conv_type:" {graph_conv_type})
        self.temporal_conv2 = TemporalConv(hidden_channels, out_channels, Kt)
        self.batch_norm == nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, laplacian):
        x = self.temporal_conv1(x)
        x = F.relu(x)

        batch, channels, seq_len, nodes = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch * seq_len, channels, nodes)
        # 每个时间步的图数据独立排布 将批次和时间步合并
        x = self.spatial_conv(x, laplacian)
        # 利用图的拉普拉斯矩阵laplacian，在每个时间步内沿nodes维度聚合邻居节点信息
        x = x.view(batch, seq_len, channels, nodes).permute(0, 2, 1, 3).contiguous()
        # 恢复原始维度顺序
        x = F.relu(x)

        x = self.temporal_conv2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        return x

class STGCN(nn.Module):
    # Spatio-Temporal Graph Convolutional Network
    def __init__(self, Kt=3, Ks=3, blocks=[[1, 32, 64], [64, 64, 128], [128, 128, 256]],
                dropout=0.1, graph_conv_type='cheb_co nv', num_nodes=307, num_features=3,
                input_length=12, output_length=12):
        super(STGCN, self).__init__()
        self.Kt = Kt
        self.Ks = Ks
        self.dropout = dropout
        self.graph_conv_type = graph_conv_type
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.output_length = output_length

        # Build STConv blocks
        self.st_conv_blocks = nn.ModuleList()
        for i, block in enumerate(blocks):
            # 每个元素[in_channels, hidden_channels, out_channels]定义每个STConvBlock的输入、隐藏和输出通道数
            in_channels = block[0] if i == 0 else blocks[i-1][2]
            # 第一个块使用预设的in_channels例如blocks[0][0]=1，后面的块则自动继承前一块的输出通道数
            hidden_channels = block[1]
            out_channels = block[2]
            self.st_conv_blocks.append(
                STConvBlock(in_channels, hidden_channels, out_channels, Kt, Ks, dropout, graph_conv_type)
            )

        # Output layer
        last_out_channels = blocks[-1][2]
        self.output_layer = nn.Conv2d(last_out_channels, output_length, kernel_size=(1,1))
        # (B, output_length, T, N)
        # 不改变时间维度T和空间节点数N 仅在通道维度做线性组合 

        # Learnable laplacian (will be replaced if provided)
        self.laplacian = nn.Parameter(torch.randn(num_nodes, num_nodes))
    
    def forward(self, x, laplacian=None):
        # x: (batch, input_length, num_nodes, num_features) 
        # or (batch, num_features, input_length, num_nodes)
        # returns: (batch, output_length, num_nodes, num_features)

        # Handle different input shapes
        if x.size(1) == self.input_length:
            # (batch, input_length, num_nodes, num_features) 
            # -> (batch, num_features, input_length, num_nodes)
            x.permute(0, 3, 1, 2).contiguous()
        elif x.size(1) == self.num_features:
            # Already in (batch, num_features, input_length, num_nodes) format
            pass
        else:
            raise ValueError(f"Input shape not supported: {x.shape}")
        
        # Use provided laplacian or learned one
        if laplacian is None:
            laplacian = self.laplacian
            # 如果外部没有传入预先定义好的图拉普拉斯矩阵则使用模型内部可学习的参数矩阵
            # Symmetric normalization
            laplacian = (laplacian + laplacian.t()) / 2
            d = torch.diag(laplacian.sum(dim=1)) # 对每一行求和,表示节点的度并转化为对角矩阵
            laplacian = torch.matmul(torch.matmul(torch.sqrt(torch.inverse(d)), laplacian), 
                                     torch.sqrt(torch.inverse(d)))
            # 对称归一化拉普拉斯
            
        # Forward through STConv blocks
        for block in self.st_conv_blocks:
            x = block(x, laplacian) # (batch, in_channels, input_length, num_nodes)
        # 依次将数据通过所有时空卷积块 block(x, laplacian)调用的是STConvBlock的forward方法

        # Output layer
        x = self.output_layer(x) # 将特征通道数从last_out_channels压缩为output_length
        # (batch, last_out_channels, T, num_nodes) -> (batch, output_length, input_length, num_nodes)
        

        # Take last time step output
        x = x[:, :, -1, :] # (batch, output_length, num_nodes)
        # 最后一个时间步的特征汇聚了整个输入序列的信息

        # Reshape to match expected output
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.num_features)
        # 在最后添加一个维度，形状变为(batch, output_length, num_nodes, 1)
        # 沿着新维度重复self.num_features次
        # (batch, output_length, num_nodes, num_features)
        return x