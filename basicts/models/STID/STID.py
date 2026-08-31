import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
    
    def forward(self, input_data: torch.tensor) -> torch.tensor: 
        """forward of MLP"""
        # input_data(torch.Tensor): input data with shape[T, N, C]
        # returns: torch.tensor
        hidden = self.fc2(self.dropout(self.act(self.fc1(input_data)))) # MLP
        hidden = hidden + input_data # residual connection
        return hidden

class STID(nn.Module):
    """Spatial-Temporal Identity"""
    def __init__(self, num_nodes=307, num_features=3, node_dim=32, input_length=12,
                 input_dim=3, embed_dim=32, output_length=12, num_layer=3,
                 temp_dim_tid=32, temp_dim_diw=32, time_of_day_size=288, day_of_week_size=7,
                 if_time_in_day=True, if_day_in_week=True, if_node=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.node_dim = node_dim
        self.input_length = input_length
        self.input_dim = input_dim 
        self.embed_dim = embed_dim
        self.output_length = output_length
        self.num_layer = num_layer
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_node = if_node
    
        # spatial embeddings
        if if_node:
        # nn.Parameter(...)：将该张量注册为模型的可学习参数
        # 它会被自动加入model.parameters() 在训练过程中通过反向传播进行梯度更新
        # 这是一个查找表Lookup Table，每个节点拥有独立的、可通过训练优化的向量表示，用于捕获节点的静态空间特性如地理位置、功能区类型等
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
        # 根据输入和输出维度动态计算初始化范围，使各层输出的方差保持一致
            nn.init.xavier_uniform_(self.node_emb)
        
        # temporal embeddings
        # 每个时间步的嵌入向量维度temp_dim_tid
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        
        # embedding layer
        # 将原始多维时序数据投影到统一隐藏空间 压缩历史 → 隐藏空间
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_length * self.input_dim, out_channels=embed_dim, kernel_size=(1, 1), bias=True
        )

        # encoding
        # 模型的特征融合与编码阶段
        # 它将前面初始化的多种嵌入空间、时间、原始序列在维度上进行拼接，并通过多层感知机MLP进行非线性特征提取
        self.hidden_dim = self.embed_dim + int(self.if_node) * self.node_dim + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        # 动态创建num_layer个相同的MLP实例
        # 并解包为nn.Sequential的参数
        # 对拼接后的多源嵌入进行跨模态特征交互与非线性变换，使模型能够学习到空间、时间与观测值之间的复杂耦合关系
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(num_layer)]
        )

        # regression
        # 该层与time_series_emb_layer形成完美的编码器-解码器对称结构
        # 隐藏空间 → 展开未来
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_length*self.num_features, kernel_size=(1, 1), bias=True
        )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """forward of STID"""
        """
        Args:
            history_data (torch.Tensor): history data with shape [B, T, N, C]
        Returns:
            torch.Tensor: prediction with shape [B, T, N, C]
        """
        # prepare data
        # 提取前input_dim个通道作为主特征
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            # STID数据集中时间特征被归一化到[0, 1]
            # 取最后一个时间步的time-of-day比例作为当前时刻标识
            t_i_d_data = history_data[..., self.num_features] # [B, T, N]
            # t_i_d_data[:, -1, :] -> [B, N]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).to(torch.long)]
            # 上一步得到的是浮点数，但嵌入表查询需要整数索引
            # 为每个batch、每个节点、当前时刻，生成一个表示一天中的时间的向量        
            # time_in_day_emb -> [B, N, embed_dim]
        else: 
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., self.num_features + int(self.if_time_in_day)]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).to(torch.long)]
        else:
            day_in_week_emb = None

        # time_series_embedding
        batch_size, seq_len, num_nodes, channels = input_data.size() # [B, T, N, C]
        input_data = input_data.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, seq_len*channels) # [B, N, T*C]
        input_data = input_data.transpose(1, 2).unsqueeze(-1)    # [B, T*C, N, 1]
        time_series_emb = self.time_series_emb_layer(input_data) # [B, embed_dim, N, 1]

        node_emb = []
        node_emb.append(self.node_emb.unsqueeze(0).expand(
                        batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
                        # [B, node_dim, N, 1]
        # expand不会复制数据，而是返回一个新的视图，在指定维度上扩展大小为1的维度
        # -1表示该维度保持原大小不变

        # temporal_embeddings 
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
            # [B, temp_dim_tid, N, 1]
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
            # [B, temp_dim_diw, N, 1]

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1) # [B, hidden_dim, N, 1]
        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden) 

        prediction = prediction.view(batch_size, self.output_length, self.num_features, num_nodes).permute(0, 1, 3, 2).contiguous()
        return prediction
