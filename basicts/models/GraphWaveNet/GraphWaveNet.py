import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NConv(nn.Module):
    # 图卷积操作：在节点维度上做矩阵乘法，聚合邻居特征
    def __init__(self):
        super(NConv, self).__init__()
    
    def forward(self, x, A):
        # 对每个(batch, channel, time_step)切片 在节点维度做x @ A
        x = x.permute(0, 1, 3, 2) # [B, C, N, T] -> [B, C, T, N]
        x = torch.matmul(x, A)
        x = x.permute(0, 1, 2, 3) # [B, C, N, T]
        return x.contiguous()

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, support_length=3, K=2, dropout=0.3):
        super(GCN, self).__init__()
        self.NConv = NConv()
        in_channels = (K * support_length + 1) * in_channels
        self.MLP = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)
        self.K = K
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, supports):
        out = [x]
        for A in supports:
            x1 = self.NConv(x, A)
            out.append(x1)
            for i in range(2, self.K + 1):
                x2 = self.NConv(x1, A)
                out.append(x2)
                x1 = x2
        
        # 沿通道维度聚合
        hidden = torch.cat(out, dim=-1)
        hidden = self.dropout(self.MLP(hidden))
        return hidden

class GraphWaveNet(nn.Module):
    def __init__(self, supports=[], aptinit=None, dropout=0.3, gcn_bool=True, input_dim=2, addaptadj=None,
                 output_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, 
                 kernel_size=2, blocks=2, layers=2, num_nodes=325, input_length=12, output_length=12, embed_dim=10):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.output_length = output_length

        self.supports = supports
        self.dropout = nn.Dropout(p=dropout)
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconvs = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=input_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.support_length = 0
        if self.supports is not None:
            self.support_length += len(self.supports)
        else:
            self.supports = []
        
        if self.gcn_bool and self.addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.rand(self.num_nodes, embed_dim), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.rand(embed_dim, self.num_nodes), requires_grad=True)
                self.support_length += 1
            else:
                # SVD on aptinit
                U, S, Vh = torch.linalg.svd(aptinit)
                # 用SVD把预定义邻接矩阵分解为两个低秩嵌入矩阵的乘积
                initemb1 = torch.matmul(U[:, :10], torch.diag(S[:10] ** 0.5))
                initemb2 = torch.matmul(torch.diag(S[:10] ** 0.5), Vh[:10, :])
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.support_length += 1
        
        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilation convs - operate on residual_channels (output of start_conv)
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, 
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, 
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                
                if self.gcn_bool:
                    self.gconvs.append(GCN(in_channels=dilation_channels, out_channels=dilation_channels, 
                                          support_length=self.support_length, dropout=dropout, K=2))
                
                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        
        self.end_conv1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)

        self.end_conv2 = nn.Conv2d(in_channels=end_channels, out_channels=output_dim, kernel_size=(1, 1), bias=True)

        self.receptive_field = receptive_field
    
    def forward(self, x):
        """
        args: 
            x: [B, T, N, C]
            supports: optional list of support matrices for GCN
        """
        
        input_length = x.size(1)
        x = x.permute(0, 3, 2, 1).contiguous() # [B, C, N, T]
        if input_length < self.receptive_field:
            x = F.pad(x, (self.receptive_field - input_length, 0, 0, 0))
        else:
            pass

        x = self.start_conv(x)
        skip = 0

        # 增加自适应邻接矩阵
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports:
            adp = F.softmax(F.relu(torch.matmul(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        
        for i in range(self.blocks * self.layers):
            # dilation conv
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.sigmoid(filter)
            gate = self.gate_convs[i](x)
            gate = torch.tanh(gate)
            x = filter * gate

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = skip + s

            # GCN 
            if self.gcn_bool and self.supports:
                if self.addaptadj:
                    x = self.gconvs[i](x, new_supports)
                else:
                    x = self.gconvs[i](x, self.supports)
            
            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)
        
        x = F.relu(skip)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)

        x = x.permute(0, 3, 2, 1).contiguous() # [B, C, N, T] -> [B, T, N, C]
        return x

