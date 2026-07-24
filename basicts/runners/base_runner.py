import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import json

class BaseRunner:
    # 把训练过程中所有需要反复编写的通用逻辑
    def __init__(self, config):
        self.config = config
        self.device = self._init_device()
        self.model = self._init_model() 
        # _init_model()：构建具体的网络结构并移动到self.device
        self.optimizer = self._init_optimizer()
        # _init_optimizer()：根据配置创建优化器
        self.criterion = self._init_criterion()
        # _init_criterion()：定义损失函数
        self.scheduler = self._init_scheduler()
        # _init_scheduler()：学习率调度器
        self.adj_matrix = None
        
        from basicts.utils import Logger
        self.log_dir = config.get('LOG_DIR', 'outputs/STGCN_PEMS04')
        self.logger = Logger(self.log_dir)
        
        self.num_epochs = config.get('TRAIN_NUM_EPOCHS', 100)
        self.print_freq = config.get('PRINT_FREQ', 10)
        self.early_stopping_patience = config.get('TRAIN_EARLY_STOPPING_PATIENCE', 20)
        self.save_model = config.get('SAVE_MODEL', True)
        
        from basicts.metrics import compute_metrics
        self.metric_names = config.get('METRICS', ['MAE', 'RMSE', 'MAPE'])
        self.compute_metrics = compute_metrics
        
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.patience_counter = 0
    
    def _init_device(self):
        device_str = self.config.get('DEVICE', 'cuda:0')
        if torch.cuda.is_available():
            device = torch.device(device_str)
            self.logger.info(f"Using GPU: {device_str}")
        else:
            device = torch.device('cpu')
            self.logger.info("CUDA not available, using CPU")
        return device
    
    def _init_model(self):
        from basicts.models import get_model
        
        model_name = self.config.get('MODEL_NAME', 'STGCN')
        model_args = self.config.get('MODEL_ARGS', {})
        
        model_args['num_nodes'] = self.config.get('NUM_NODES', 307)
        model_args['num_features'] = self.config.get('NUM_FEATURES', 3)
        model_args['input_length'] = self.config.get('INPUT_LENGTH', 12)
        model_args['output_length'] = self.config.get('OUTPUT_LENGTH', 12)
        
        model = get_model(model_name)(**model_args) # (**model_args)实例化
        model = model.to(self.device) # 将模型参数和缓冲区移动到CPU/GPU
        
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Model args: {model_args}")
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        # 而且只统计model.parameters()
        
        return model
    
    def _init_optimizer(self):
        lr = self.config.get('TRAIN_LEARNING_RATE', 0.001)
        weight_decay = self.config.get('TRAIN_WEIGHT_DECAY', 0.0001)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer  # 对self.model的所有可训练参数创建Adam优化器
    
    def _init_criterion(self):
        from basicts.losses import get_loss
        loss_name = self.config.get('LOSS_FUNCTION', 'masked_mae')
        criterion = get_loss(loss_name)
        return criterion
    
    def _init_scheduler(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        # factor: 当指标停止改善时，学习率会乘以这个因子
        # patience: 允许连续多少个epoch验证指标没有改善，之后才触发学习率下降
        # verbose=True: 触发学习率下降时，会打印一条日志
        return scheduler

        # 创建了一个ReduceLROnPlateau调度器，并绑定到之前创建的优化器self.optimizer
    
    def _train_epoch(self, train_loader):
        # 一个PyTorch DataLoader，每次迭代输出(x, y)，即输入序列和真值
        self.model.train() # 将模型切换到训练模式，启用Dropout、BatchNorm的统计量更新
        total_loss = 0.0
        total_preds = []
        total_targets = []
        # 用列表收集所有的预测和目标 最后在epoch结束时一次性concat，用于计算整个epoch的评估指标
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        # tqdm提供了友好的进度条，leave=False 表示该epoch结束后进度条会消失

        for batch_idx, (x, y) in enumerate(progress_bar):
            # 同时通过enumerate获取batch_idx，便于后续判断是否到了打印频率
            # 每次迭代都会吐出一个形状为(BATCH_SIZE, ...)的张量x和y
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)
            # 将数据移到GPU/CPU
            # 同时强制转为float32，避免因数据加载时产生float64导致后续计算慢或报错

            self.optimizer.zero_grad()
            # 每次迭代必须清空梯度，否则会累加
            laplacian = None
            if self.adj_matrix is not None:
                laplacian = self._compute_laplacian(self.adj_matrix).to(self.device)
                # 如果图邻接矩阵 adj_matrix 存在，就动态计算拉普拉斯并搬到设备上
            pred = self.model(x, laplacian)
            loss = self.criterion(pred, y)
            
            loss.backward()
            self.optimizer.step()
            # 标准的梯度回传与权重更新
            
            total_loss += loss.item() * x.size(0)
            # loss.item() * x.size(0)将当前batch的总损失加进去
            total_preds.append(pred.detach().cpu())
            total_targets.append(y.detach().cpu())
            # .detach().cpu()切断梯度并将张量移到CPU，防止占用GPU显存
            # 同时方便后续 numpy 计算指标
            if (batch_idx + 1) % self.print_freq == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader.dataset)
        # 总损失除以数据集总样本数，得到整个epoch的平均损失
        total_preds = torch.cat(total_preds, dim=0)
        total_targets = torch.cat(total_targets, dim=0)
        # 沿batch维度dim=0拼接，获得整个epoch的完整预测矩阵和目标矩阵
        metrics = self.compute_metrics(total_preds, total_targets, self.metric_names)
        
        return avg_loss, metrics
    
    def _val_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_preds = []
        total_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                
                laplacian = None
                if self.adj_matrix is not None:
                    laplacian = self._compute_laplacian(self.adj_matrix).to(self.device)
                
                pred = self.model(x, laplacian)
                loss = self.criterion(pred, y)
                
                total_loss += loss.item() * x.size(0)
                total_preds.append(pred.detach().cpu())
                total_targets.append(y.detach().cpu())
        
        avg_loss = total_loss / len(val_loader.dataset)
        total_preds = torch.cat(total_preds, dim=0)
        total_targets = torch.cat(total_targets, dim=0)
        metrics = self.compute_metrics(total_preds, total_targets, self.metric_names)
        
        return avg_loss, metrics
    
    def _compute_laplacian(self, adj_matrix):
        # 原始的邻接矩阵（图的拓扑结构）转换为一个对称归一化矩阵，送入后续的图卷积层使用
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
            # 如果是numpy数组就自动转成torch.float32
        
        d = adj_matrix.sum(dim=1)
        # sum(dim=1)对每一行求和，得到一个长度为N的向量d，d[i]就是节点i的度
        d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
        # 加入一个极小的ε，防止某个节点的度恰好为0导致除零错误
        # 度矩阵的-1/2次方的对角线值
        d_sqrt_inv = torch.diag(d_sqrt_inv)
        # torch.diag把一个向量变成[N, N]的方阵
        # 只有主对角线有值，其余为0。它等价于矩阵 D^{-1/2}
        laplacian = torch.matmul(torch.matmul(d_sqrt_inv, adj_matrix), d_sqrt_inv)
        
        return laplacian
    
    def _save_model(self, epoch):
        checkpoint_path = os.path.join(self.log_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),    # 模型权重
            # 最核心的权重，加载后可直接用于推理或继续训练
            'optimizer_state_dict': self.optimizer.state_dict(),    # 优化器状态(动量，二阶矩)
            # Adam等优化器内部维护了每个参数的历史梯度信息如动量
            # 如果不保存，中断后重训相当于优化器从零开始，会破坏学习动态，可能导致收敛变差
            'best_val_loss': self.best_val_loss,            # 到目前位置的最佳验证损失
            # 方便恢复训练循环的状态，明确当前进度以及早停相关的历史最优信息
            'config': self.config                           # 完整实验配置
        }, checkpoint_path)
        self.logger.info(f"Model saved to {checkpoint_path}")
    
    def run(self, train_loader, val_loader=None, test_loader=None):
        self.logger.info("=" * 60)
        self.logger.info(f"Training started at {datetime.now()}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
        self.logger.info("=" * 60)
        
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"\n--- Epoch {epoch}/{self.num_epochs} ---")
            
            train_loss, train_metrics = self._train_epoch(train_loader)
            self.logger.log_train_metrics(epoch, train_loss, train_metrics)
            
            if val_loader is not None:
                val_loss, val_metrics = self._val_epoch(val_loader)
                self.logger.log_val_metrics(epoch, val_loss, val_metrics)
                
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_metrics = val_metrics
                    self.patience_counter = 0
                    
                    if self.save_model:
                        self._save_model(epoch)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience and self.early_stopping_patience > 0:
                        self.logger.info(f"Early stopping triggered after {epoch} epochs")
                        break
            
            self.logger.info("-" * 60)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training completed")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best validation metrics: {self.best_val_metrics}")
        
        self.logger.save_best_metrics(self.best_val_metrics)
        
        if test_loader is not None:
            self.logger.info("\n--- Testing ---")
            test_loss, test_metrics = self._val_epoch(test_loader)
            self.logger.info(f"Test Loss: {test_loss:.6f}")
            for name, value in test_metrics.items():
                self.logger.info(f"Test {name}: {value:.4f}")
        
        self.logger.info(f"Training finished at {datetime.now()}")
        self.logger.info("=" * 60)