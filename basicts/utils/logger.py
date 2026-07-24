import os
import logging
import json

class Logger:
    def __init__(self, log_dir, log_filename='log.txt'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True) # 目录不存在则创建
        self.log_file = os.path.join(log_dir, log_filename)
        
        self.logger = logging.getLogger('STGCN') # 命名日志器
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear() # 如果这段初始化代码被多次调用
                                     # 不清空的话每次都会重复添加handler
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        # 全量日志，格式带时间、级别、消息%(asctime)s-%(levelname)s-%(message)s，适合事后回溯
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        # 简化格式 %(levelname)s - %(message)s
        # 去掉了时间，让训练时的实时输出更清爽，不会被长前缀打乱节奏
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        # 两者等级都设成INFO，意味着logger.debug()不会显示
        
        self.train_metrics = []
        self.val_metrics = []
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def log_train_metrics(self, epoch, loss, metrics):
        log_str = f"Epoch {epoch} - Train Loss: {loss:.6f}"
        # 先把 epoch和loss以及一个metrics字典
        for name, value in metrics.items():
            log_str += f", {name}: {value:.4f}"
        self.info(log_str)
        self.train_metrics.append({'epoch': epoch, 'loss': loss, **metrics})
        # 再以字典形式存入列表，打平了所有指标，每条记录都完整独立
    
    def log_val_metrics(self, epoch, loss, metrics):
        log_str = f"Epoch {epoch} - Val Loss: {loss:.6f}"
        for name, value in metrics.items():
            log_str += f", {name}: {value:.4f}"
        self.info(log_str)
        self.val_metrics.append({'epoch': epoch, 'loss': loss, **metrics})
        # 以及metrics字典里的所有键值对
    
    def save_metrics(self, filename='metrics.json'):
        metrics_path = os.path.join(self.log_dir, filename)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({'train': self.train_metrics, 'val': self.val_metrics}, f, indent=2)
            # 把整个训练过程中的所有train/val指标打包成一个JSON
            # 每个列表里的元素 indent=2让输出的JSON文件带缩进
    def save_best_metrics(self, metrics, filename='best_val_metrics.json'):
        metrics_path = os.path.join(self.log_dir, filename)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            # 单独保存最佳验证集的指标