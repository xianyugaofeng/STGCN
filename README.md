# STGCN
STCGN/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包列表
├── setup.py                           # 安装脚本（可选）
│
├── configs/                           # ⭐ 配置文件目录（smoke run 主要修改这里）
│   ├── STGCN_PEMS04.json              # STGCN + PEMS04 标准配置
│   └── STGCN_PEMS04_smoke.json        # ⭐ Smoke Run 配置（新建）
│
├── scripts/                           # 入口脚本目录
│   ├── train.py                       # 训练入口脚本
│   └── test.py                        # 测试入口脚本（可选）
│
├── basicts/                           # 框架核心代码（不修改）
│   ├── __init__.py
│   ├── runners/                       # Runner 模块
│   │   ├── __init__.py
│   │   ├── base_runner.py             # 基础训练/验证/测试逻辑
│   │   └── runner_zoo.py              # 各种 Runner 实现
│   │
│   ├── models/                        # 模型定义
│   │   ├── __init__.py
│   │   └── STGCN/                     # STGCN 模型
│   │       ├── __init__.py
│   │       └── stgcn.py               # STGCN 核心实现（不修改）
│   │
│   ├── datasets/                      # 数据集处理
│   │   ├── __init__.py
│   │   └── dataset_zoo.py             # 数据集加载器
│   │
│   ├── losses/                        # 损失函数
│   │   ├── __init__.py
│   │   └── loss_zoo.py                # MAE, MSE, MAPE 等
│   │
│   ├── metrics/                       # 评估指标
│   │   ├── __init__.py
│   │   └── metric_zoo.py              # MAE, RMSE, MAPE 计算
│   │
│   └── utils/                         # 工具函数
│       ├── __init__.py
│       ├── config.py                  # 配置加载工具
│       ├── logger.py                  # 日志工具
│       └── data_utils.py              # 数据处理工具
│
├── datasets/                          # 数据集目录
│   └── PEMS04/                        # PEMS04 数据集
│       ├── PEMS04.npz                 # 主数据文件 (307节点, ~16000时间步, 3特征)
│       ├── adj_PEMS04.pkl             # 邻接矩阵（二进制）
│       ├── adj_PEMS04_distance.pkl    # 距离加权邻接矩阵
│       └── PEMS04.csv                 # 原始CSV数据（可选）
│
└── outputs/                           # 输出目录（自动创建）
    └── smoke_STGCN_PEMS04/            # Smoke Run 输出
        ├── log.txt                    # 训练日志
        └── best_val_metrics.json      # 验证指标（可选）