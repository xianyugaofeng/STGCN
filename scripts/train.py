import os
import sys
import argparse
from typing import Dict

import json

# 确保项目根目录在sys.path中，以便正确导入basicts模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import torch
from easytorch import launcher

def parse_args() -> argparse.Namespace:
    # 用来说明这个函数会返回什么类型的数据,用来存放命令行解析后的所有参数
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BasicTS Training Entry')

    # 核心参数: 配置文件路径
    parser.add_argument('-c', '--cfg',
                        type=str,
                        required=True,
                        help='配置文件路径(例如:configs/STGCN_PEMS04.py'
    )

    # 分布式训练参数
    parser.add_argument('--gpus',
                        type=str,
                        default=None,
                        help='指定使用的GPU'
    )

    # 分布式后端
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='分布式训练后端(GPU用nccl，CPU用gloo'
    )

    # 分布式节点数
    parser.add_argument('--num_nodes',
                        type=int,
                        default=1,
                        help='分布式训练的节点数'
    )

    # 是否使用分布式训练
    parser.add_argument('--ddp',
                        action='store_true',
                        help='是否启用分布式训练DistributedDataParallel'
    )

    # 可选: 命令行覆盖配置参数(格式: key=value)
    parser.add_argument('opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='通过命令行覆盖配置文件的参数，例如: TRAIN.NUM_EPOCHS=1 TRAIN.BATCH_SIZE=8'
    )

    return parser.parse_args()

def load_config(cfg_path: str) -> Dict:
    # 加载Python配置文件并返回配置字典
    # cfg_path: 配置文件格式
    import importlib.util

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    # 根据扩展名选择加载方式
    ext = os.path.splitext(cfg_path)[1].lower()
    # 用于将文件路径拆分为主文件名和扩展名两部分 它返回一个元组(root, ext)
    print(f'[INFO]加载配置文件: {cfg_path}')

    if ext == '.json':
        # 直接读取 JSON 文件
        with open(cfg_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return config
    elif ext == '.py':
        # 动态加载Python配置文件
        spec = importlib.util.spec_from_file_location('config', cfg_path)
        # 给模块起一个内部名字 创建一个模块规范ModuleSpec对象
        # 得到一个包含加载器、路径等信息的spec对象
        config_module = importlib.util.module_from_spec(spec)
        # 创建一个空的模块对象
        spec.loader.exec_module(config_module)
        # 真正执行配置文件中的代码，并把所有顶层定义的变量、函数、类都放进config_module的命名空间中
        # 配置文件里写的SEED=42、DATASET_NAME="PEMS04"等，就变成了config_module.SEED、config_module.DATASET_NAME

        # 将模块中的所有大小变量提取为配置字典
        config = {}
        for key in dir(config_module):
            # dir(config_module)列出该模块中所有名字
            if not key.startswith('_'): # not key.startswith('_')过滤掉下划线开头的私有/内置变量，只保留用户定义的配置变量
                config[key] = getattr(config_module, key)
                # 从动态加载的配置模块中，按变量名取出实际值，然后存入配置字典config里

        return config
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}，仅支持 .py 或 .json")

def override_config(config: Dict, opts: list) -> Dict:
    # 使用命令行参数覆盖配置文件中的值,返回更新后的配置字典
    # config: 原始配置字典
    # opts: 命令行覆盖参数列表
    if opts is None:
        return config

    for opt in opts:
        if '=' not in opt:
            continue
        # 将一个形如TRAIN_BATCH_SIZE=8的命令行参数，拆分成键TRAIN_BATCH_SIZE和值8
        # maxsplit=1能保证只从第一个等号处切一刀
        key, value = opt.split('=', 1)
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        config[key] = value
    return config

def main():
    # 主训练过程
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件
    config = load_config(args.cfg)

    # 命令行参数覆盖
    if args.opt:
        print(f'[INFO]应用命令行参数覆盖: {args.cfg}')
        config = override_config(config, args.opts)
