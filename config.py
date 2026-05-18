# config.py配置文件
import torch
import os
import random
import numpy as np


def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # 基础配置
    project_name = "融合动态选词与自适应编码的生成式文本隐写算法"
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据配置
    vocab_size = 50257  # GPT-2的词表大小
    max_length = 128  # 英文文本配置

    # 模型配置 - 自动检测本地模型
    @property
    def model_name(self):
        # 可能的模型路径（按优先级）
        possible_paths = [
            os.path.join("data", "models", "distilgpt2"),
            os.path.join("data", "models", "distilgpt2_fixed"),
            os.path.join("data", "models", "gpt2"),
            os.path.join("data", "models", "uer_gpt2-chinese-distil"),
        ]

        for path in possible_paths:
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                print(f"找到本地模型: {path}")
                return path

        # 如果都没有找到，返回默认路径并警告
        default_path = os.path.join("data", "models", "distilgpt2")
        print(f"警告: 未找到本地模型，使用默认路径: {default_path}")
        return default_path

    hidden_size = 768
    vocab_size = 50257

    # 动态选词配置
    selection_window = 5
    selection_top_k = 50
    selection_temperature = 1.5

    # 自适应编码配置 - 强制1比特确保可提取
    encoding_bits = 1
    adaptive_threshold = 0.3

    # 训练配置
    batch_size = 2
    learning_rate = 2e-5
    num_epochs = 10
    warmup_steps = 100
    max_grad_norm = 1.0

    # 路径配置
    @property
    def data_dir(self):
        return os.path.join(os.path.dirname(__file__), "data")

    @property
    def model_dir(self):
        return os.path.join(os.path.dirname(__file__), "models")

    @property
    def checkpoint_dir(self):
        return os.path.join(os.path.dirname(__file__), "checkpoints")

    def __init__(self):
        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # 设置随机种子
        set_seed(self.seed)


config = Config()

if __name__ == "__main__":
    print("=" * 60)
    print(f"项目: {config.project_name}")
    print(f"设备: {config.device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"模型: {config.model_name}")
    print(f"最大序列长度: {config.max_length}")
    print(f"批次大小: {config.batch_size}")
    print(f"编码比特数: {config.encoding_bits}")
    print("=" * 60)