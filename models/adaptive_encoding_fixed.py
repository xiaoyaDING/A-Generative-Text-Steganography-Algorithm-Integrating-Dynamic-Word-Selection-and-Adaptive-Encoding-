# models/adaptive_encoding_fixed.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class AdaptiveEncoder(nn.Module):
    """
    自适应编码模块
    根据文本特性自适应调整编码策略
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 编码容量预测器
        self.capacity_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.encoding_bits)
        )

        # 编码掩码生成器
        self.mask_generator = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, config.vocab_size),
            nn.Sigmoid()
        )

        # 噪声注入器（用于增强安全性）
        self.noise_injector = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def encode_bits(self,
                    secret_bits: torch.Tensor,
                    candidate_probs: torch.Tensor,
                    context_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将秘密比特编码到候选词中
        """
        # 确保输入维度正确
        if candidate_probs.dim() == 1:
            candidate_probs = candidate_probs.unsqueeze(0)
        if context_embeddings.dim() == 1:
            context_embeddings = context_embeddings.unsqueeze(0)

        batch_size = candidate_probs.size(0)

        # 预测编码容量
        capacity = self.capacity_predictor(context_embeddings)
        capacity = torch.sigmoid(capacity)

        # 生成编码掩码
        encoding_mask = self.mask_generator(context_embeddings)

        # 调整候选概率
        if encoding_mask.size(1) != candidate_probs.size(1):
            encoding_mask = encoding_mask[:, :candidate_probs.size(1)]

        adjusted_probs = candidate_probs * encoding_mask
        adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # 将秘密比特转换为十进制
        secret_decimal = self.bits_to_decimal(secret_bits)

        # 使用算术编码选择词
        cum_probs = torch.cumsum(adjusted_probs, dim=-1)

        # 找到对应的词索引
        secret_decimal = secret_decimal.unsqueeze(1)
        encoding_mask = (cum_probs >= secret_decimal).float()

        # 获取第一个为1的位置
        encoded_indices = torch.argmax(encoding_mask, dim=-1)

        # 计算编码效率
        batch_indices = torch.arange(batch_size, device=encoded_indices.device)
        actual_probs = adjusted_probs[batch_indices, encoded_indices]
        actual_capacity = torch.log2(actual_probs + 1e-10)
        encoding_efficiency = actual_capacity / (capacity.mean(dim=-1) + 1e-10)

        return encoded_indices, encoding_efficiency

    def decode_bits(self,
                    encoded_indices: torch.Tensor,
                    candidate_probs: torch.Tensor,
                    context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        从编码词中提取秘密比特
        """
        # 确保所有输入都是2D
        if candidate_probs.dim() == 1:
            candidate_probs = candidate_probs.unsqueeze(0)
        if context_embeddings.dim() == 1:
            context_embeddings = context_embeddings.unsqueeze(0)
        if encoded_indices.dim() > 1:
            encoded_indices = encoded_indices.squeeze(-1)

        batch_size = candidate_probs.size(0)

        # 生成相同的编码掩码
        encoding_mask = self.mask_generator(context_embeddings)

        # 调整大小匹配
        if encoding_mask.size(1) != candidate_probs.size(1):
            encoding_mask = encoding_mask[:, :candidate_probs.size(1)]

        adjusted_probs = candidate_probs * encoding_mask
        adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # 计算累积概率
        cum_probs = torch.cumsum(adjusted_probs, dim=-1)

        # 确保cum_probs是2D
        if cum_probs.dim() == 1:
            cum_probs = cum_probs.unsqueeze(0)

        # 获取编码区间的中点值
        batch_indices = torch.arange(batch_size, device=encoded_indices.device)
        prev_probs = torch.zeros(batch_size, device=encoded_indices.device)

        mask = encoded_indices > 0
        if mask.any():
            # 确保索引安全
            safe_indices = torch.clamp(encoded_indices[mask] - 1, 0, cum_probs.size(1) - 1)
            # 使用gather而不是2D索引，更安全
            prev_probs[mask] = torch.gather(
                cum_probs[mask],
                1,
                safe_indices.unsqueeze(1)
            ).squeeze(1)

        # 获取当前概率
        curr_indices = torch.clamp(encoded_indices, 0, cum_probs.size(1) - 1)
        curr_probs = torch.gather(
            cum_probs,
            1,
            curr_indices.unsqueeze(1)
        ).squeeze(1)

        # 计算区间中点
        interval_mid = (prev_probs + curr_probs) / 2

        # 转换回比特
        decoded_bits = self.decimal_to_bits(interval_mid.unsqueeze(-1), self.config.encoding_bits)

        return decoded_bits

    def bits_to_decimal(self, bits: torch.Tensor) -> torch.Tensor:
        """将比特流转换为十进制数"""
        if bits.dim() == 1:
            bits = bits.unsqueeze(0)
        num_bits = bits.size(-1)
        powers = torch.tensor([2 ** (-i - 1) for i in range(num_bits)],
                              device=bits.device)
        decimal = torch.sum(bits * powers, dim=-1)
        return decimal

    def decimal_to_bits(self, decimal: torch.Tensor, num_bits: int) -> torch.Tensor:
        """将十进制数转换为比特流"""
        if decimal.dim() == 1:
            decimal = decimal.unsqueeze(-1)
        decimal = decimal.unsqueeze(-1)
        bits = torch.zeros((*decimal.shape[:-1], num_bits), device=decimal.device)

        for i in range(num_bits):
            bits[..., i] = (decimal * 2).floor()
            decimal = decimal * 2 - bits[..., i:i + 1]

        return bits