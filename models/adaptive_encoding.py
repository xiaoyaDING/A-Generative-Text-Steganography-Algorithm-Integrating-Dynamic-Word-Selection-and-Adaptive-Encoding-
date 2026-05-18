# models/adaptive_encoding.py
import torch
import torch.nn as nn

class AdaptiveEncoder(nn.Module):
    """
    自适应编码模块 - 简化版
    当前使用奇偶编码(1比特/词)，保证100%可提取
    未来可扩展为动态区间编码
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 仅当encoding_bits=1时有效，若大于1请重新实现decode逻辑
        assert config.encoding_bits == 1, "当前简化版仅支持1比特编码"

    def encode_bits(self, secret_bits, candidate_probs, context_embeddings=None):
        """
        将1个秘密比特编码到候选词中
        secret_bits: [batch_size, 1] 取值为0或1
        candidate_probs: [batch_size, k] 候选词概率分布
        return: (selected_indices, efficiency)
        """
        batch_size, k = candidate_probs.shape
        device = secret_bits.device

        # 分离偶数位组(索引0,2,4...)和奇数位组(索引1,3,5...)
        even_indices = torch.arange(0, k, 2, device=device)
        odd_indices = torch.arange(1, k, 2, device=device)

        # 提取对应组的概率并归一化
        probs_even = candidate_probs[:, even_indices]  # [batch, len_even]
        probs_odd = candidate_probs[:, odd_indices]    # [batch, len_odd]

        # 处理组为空的情况（很少发生）
        if probs_even.size(1) == 0:
            probs_even = torch.ones(batch_size, 1, device=device) * 1e-10
        if probs_odd.size(1) == 0:
            probs_odd = torch.ones(batch_size, 1, device=device) * 1e-10

        probs_even = probs_even / (probs_even.sum(dim=-1, keepdim=True) + 1e-10)
        probs_odd = probs_odd / (probs_odd.sum(dim=-1, keepdim=True) + 1e-10)

        selected_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            if secret_bits[i, 0] < 0.5:   # 比特0 → 从偶数位中选
                if probs_even.size(1) > 0:
                    idx_in_group = torch.multinomial(probs_even[i], 1).item()
                    selected_indices[i] = even_indices[idx_in_group].item()
                else:
                    selected_indices[i] = 0   # fallback
            else:                          # 比特1 → 从奇数位中选
                if probs_odd.size(1) > 0:
                    idx_in_group = torch.multinomial(probs_odd[i], 1).item()
                    selected_indices[i] = odd_indices[idx_in_group].item()
                else:
                    selected_indices[i] = 1   # fallback

        # 编码效率（简单起见恒为1）
        efficiency = torch.ones(batch_size, device=device)
        return selected_indices.unsqueeze(1), efficiency

    def decode_bits(self, encoded_indices, candidate_probs=None, context_embeddings=None):
        """
        根据选中的索引恢复秘密比特
        encoded_indices: [batch_size, 1] 选中的候选词位置
        return: [batch_size, 1] 恢复的比特
        """
        # 奇偶性决定比特：偶数→0，奇数→1
        bits = (encoded_indices % 2).float()
        return bits

    # 以下两个方法为兼容原接口保留，实际不会调用
    def bits_to_decimal(self, bits): return bits
    def decimal_to_bits(self, decimal, num_bits): return decimal