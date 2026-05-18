# models/dynamic_selection.py
import torch
import torch.nn as nn


class DynamicWordSelector(nn.Module):
    """
    动态选词模块
    使用双向LSTM编码上下文，根据上下文动态选择最合适的词进行隐写
    对应论文3.2节
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ========== 双向LSTM上下文编码器（论文3.2节）==========
        self.context_encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # 双向LSTM
        )

        # ========== 评分网络（MLP）==========
        self.scoring_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # ========== 自适应温度控制器（论文3.2节）==========
        self.temperature_controller = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, context_embeddings, candidate_embeddings, candidate_probs):
        """
        Args:
            context_embeddings: [batch_size, seq_len, hidden_size] 上下文词嵌入序列
            candidate_embeddings: [batch_size, k, hidden_size] 候选词嵌入
            candidate_probs: [batch_size, k] 语言模型原始概率
        Returns:
            final_probs: [batch_size, k] 动态调整后的概率分布
            temperatures: [batch_size] 自适应温度值
        """
        batch_size = context_embeddings.size(0)
        k = candidate_embeddings.size(1)

        # ========== 步骤1：双向LSTM编码上下文 ==========
        # 论文公式(3-1): H = BiLSTM(Embed(X))
        context_features, (h_n, c_n) = self.context_encoder(context_embeddings)

        # 获取双向LSTM的最终隐藏状态
        # h_n shape: [num_layers*2, batch, hidden_size//2]
        forward_hidden = h_n[-2, :, :]  # 最后一层前向 [batch, hidden//2]
        backward_hidden = h_n[-1, :, :]  # 最后一层后向 [batch, hidden//2]
        context_feat = torch.cat([forward_hidden, backward_hidden], dim=-1)  # [batch, hidden]

        # ========== 步骤2：计算候选词适配性分数 ==========
        # 扩展上下文特征到每个候选词
        context_expanded = context_feat.unsqueeze(1).expand(-1, k, -1)  # [batch, k, hidden]

        # 拼接上下文特征和候选词嵌入
        combined = torch.cat([context_expanded, candidate_embeddings], dim=-1)  # [batch, k, hidden*2]

        # 评分网络计算原始分数 s_i = MLP(f_i)
        raw_scores = self.scoring_network(combined).squeeze(-1)  # [batch, k]

        # ========== 步骤3：自适应温度控制 ==========
        # 论文公式(3-3): τ = 0.5 + 1.5 × σ(MLP_τ(h_ctx))
        context_avg = context_embeddings.mean(dim=1)  # [batch, hidden]
        base_temp = self.temperature_controller(context_avg)  # [batch, 1]
        temperatures = 0.5 + 1.5 * base_temp  # 范围[0.5, 2.0]

        # 温度调节
        adjusted_scores = raw_scores / temperatures

        # ========== 步骤4：融合语言模型概率 ==========
        log_probs = torch.log(candidate_probs + 1e-10)
        final_scores = adjusted_scores + log_probs

        # ========== 步骤5：输出概率分布 ==========
        final_probs = torch.softmax(final_scores, dim=-1)

        return final_probs, temperatures.squeeze(-1)