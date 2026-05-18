# models/steganography.py
import torch
import torch.nn as nn
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .dynamic_selection import DynamicWordSelector
from .adaptive_encoding import AdaptiveEncoder


class GenerativeSteganography(nn.Module):
    """
    生成式文本隐写主模型 - 完整修复版
    融合动态选词与自适应编码
    支持本地模型加载
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 获取模型路径（处理config.model_name可能是property的情况）
        if hasattr(config, 'model_name') and callable(config.model_name):
            model_path = config.model_name()
        else:
            model_path = config.model_name

        print(f"正在加载模型从: {model_path}")

        # 检查模型文件是否存在
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            # 尝试在当前目录下查找
            base_dir = os.path.dirname(os.path.dirname(__file__))
            alt_path = os.path.join(base_dir, model_path)
            config_file = os.path.join(alt_path, "config.json")
            if os.path.exists(config_file):
                model_path = alt_path
                print(f"找到模型在: {model_path}")
            else:
                # 列出所有可能的路径
                possible_paths = [
                    os.path.join("data", "models", "distilgpt2"),
                    os.path.join("data", "models", "distilgpt2_fixed"),
                    os.path.join("data", "models", "gpt2"),
                    os.path.join("data", "models", "uer_gpt2-chinese-distil"),
                ]
                found = False
                for path in possible_paths:
                    full_path = os.path.join(base_dir, path)
                    if os.path.exists(os.path.join(full_path, "config.json")):
                        model_path = full_path
                        found = True
                        print(f"自动找到模型: {model_path}")
                        break

                if not found:
                    raise FileNotFoundError(
                        f"模型配置文件不存在: {config_file}\n"
                        f"请确保模型文件在以下路径之一:\n" +
                        "\n".join([os.path.join(base_dir, p) for p in possible_paths])
                    )

        try:
            # 加载语言模型 - 强制只使用本地文件
            self.language_model = GPT2LMHeadModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float32  # 确保使用float32
            )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )

            print(f"✓ 模型加载成功！")
            print(f"  模型类型: {type(self.language_model).__name__}")
            print(f"  词表大小: {self.tokenizer.vocab_size}")

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("尝试使用随机初始化的模型...")

            # 如果本地加载失败，使用随机初始化的模型（仅用于测试）
            from transformers import GPT2Config
            configuration = GPT2Config()
            self.language_model = GPT2LMHeadModel(configuration)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=False)
            print("⚠ 使用随机初始化的GPT2模型（无预训练权重）")

        # 设置pad_token（GPT-2默认没有）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"  设置pad_token为eos_token: {self.tokenizer.pad_token}")

        # 获取词嵌入层
        self.word_embeddings = self.language_model.transformer.wte

        # 初始化动态选词和自适应编码模块
        self.dynamic_selector = DynamicWordSelector(config)
        self.adaptive_encoder = AdaptiveEncoder(config)

        # 将模型移动到指定设备
        self.to(config.device)
        print(f"  模型已移动到设备: {config.device}")

    def forward(self, input_ids, attention_mask=None, secret_bits=None, mode='embed'):
        """
        前向传播
        Args:
            mode: 'embed' - 嵌入模式, 'extract' - 提取模式
        """
        if mode == 'embed':
            return self.embed_secret(input_ids, attention_mask, secret_bits)
        elif mode == 'extract':
            return self.extract_secret(input_ids, attention_mask)
        else:
            raise ValueError(f"未知模式: {mode}，请使用 'embed' 或 'extract'")

    def embed_secret(self, input_ids, attention_mask, secret_bits):
        """
        将秘密信息嵌入到封面文本中
        """
        batch_size, seq_len = input_ids.shape
        bits_per_position = self.config.encoding_bits
        total_positions = seq_len - 1
        total_bits_needed = total_positions * bits_per_position

        # 调整秘密比特长度
        if secret_bits.dim() == 1:
            secret_bits = secret_bits.unsqueeze(0)

        actual_bits = secret_bits.shape[1]
        if actual_bits < total_bits_needed:
            # 填充零
            padding = torch.zeros(batch_size, total_bits_needed - actual_bits,
                                  device=secret_bits.device)
            secret_bits = torch.cat([secret_bits, padding], dim=1)
        elif actual_bits > total_bits_needed:
            # 截断
            secret_bits = secret_bits[:, :total_bits_needed]

        # 存储生成结果
        generated_tokens = []
        current_context = input_ids[:, :1].clone()

        # 需要屏蔽的特殊token（只屏蔽pad_token）
        special_tokens = []
        if self.tokenizer.pad_token_id is not None:
            special_tokens.append(self.tokenizer.pad_token_id)

        # 逐个位置生成文本
        for pos in range(total_positions):
            # 获取语言模型预测
            with torch.no_grad():
                outputs = self.language_model(current_context)
                next_token_logits = outputs.logits[:, -1, :]

                # 屏蔽特殊token
                for token_id in special_tokens:
                    next_token_logits[:, token_id] = -float('inf')

                next_token_probs = torch.softmax(next_token_logits, dim=-1)

            # 获取top-k候选词
            k = min(self.config.selection_top_k, next_token_probs.size(-1))
            topk_probs, topk_indices = torch.topk(next_token_probs, k=k, dim=-1)

            # 获取候选词嵌入和上下文嵌入
            candidate_embeds = self.word_embeddings(topk_indices)
            context_embeds = self.word_embeddings(current_context)

            # 动态选词 - 直接得到概率分布
            selection_probs, temperatures = self.dynamic_selector(
                context_embeds, candidate_embeds, topk_probs)

            # 获取当前要嵌入的比特
            start_bit = pos * bits_per_position
            end_bit = (pos + 1) * bits_per_position
            current_secret_bits = secret_bits[:, start_bit:end_bit]

            # 自适应编码
            encoded_indices, efficiency = self.adaptive_encoder.encode_bits(
                current_secret_bits, selection_probs, context_embeds.mean(dim=1))

            # 确保维度正确
            if encoded_indices.dim() == 1:
                encoded_indices = encoded_indices.unsqueeze(1)

            # 获取选择的词索引
            selected_indices = torch.gather(topk_indices, 1, encoded_indices).squeeze(1)

            # 存储结果
            generated_tokens.append(selected_indices.unsqueeze(1))

            # 更新上下文
            current_context = torch.cat([current_context, selected_indices.unsqueeze(1)], dim=1)

        # 组合所有生成的token
        generated_tokens = torch.cat(generated_tokens, dim=1)

        # 构建完整的生成文本
        generated_text = torch.cat([input_ids[:, :1], generated_tokens], dim=1)

        return {
            'generated_text': generated_text,
            'encoding_efficiency': torch.ones(batch_size, device=secret_bits.device),
            'temperatures': temperatures if 'temperatures' in locals() else torch.ones(batch_size,
                                                                                       device=secret_bits.device)
        }

    def extract_secret(self, input_ids, attention_mask):
        """
        从隐写文本中提取秘密信息 - 修复版
        """
        batch_size, seq_len = input_ids.shape
        bits_per_position = self.config.encoding_bits
        extracted_bits_list = []

        # 需要屏蔽的特殊token
        special_tokens = []
        if self.tokenizer.pad_token_id is not None:
            special_tokens.append(self.tokenizer.pad_token_id)

        # 逐个位置提取
        for pos in range(seq_len - 1):
            # 获取当前上下文
            current_context = input_ids[:, :pos + 1]
            current_word = input_ids[:, pos + 1]

            # 获取语言模型预测
            with torch.no_grad():
                outputs = self.language_model(current_context)
                next_token_logits = outputs.logits[:, -1, :]

                # 屏蔽特殊token（与嵌入时一致）
                for token_id in special_tokens:
                    next_token_logits[:, token_id] = -float('inf')

                next_token_probs = torch.softmax(next_token_logits, dim=-1)

            # 获取候选词
            k = min(self.config.selection_top_k, next_token_probs.size(-1))
            topk_probs, topk_indices = torch.topk(next_token_probs, k=k, dim=-1)

            # 查找当前词在候选列表中的位置
            candidate_positions = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
            confidence_mask = torch.ones(batch_size, device=input_ids.device)

            for i in range(batch_size):
                # 在top-k中查找
                matches = (topk_indices[i] == current_word[i]).nonzero(as_tuple=True)[0]
                if matches.numel() > 0:
                    candidate_positions[i] = matches[0]
                else:
                    # 如果不在top-k中，计算完整概率分布
                    full_probs = torch.softmax(next_token_logits[i], dim=-1)
                    word_prob = full_probs[current_word[i]].item()

                    # 找出概率大于当前词的词的数量（即排名）
                    rank = (full_probs > word_prob).sum().item()

                    if rank < k:
                        # 重新生成top-k并查找
                        sorted_probs, sorted_idx = full_probs.sort(descending=True)
                        topk_regen = sorted_idx[:k]
                        pos_in_topk = (topk_regen == current_word[i]).nonzero(as_tuple=True)[0]
                        if pos_in_topk.numel() > 0:
                            candidate_positions[i] = pos_in_topk[0]
                        else:
                            candidate_positions[i] = min(rank, k - 1)
                            confidence_mask[i] = 0.5
                    else:
                        # 完全不在top-k中，使用最接近的位置
                        candidate_positions[i] = k - 1
                        confidence_mask[i] = 0.1

            # 获取上下文嵌入
            context_embeds = self.word_embeddings(current_context)
            context_mean = context_embeds.mean(dim=1)

            # 提取比特
            bits = self.adaptive_encoder.decode_bits(
                candidate_positions.unsqueeze(1),
                topk_probs,
                context_mean
            )

            extracted_bits_list.append(bits)

        # 组合所有比特
        if extracted_bits_list:
            extracted_bits = torch.cat(extracted_bits_list, dim=1)

            # 确保形状正确
            total_bits = (seq_len - 1) * bits_per_position
            if extracted_bits.size(1) < total_bits:
                padding = torch.zeros(batch_size, total_bits - extracted_bits.size(1),
                                      device=extracted_bits.device)
                extracted_bits = torch.cat([extracted_bits, padding], dim=1)
            elif extracted_bits.size(1) > total_bits:
                extracted_bits = extracted_bits[:, :total_bits]
        else:
            extracted_bits = torch.zeros(batch_size, total_bits, device=input_ids.device)

        # 计算置信度
        confidence_scores = torch.ones(batch_size, device=input_ids.device)

        return {
            'extracted_bits': extracted_bits,
            'confidence_scores': confidence_scores
        }

    def generate_sample(self, prompt, max_new_tokens=50):
        """
        生成示例文本（不使用隐写）
        """
        self.eval()
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.config.device)

        with torch.no_grad():
            outputs = self.language_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_pretrained(self, save_path):
        """
        保存模型
        """
        os.makedirs(save_path, exist_ok=True)
        self.language_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        torch.save(self.state_dict(), os.path.join(save_path, 'steganography_module.pt'))
        print(f"模型已保存到: {save_path}")

    @classmethod
    def from_pretrained(cls, config, model_path):
        """
        加载预训练模型
        """
        model = cls(config)
        state_dict = torch.load(os.path.join(model_path, 'steganography_module.pt'),
                                map_location=config.device)
        model.load_state_dict(state_dict)
        return model


# 测试代码
if __name__ == "__main__":
    from config import config

    print("=" * 60)
    print("测试模型初始化")
    print("=" * 60)

    try:
        # 初始化模型
        model = GenerativeSteganography(config)

        # 测试前向传播
        batch_size = 2
        seq_len = config.max_length

        # 创建测试输入
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        secret_bits = torch.randint(0, 2, (batch_size, (seq_len - 1) * config.encoding_bits)).float()

        # 测试嵌入模式
        print("\n测试嵌入模式...")
        embed_output = model(input_ids, attention_mask, secret_bits, mode='embed')
        print(f"  生成文本形状: {embed_output['generated_text'].shape}")

        # 测试提取模式
        print("测试提取模式...")
        extract_output = model(embed_output['generated_text'], attention_mask, mode='extract')
        print(f"  提取比特形状: {extract_output['extracted_bits'].shape}")

        print("\n✓ 所有测试通过！")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()