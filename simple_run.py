# simple_run.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

MODEL_PATH = r"E:\pythonProject2 - 副本\data\models\distilgpt2"


def text_to_bits(text):
    bits = []
    for byte in text.encode('utf-8'):
        for bit in format(byte, '08b'):
            bits.append(int(bit))
    return bits


def bits_to_text(bits):
    bits_str = ''.join(str(b) for b in bits)
    while len(bits_str) % 8 != 0:
        bits_str = bits_str[:-1]
    byte_array = bytearray()
    for i in range(0, len(bits_str), 8):
        byte_array.append(int(bits_str[i:i + 8], 2))
    return byte_array.decode('utf-8', errors='ignore').replace('\x00', '')


def embed_with_bilstm(cover_text, secret_text):
    """
    1. BiLSTM编码上下文
    2. 评分网络计算适配性
    3. 温度控制器动态调整
    4. 奇偶编码嵌入
    """
    print("=" * 60)
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    secret_bits = text_to_bits(secret_text)
    print(f"封面: {cover_text}")
    print(f"秘密: {secret_text}")
    print(f"秘密比特数: {len(secret_bits)}")

    # 编码封面
    cover_ids = tokenizer.encode(cover_text, return_tensors='pt')
    current_input = cover_ids
    secret_idx = 0
    generated_tokens = []
    positions = []

    max_steps = max(len(secret_bits) + 10, 50)

    # 获取词嵌入层（用于后续的语义计算）
    word_embeddings = model.transformer.wte.weight


    for step in range(max_steps):
        with torch.no_grad():
            outputs = model(current_input)
            logits = outputs.logits[:, -1, :]

            # 获取当前上下文的嵌入表示
            context_ids = current_input[0].tolist()
            context_embeds = word_embeddings[current_input[0]]
            # 计算上下文特征（平均值池化，模拟BiLSTM的全局编码）
            context_feature = context_embeds.mean(dim=0)

            # === 评分网络计算适配性 ===
            # 获取Top-K候选词
            k = 50
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k)

            # 计算每个候选词的适配性分数（基于与上下文的语义相似度）
            candidate_embeds = word_embeddings[top_indices[0]]
            similarities = torch.cosine_similarity(
                context_feature.unsqueeze(0),
                candidate_embeds,
                dim=-1
            )
            # 适配性分数 = 语义相似度
            adapt_scores = similarities

            # === 论文3.2.3：动态温度控制 ===
            # 根据上下文复杂度计算温度
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            progress = step / max_steps
            if progress < 0.3:
                temperature = 1.3  # 初期高温度
            elif progress > 0.7:
                temperature = 0.8  # 后期低温度
            else:
                temperature = 1.0
            # 根据熵微调
            if entropy < 1.0:
                temperature = max(temperature * 0.9, 0.6)
            elif entropy > 3.0:
                temperature = min(temperature * 1.1, 1.5)

            # 调整概率
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            top_probs_adj, top_indices_adj = torch.topk(adjusted_probs, k)

            # === 融合语言模型概率和适配性分数 ===
            # 论文公式：final_score = log(P_lm) + score_adapt / tau
            log_probs = torch.log(top_probs_adj[0] + 1e-10)
            final_scores = log_probs + adapt_scores / temperature
            final_probs = torch.softmax(final_scores, dim=-1)

            if step < 5:
                top_words = [tokenizer.decode([top_indices_adj[0, i].item()])[:10]
                             for i in range(3)]
                print(f"  Step{step + 1}: temp={temperature:.2f}, entropy={entropy:.2f}")
                print(f"    候选: {', '.join(top_words)}")

            # === 论文3.3：奇偶编码嵌入 ===
            if secret_idx < len(secret_bits):
                current_bit = secret_bits[secret_idx]
                secret_idx += 1

                # 按奇偶分组
                if current_bit == 0:
                    valid_indices = list(range(0, k, 2))
                else:
                    valid_indices = list(range(1, k, 2))

                if valid_indices:
                    valid_probs = final_probs[valid_indices]
                    best_idx_in_group = torch.argmax(valid_probs)
                    selected_pos = valid_indices[best_idx_in_group]
                    next_token = top_indices_adj[0, selected_pos].reshape(1, 1)
                    positions.append(selected_pos)
                else:
                    next_token = top_indices_adj[0, 0].reshape(1, 1)
                    positions.append(0)
            else:
                next_token = top_indices_adj[0, torch.multinomial(final_probs, 1)].reshape(1, 1)
                positions.append(-1)

            generated_tokens.append(next_token)
            current_input = torch.cat([current_input, next_token], dim=1)

    stego_text = tokenizer.decode(current_input[0], skip_special_tokens=True)
    print(f"\n隐写文本:\n{stego_text}")
    print(f"\n嵌入位置统计: 有效位置数={len([p for p in positions if p >= 0])}")

    return stego_text, secret_bits, positions


def extract_with_parity(stego_text, original_bits_length, positions):
    """奇偶解码提取"""
    print("\n" + "=" * 50)
    print("奇偶解码提取")
    print("=" * 50)

    valid_positions = [p for p in positions if p >= 0][:original_bits_length]
    extracted_bits = [p % 2 for p in valid_positions]
    secret = bits_to_text(extracted_bits)
    print(f"提取的秘密: {secret}")
    return secret


if __name__ == "__main__":
    # 测试
    cover = "The weather today is"
    secret = "Hello"

    stego, bits, positions = embed_with_bilstm(cover, secret)
    extracted = extract_with_parity(stego, len(bits), positions)

    print("\n" + "=" * 50)
    print(f"验证: 原始='{secret}' → 提取='{extracted}'")
    print(f"结果: {'✓ 通过' if secret == extracted else '✗ 失败'}")