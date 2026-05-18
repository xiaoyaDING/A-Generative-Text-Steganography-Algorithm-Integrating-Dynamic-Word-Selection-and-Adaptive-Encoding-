# experiments/detection_experiment.py - 修复版
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed import embed_secret, extract_secret, text_to_bits, bits_to_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = r"E:\pythonProject2 - 副本\data\models\distilgpt2"


# ========== Baseline 方法 ==========

def fixed_length_embed_simple(probs, secret_bit):
    """定长编码：根据比特选择组，组内选概率最高的"""
    k = len(probs)
    group_size = max(1, k // 2)
    if secret_bit == 0:
        start_idx = 0
    else:
        start_idx = group_size
    end_idx = min(start_idx + group_size, k)
    group_probs = probs[start_idx:end_idx]
    best_in_group = np.argmax(group_probs)
    return start_idx + best_in_group


def generate_with_baseline(cover_text, secret_text, method='fixed'):
    """使用Baseline方法生成隐写文本"""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    secret_bits = text_to_bits(secret_text)
    cover_ids = tokenizer.encode(cover_text, return_tensors='pt')
    current_input = cover_ids
    secret_idx = 0
    generated_tokens = []

    temperature = 0.9
    top_k = 50

    for step in range(50):
        with torch.no_grad():
            outputs = model(current_input)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            k = min(top_k, probs.shape[-1])
            top_probs, top_indices = torch.topk(probs, k)
            probs_list = top_probs[0].cpu().numpy()

            if secret_idx < len(secret_bits):
                current_bit = secret_bits[secret_idx]
                secret_idx += 1

                if method == 'fixed':
                    selected_pos = fixed_length_embed_simple(probs_list, current_bit)
                else:  # huffman
                    selected_pos = np.argmax(probs_list)

                next_token = top_indices[0, selected_pos].reshape(1, 1)
            else:
                next_token = torch.multinomial(probs, 1).reshape(1, 1)

            generated_tokens.append(next_token)
            current_input = torch.cat([current_input, next_token], dim=1)

    return tokenizer.decode(current_input[0], skip_special_tokens=True)


def extract_statistical_features(text, tokenizer):
    """提取文本的统计特征"""
    tokens = tokenizer.encode(text)

    if len(tokens) == 0:
        return {'avg_token_length': 0, 'token_frequency_entropy': 0, 'repetition_rate': 0, 'unique_token_ratio': 0}

    # 平均token长度（字符数）
    avg_len = np.mean([len(tokenizer.decode([t])) for t in tokens])

    # 词频熵
    token_counter = Counter(tokens)
    probs = np.array(list(token_counter.values())) / len(tokens)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # 重复率
    repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    repetition = repeats / len(tokens) if len(tokens) > 0 else 0

    # 唯一词比例
    unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

    return {
        'avg_token_length': avg_len,
        'token_frequency_entropy': entropy,
        'repetition_rate': repetition,
        'unique_token_ratio': unique_ratio
    }


def detection_experiment_with_baseline():
    """实验：抗检测能力实验（含Baseline对比）"""
    print("=" * 60)
    print("实验4：抗检测能力实验（含Baseline对比）")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_samples = 10  # 减少样本加速
    prompts = [
                  "The weather today is",
                  "Deep learning is",
                  "Natural language",
                  "Artificial intelligence",
                  "Machine learning"
              ] * (n_samples // 5 + 1)
    prompts = prompts[:n_samples]

    secret = "Hello"

    # 存储各方法生成的文本
    all_texts = {
        'natural': [],
        'our_method': [],
        'fixed_length': [],
        'huffman': []
    }

    print("\n生成测试文本...")

    for prompt in tqdm(prompts, desc="生成文本"):
        # 自然文本
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=30, do_sample=True, temperature=0.8)
                natural = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_texts['natural'].append(natural)
        except Exception as e:
            print(f"自然文本生成错误: {e}")
            all_texts['natural'].append(prompt)

        # 本文方法
        try:
            stego, _, _ = embed_secret(prompt, secret)
            all_texts['our_method'].append(stego)
        except Exception as e:
            print(f"本文方法错误: {e}")
            all_texts['our_method'].append(prompt)

        # 定长编码
        try:
            stego_fixed = generate_with_baseline(prompt, secret, method='fixed')
            all_texts['fixed_length'].append(stego_fixed)
        except Exception as e:
            print(f"定长编码错误: {e}")
            all_texts['fixed_length'].append(prompt)

        # 哈夫曼编码
        try:
            stego_huffman = generate_with_baseline(prompt, secret, method='huffman')
            all_texts['huffman'].append(stego_huffman)
        except Exception as e:
            print(f"哈夫曼编码错误: {e}")
            all_texts['huffman'].append(prompt)

    # 提取特征
    print("\n提取统计特征...")
    features = {}
    for method in all_texts:
        method_features = []
        for text in all_texts[method]:
            method_features.append(extract_statistical_features(text, tokenizer))
        features[method] = method_features

    feature_names = ['avg_token_length', 'token_frequency_entropy', 'repetition_rate', 'unique_token_ratio']
    feature_names_cn = ['平均Token长度', '词频熵', '重复率', '唯一词比例']

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    methods = ['natural', 'our_method', 'fixed_length', 'huffman']
    method_labels = ['自然文本', '本文方法', '定长编码', '哈夫曼编码']
    colors = ['#2E8B57', '#2E86AB', '#F18F01', '#A23B72']

    for i, (feat, feat_cn) in enumerate(zip(feature_names, feature_names_cn)):
        ax = axes[i // 2, i % 2]

        data = []
        for method in methods:
            # 修复：正确提取特征值列表
            vals = [f[feat] for f in features[method]]
            data.append(vals)

        bp = ax.boxplot(data, labels=method_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # 计算p值（与自然文本对比）
        p_values = {}
        natural_vals = [f[feat] for f in features['natural']]

        for j, method in enumerate(['our_method', 'fixed_length', 'huffman']):
            method_vals = [f[feat] for f in features[method]]
            t_stat, p_val = stats.ttest_ind(natural_vals, method_vals)
            p_values[method] = p_val

        ax.set_title(
            f'{feat_cn}\n本文方法p={p_values["our_method"]:.4f} | 定长p={p_values["fixed_length"]:.4f} | 哈夫曼p={p_values["huffman"]:.4f}')
        ax.set_ylabel('值')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/detection_comparison.png', dpi=300)
    plt.show()

    # 打印结果表格
    print("\n" + "=" * 70)
    print("统计检测结果对比")
    print("=" * 70)
    print(f"{'特征':<20} {'方法':<12} {'均值±标准差':<25} {'p值':<10} {'显著性'}")
    print("-" * 70)

    for feat, feat_cn in zip(feature_names, feature_names_cn):
        # 自然文本
        natural_vals = [f[feat] for f in features['natural']]
        print(
            f"{feat_cn:<20} {'自然文本':<12} {np.mean(natural_vals):.4f}±{np.std(natural_vals):.4f} {'-':<10} {'-':<10}")

        for method, label in zip(['our_method', 'fixed_length', 'huffman'], ['本文方法', '定长编码', '哈夫曼编码']):
            method_vals = [f[feat] for f in features[method]]
            t_stat, p_val = stats.ttest_ind(natural_vals, method_vals)
            result = "✓ 无差异" if p_val > 0.05 else "✗ 有差异"
            print(
                f"{feat_cn:<20} {label:<12} {np.mean(method_vals):.4f}±{np.std(method_vals):.4f} {p_val:.4f}    {result}")
        print("-" * 70)

    print("\n结论：")
    print("- 如果本文方法的p值 > 0.05，说明与自然文本无显著差异，抗检测能力强")
    print("- 如果Baseline方法的p值 < 0.05，说明容易被检测")

    return features


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    detection_experiment_with_baseline()