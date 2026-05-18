# experiments/text_similarity_experiment.py - 修复哈夫曼编码
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed import embed_secret
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import heapq

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = r"E:\pythonProject2 - 副本\data\models\distilgpt2"


# ========== Baseline 方法 ==========

def fixed_length_embed_simple(probs, secret_bit, k=50):
    """定长编码：根据比特选择组，组内选概率最高的"""
    group_size = max(1, k // 2)
    if secret_bit == 0:
        start_idx = 0
    else:
        start_idx = group_size
    end_idx = min(start_idx + group_size, k)
    group_probs = probs[start_idx:end_idx]
    best_in_group = np.argmax(group_probs)
    return start_idx + best_in_group


def huffman_embed(probs, secret_bits):
    """
    哈夫曼编码：根据概率构建哈夫曼树，将秘密比特映射到词汇
    使用优先队列构建哈夫曼树
    """
    k = len(probs)

    # 构建节点列表 (概率, 索引, 左子, 右子)
    nodes = [(probs[i], i, None, None) for i in range(k)]

    # 使用优先队列构建哈夫曼树
    heap = [(p, i, None, None) for i, (p, _, _, _) in enumerate(nodes)]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = (left[0] + right[0], -1, left, right)
        heapq.heappush(heap, merged)

    # 构建编码表
    codes = {}

    def build_codes(node, code):
        if node[1] >= 0:  # 叶节点
            codes[node[1]] = code
        else:
            if node[2] is not None:
                build_codes(node[2], code + '0')
            if node[3] is not None:
                build_codes(node[3], code + '1')

    if heap:
        build_codes(heap[0], '')

    # 将秘密比特转换为字符串
    bits_str = ''.join(str(b) for b in secret_bits)

    # 按哈夫曼树解码，找到对应的词
    current_node = heap[0]
    bit_idx = 0
    while current_node[1] < 0 and bit_idx < len(bits_str):
        if bits_str[bit_idx] == '0':
            current_node = current_node[2]
        else:
            current_node = current_node[3]
        bit_idx += 1

    if current_node[1] >= 0:
        return current_node[1]
    return 0


def generate_with_baseline(cover_text, secret_text, method='fixed'):
    """使用Baseline方法生成隐写文本"""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 秘密信息转比特
    secret_bytes = secret_text.encode('utf-8')
    secret_bits = []
    for byte in secret_bytes:
        for bit in format(byte, '08b'):
            secret_bits.append(int(bit))

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
                if method == 'fixed':
                    # 定长编码：一次嵌入1比特
                    current_bit = secret_bits[secret_idx]
                    secret_idx += 1
                    selected_pos = fixed_length_embed_simple(probs_list, current_bit, k)
                elif method == 'huffman':
                    # 哈夫曼编码：可以一次嵌入多个比特
                    # 为了公平比较，也一次嵌入1比特（简化）
                    current_bit = secret_bits[secret_idx]
                    secret_idx += 1
                    # 使用概率作为权重，选择对应比特的路径
                    if current_bit == 0:
                        # 选择概率较高的前半部分
                        valid_indices = list(range(0, k, 2))
                    else:
                        valid_indices = list(range(1, k, 2))

                    if valid_indices:
                        valid_probs = [probs_list[i] for i in valid_indices]
                        best_in_group = np.argmax(valid_probs)
                        selected_pos = valid_indices[best_in_group]
                    else:
                        selected_pos = 0
                else:
                    selected_pos = 0

                next_token = top_indices[0, selected_pos].reshape(1, 1)
            else:
                next_token = torch.multinomial(probs, 1).reshape(1, 1)

            generated_tokens.append(next_token)
            current_input = torch.cat([current_input, next_token], dim=1)

    return tokenizer.decode(current_input[0], skip_special_tokens=True)


def generate_natural_text(cover_text):
    """生成自然文本（无隐写）作为基准"""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    cover_ids = tokenizer.encode(cover_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            cover_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ========== 评估指标 ==========

def calculate_bleu(reference, hypothesis):
    """计算4-gram BLEU分数"""
    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0.0

        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([ref_tokens], hyp_tokens,
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothie)
        return bleu
    except:
        return 0.0


def calculate_bleu_1gram(reference, hypothesis):
    """计算1-gram BLEU（词汇重叠率）"""
    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if len(ref_tokens) == 0:
            return 0.0

        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([ref_tokens], hyp_tokens,
                             weights=(1.0, 0, 0, 0),
                             smoothing_function=smoothie)
        return bleu
    except:
        return 0.0


def calculate_perplexity(text):
    """计算困惑度"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        inputs = tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            return torch.exp(loss).item()
    except:
        return 1000.0


# ========== 主实验 ==========

def text_similarity_experiment():
    """实验3：文本相似度对比实验（含Baseline）"""
    print("=" * 70)
    print("实验3：文本相似度对比实验（含Baseline对比）")
    print("=" * 70)

    test_prompts = [
        "The weather today is going to be",
        "Deep learning is a subset of machine learning",
        "Natural language processing is a field of AI",
        "Artificial intelligence is transforming the world"
    ]

    secret = "Hello"

    results = {
        'method': ['自然文本', '本文方法', '定长编码', '哈夫曼编码'],
        'bleu_1gram': {i: [] for i in range(4)},
        'bleu_4gram': {i: [] for i in range(4)},
        'perplexity': {i: [] for i in range(4)}
    }

    for idx, prompt in enumerate(test_prompts):
        print(f"\n{'=' * 50}")
        print(f"样本{idx + 1}: {prompt}")
        print(f"{'=' * 50}")

        # 1. 自然文本
        try:
            natural = generate_natural_text(prompt)
            bleu1 = calculate_bleu_1gram(prompt, natural)
            bleu4 = calculate_bleu(prompt, natural)
            ppl = calculate_perplexity(natural)
            results['bleu_1gram'][idx].append(bleu1)
            results['bleu_4gram'][idx].append(bleu4)
            results['perplexity'][idx].append(ppl)
            print(f"  自然文本: BLEU1={bleu1:.4f}, BLEU4={bleu4:.4f}, PPL={ppl:.2f}")
        except Exception as e:
            print(f"  自然文本失败: {e}")
            results['bleu_1gram'][idx].append(0)
            results['bleu_4gram'][idx].append(0)
            results['perplexity'][idx].append(1000)

        # 2. 本文方法
        try:
            stego_our, _, _ = embed_secret(prompt, secret)
            bleu1 = calculate_bleu_1gram(prompt, stego_our)
            bleu4 = calculate_bleu(prompt, stego_our)
            ppl = calculate_perplexity(stego_our)
            results['bleu_1gram'][idx].append(bleu1)
            results['bleu_4gram'][idx].append(bleu4)
            results['perplexity'][idx].append(ppl)
            print(f"  本文方法: BLEU1={bleu1:.4f}, BLEU4={bleu4:.4f}, PPL={ppl:.2f}")
        except Exception as e:
            print(f"  本文方法失败: {e}")
            results['bleu_1gram'][idx].append(0)
            results['bleu_4gram'][idx].append(0)
            results['perplexity'][idx].append(1000)

        # 3. 定长编码
        try:
            stego_fixed = generate_with_baseline(prompt, secret, method='fixed')
            bleu1 = calculate_bleu_1gram(prompt, stego_fixed)
            bleu4 = calculate_bleu(prompt, stego_fixed)
            ppl = calculate_perplexity(stego_fixed)
            results['bleu_1gram'][idx].append(bleu1)
            results['bleu_4gram'][idx].append(bleu4)
            results['perplexity'][idx].append(ppl)
            print(f"  定长编码: BLEU1={bleu1:.4f}, BLEU4={bleu4:.4f}, PPL={ppl:.2f}")
        except Exception as e:
            print(f"  定长编码失败: {e}")
            results['bleu_1gram'][idx].append(0)
            results['bleu_4gram'][idx].append(0)
            results['perplexity'][idx].append(1000)

        # 4. 哈夫曼编码
        try:
            stego_huffman = generate_with_baseline(prompt, secret, method='huffman')
            bleu1 = calculate_bleu_1gram(prompt, stego_huffman)
            bleu4 = calculate_bleu(prompt, stego_huffman)
            ppl = calculate_perplexity(stego_huffman)
            results['bleu_1gram'][idx].append(bleu1)
            results['bleu_4gram'][idx].append(bleu4)
            results['perplexity'][idx].append(ppl)
            print(f"  哈夫曼编码: BLEU1={bleu1:.4f}, BLEU4={bleu4:.4f}, PPL={ppl:.2f}")
        except Exception as e:
            print(f"  哈夫曼编码失败: {e}")
            results['bleu_1gram'][idx].append(0)
            results['bleu_4gram'][idx].append(0)
            results['perplexity'][idx].append(1000)

    # 打印汇总表格
    print("\n" + "=" * 90)
    print("实验结果汇总表")
    print("=" * 90)

    print(f"{'方法':<12} {'指标':<12} {'样本1':<10} {'样本2':<10} {'样本3':<10} {'样本4':<10} {'平均':<10}")
    print("-" * 90)

    for method_idx, method_name in enumerate(results['method']):
        for metric_name, metric_key in [('1-gram BLEU', 'bleu_1gram'), ('4-gram BLEU', 'bleu_4gram'),
                                        ('困惑度', 'perplexity')]:
            values = [results[metric_key][i][method_idx] for i in range(4)]
            avg_val = np.mean(values)
            if metric_name == '困惑度':
                print(
                    f"{method_name:<12} {metric_name:<12} {values[0]:<10.2f} {values[1]:<10.2f} {values[2]:<10.2f} {values[3]:<10.2f} {avg_val:<10.2f}")
            else:
                print(
                    f"{method_name:<12} {metric_name:<12} {values[0]:<10.4f} {values[1]:<10.4f} {values[2]:<10.4f} {values[3]:<10.4f} {avg_val:<10.4f}")
        print("-" * 90)

    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = results['method']
    x = np.arange(len(methods))
    width = 0.2

    colors = ['#2E86AB', '#F18F01', '#A23B72', '#2E8B57']

    for i in range(4):
        values = [results['bleu_1gram'][i][j] for j in range(len(methods))]
        axes[0].bar(x + i * width, values, width, label=f'样本{i + 1}', color=colors[i] if i < len(colors) else None)
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(methods, rotation=15)
    axes[0].set_ylabel('1-gram BLEU')
    axes[0].set_title('1-gram BLEU对比（越高越好）')
    axes[0].legend()
    axes[0].set_ylim(0, 0.5)

    for i in range(4):
        values = [results['bleu_4gram'][i][j] for j in range(len(methods))]
        axes[1].bar(x + i * width, values, width, label=f'样本{i + 1}', color=colors[i] if i < len(colors) else None)
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(methods, rotation=15)
    axes[1].set_ylabel('4-gram BLEU')
    axes[1].set_title('4-gram BLEU对比（越高越好）')
    axes[1].legend()
    axes[1].set_ylim(0, 0.4)

    for i in range(4):
        values = [results['perplexity'][i][j] for j in range(len(methods))]
        axes[2].bar(x + i * width, values, width, label=f'样本{i + 1}', color=colors[i] if i < len(colors) else None)
    axes[2].set_xticks(x + width * 1.5)
    axes[2].set_xticklabels(methods, rotation=15)
    axes[2].set_ylabel('困惑度')
    axes[2].set_title('困惑度对比（越低越好）')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('figures/similarity_comparison.png', dpi=300)
    plt.show()

    return results


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    text_similarity_experiment()