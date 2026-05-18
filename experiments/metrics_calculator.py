# experiments/metrics_calculator.py
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed import embed_secret, text_to_bits, bits_to_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = r"E:\pythonProject2 - 副本\data\models\distilgpt2"


def get_model_and_tokenizer():
    """获取模型和分词器"""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def calculate_perplexity(text, model, tokenizer):
    """
    计算困惑度（Perplexity）
    参考论文公式(3-13)
    """
    try:
        inputs = tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        print(f"困惑度计算错误: {e}")
        return 1000.0


def calculate_token_distribution(texts, tokenizer):
    """
    计算词频分布（用于KL散度）
    """
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    token_counter = Counter(all_tokens)
    total = len(all_tokens)

    # 计算概率分布
    distribution = {}
    for token, count in token_counter.items():
        distribution[token] = count / total

    return distribution


def calculate_kl_divergence(p_dist, q_dist, epsilon=1e-10):
    """
    计算KL散度 D_KL(P || Q)
    参考论文公式(3-14)
    """
    # 获取所有token的并集
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    kl_div = 0.0
    for token in all_tokens:
        p = p_dist.get(token, epsilon)
        q = q_dist.get(token, epsilon)
        kl_div += p * np.log(p / q)

    return kl_div


def calculate_js_divergence(p_dist, q_dist):
    """
    计算JS散度
    参考论文公式(3-15)
    """
    # 获取所有token的并集
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    # 计算平均分布 M = (P + Q) / 2
    m_dist = {}
    for token in all_tokens:
        p = p_dist.get(token, 0)
        q = q_dist.get(token, 0)
        m_dist[token] = (p + q) / 2

    # 计算 KL(P || M) 和 KL(Q || M)
    kl_pm = calculate_kl_divergence(p_dist, m_dist)
    kl_qm = calculate_kl_divergence(q_dist, m_dist)

    # JS散度 = (KL(P||M) + KL(Q||M)) / 2
    js_div = (kl_pm + kl_qm) / 2

    return js_div


def generate_natural_texts(prompts, model, tokenizer, n_samples=20):
    """
    生成自然文本作为对比基准
    """
    natural_texts = []

    for prompt in prompts[:n_samples]:
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                natural_texts.append(text)
        except Exception as e:
            print(f"生成自然文本错误: {e}")
            natural_texts.append(prompt)

    return natural_texts


def generate_stego_texts(prompts, secret, n_samples=20):
    """
    生成隐写文本
    """
    stego_texts = []

    for prompt in prompts[:n_samples]:
        try:
            stego, _, _ = embed_secret(prompt, secret)
            stego_texts.append(stego)
        except Exception as e:
            print(f"生成隐写文本错误: {e}")
            stego_texts.append(prompt)

    return stego_texts


def calculate_all_metrics():
    """
    计算所有实验指标
    """
    print("=" * 60)
    print("计算困惑度、KL散度和JS散度")
    print("=" * 60)

    model, tokenizer = get_model_and_tokenizer()

    # 测试数据
    prompts = [
                  "The weather today is",
                  "Deep learning is",
                  "Natural language processing",
                  "Artificial intelligence",
                  "Machine learning"
              ] * 4  # 20个样本

    secret = "Hello"

    # 生成文本
    print("\n生成自然文本...")
    natural_texts = generate_natural_texts(prompts, model, tokenizer, n_samples=20)

    print("生成隐写文本...")
    stego_texts = generate_stego_texts(prompts, secret, n_samples=20)

    # 计算困惑度
    print("\n计算困惑度...")
    natural_perplexities = []
    stego_perplexities = []

    for text in natural_texts:
        ppl = calculate_perplexity(text, model, tokenizer)
        natural_perplexities.append(ppl)

    for text in stego_texts:
        ppl = calculate_perplexity(text, model, tokenizer)
        stego_perplexities.append(ppl)

    avg_natural_ppl = np.mean(natural_perplexities)
    avg_stego_ppl = np.mean(stego_perplexities)
    std_natural_ppl = np.std(natural_perplexities)
    std_stego_ppl = np.std(stego_perplexities)

    print(f"自然文本困惑度: {avg_natural_ppl:.2f} ± {std_natural_ppl:.2f}")
    print(f"隐写文本困惑度: {avg_stego_ppl:.2f} ± {std_stego_ppl:.2f}")

    # 计算词频分布
    print("\n计算词频分布...")
    natural_dist = calculate_token_distribution(natural_texts, tokenizer)
    stego_dist = calculate_token_distribution(stego_texts, tokenizer)

    # 计算KL散度和JS散度
    kl_div = calculate_kl_divergence(natural_dist, stego_dist)
    js_div = calculate_js_divergence(natural_dist, stego_dist)

    print(f"KL散度: {kl_div:.4f}")
    print(f"JS散度: {js_div:.4f}")

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 困惑度对比
    axes[0].bar(['自然文本', '隐写文本'],
                [avg_natural_ppl, avg_stego_ppl],
                yerr=[std_natural_ppl, std_stego_ppl],
                color=['#2E8B57', '#2E86AB'], capsize=5)
    axes[0].set_ylabel('困惑度')
    axes[0].set_title('困惑度对比（越低越好）')
    for i, v in enumerate([avg_natural_ppl, avg_stego_ppl]):
        axes[0].text(i, v + 2, f'{v:.1f}', ha='center')

    # KL/JS散度
    axes[1].bar(['KL散度', 'JS散度'], [kl_div, js_div], color=['#F18F01', '#A23B72'])
    axes[1].set_ylabel('散度值')
    axes[1].set_title('概率分布差异')
    for i, v in enumerate([kl_div, js_div]):
        axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig('figures/perplexity_metrics.png', dpi=300)
    plt.show()

    return {
        'natural_perplexity': {'mean': avg_natural_ppl, 'std': std_natural_ppl},
        'stego_perplexity': {'mean': avg_stego_ppl, 'std': std_stego_ppl},
        'kl_divergence': kl_div,
        'js_divergence': js_div
    }


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    results = calculate_all_metrics()

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"自然文本困惑度: {results['natural_perplexity']['mean']:.2f} ± {results['natural_perplexity']['std']:.2f}")
    print(f"隐写文本困惑度: {results['stego_perplexity']['mean']:.2f} ± {results['stego_perplexity']['std']:.2f}")
    print(f"KL散度: {results['kl_divergence']:.4f}")
    print(f"JS散度: {results['js_divergence']:.4f}")