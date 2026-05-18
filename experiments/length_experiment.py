# experiments/length_experiment.py
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.steganography import GenerativeSteganography
from config import config
from embed import embed_secret, extract_secret, find_model_file
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def length_experiment():
    """不同秘密信息长度实验"""
    print("=" * 60)
    print("实验4：不同秘密信息长度实验")
    print("=" * 60)

    # 加载模型
    model_path = find_model_file()
    if not model_path:
        print("错误：未找到模型文件")
        return

    model = GenerativeSteganography(config)
    checkpoint = torch.load(model_path, map_location=config.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 封面文本
    cover_text = "The quick brown fox jumps over the lazy dog."

    # 测试不同长度的秘密信息
    lengths = [4, 8, 12, 16, 20, 24, 28, 32]
    results = {
        'length': [],
        'embed_success': [],
        'extract_accuracy': [],
        'extract_match': [],
        'generation_time': []
    }

    import time

    for msg_len in tqdm(lengths, desc="测试长度"):
        secret = "A" * msg_len

        # 测量嵌入时间
        start_time = time.time()
        stego_text = embed_secret(cover_text, secret, model_path)
        embed_time = time.time() - start_time

        # 提取
        extracted, confidence = extract_secret(stego_text, model_path)

        # 计算准确率
        if extracted:
            # 计算字符级匹配
            match = secret in extracted
            # 计算比特级准确率
            from embed import text_to_bits
            secret_bits = text_to_bits(secret)
            extracted_bits = text_to_bits(extracted[:len(secret)])
            if len(secret_bits) > 0 and len(extracted_bits) > 0:
                min_len = min(len(secret_bits), len(extracted_bits))
                correct = (secret_bits[:min_len] == extracted_bits[:min_len]).sum().item()
                accuracy = correct / min_len
            else:
                accuracy = 0
        else:
            match = False
            accuracy = 0

        results['length'].append(msg_len)
        results['embed_success'].append(1 if stego_text else 0)
        results['extract_accuracy'].append(accuracy)
        results['extract_match'].append(1 if match else 0)
        results['generation_time'].append(embed_time)

        print(f"\n长度 {msg_len}:")
        print(f"  提取准确率: {accuracy:.4f}")
        print(f"  完全匹配: {'✓' if match else '✗'}")
        print(f"  生成时间: {embed_time:.2f}s")

    # 绘制结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(results['length'], results['extract_accuracy'], 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='理想准确率')
    plt.xlabel('秘密信息长度 (字符)', fontsize=12)
    plt.ylabel('提取准确率', fontsize=12)
    plt.title('不同长度下的提取准确率', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    colors = ['green' if m else 'red' for m in results['extract_match']]
    plt.bar(results['length'], results['extract_match'], color=colors)
    plt.xlabel('秘密信息长度 (字符)', fontsize=12)
    plt.ylabel('是否完全匹配 (1=是, 0=否)', fontsize=12)
    plt.title('不同长度下的完全匹配情况', fontsize=14)
    plt.ylim(-0.1, 1.1)

    plt.subplot(1, 3, 3)
    plt.plot(results['length'], results['generation_time'], 'go-', linewidth=2, markersize=6)
    plt.xlabel('秘密信息长度 (字符)', fontsize=12)
    plt.ylabel('生成时间 (秒)', fontsize=12)
    plt.title('不同长度下的生成时间', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/length_analysis.png', dpi=300)
    plt.show()

    # 计算临界长度
    max_capacity = (config.max_length - 1) * config.encoding_bits / 8  # 字节
    print("\n" + "=" * 40)
    print("实验结果分析")
    print("=" * 40)
    print(f"理论最大容量: {max_capacity:.1f} 字符")
    print(f"实际有效长度: ≤{max_capacity:.0f} 字符")

    # 统计
    success_rates = []
    for i, l in enumerate(lengths):
        if l <= max_capacity:
            success_rates.append(results['extract_accuracy'][i])

    print(f"容量内平均准确率: {np.mean(success_rates):.4f}")

    return results


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    length_experiment()