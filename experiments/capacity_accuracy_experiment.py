# experiments/capacity_accuracy_experiment.py
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed import embed_secret, extract_secret, text_to_bits, bits_to_text
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置SVG格式支持中文
matplotlib.rcParams['svg.fonttype'] = 'none'  # 保存文本为文字而非路径


def test_capacity_accuracy():
    """实验1：不同秘密信息长度下的提取准确率"""
    print("=" * 60)
    print("实验1：嵌入容量与提取准确率关系实验")
    print("=" * 60)

    cover_texts = [
        "The weather today is",
        "Deep learning is a subset of machine learning.",
        "Natural language processing enables computers to understand text.",
    ]

    test_lengths = [4, 8, 12, 16, 20, 24, 28, 32]

    results = {'accuracy': [], 'capacity': []}

    for msg_len in test_lengths:
        secret = "A" * msg_len
        print(f"\n测试长度: {msg_len} 字符")

        accuracies = []

        for cover in cover_texts:
            try:
                stego, bits, info = embed_secret(cover, secret)
                extracted = extract_secret(stego, len(bits), info)

                if extracted:
                    correct = sum(1 for a, b in zip(secret, extracted) if a == b)
                    accuracy = correct / max(len(secret), len(extracted))
                else:
                    accuracy = 0

                accuracies.append(accuracy)
                print(f"  封面: {cover[:30]}... 准确率={accuracy:.4f}")

            except Exception as e:
                print(f"  出错: {e}")
                accuracies.append(0)

        avg_accuracy = np.mean(accuracies)
        results['accuracy'].append(avg_accuracy)
        results['capacity'].append(msg_len)
        print(f"  平均准确率: {avg_accuracy:.4f}")

    # 绘制结果 - 保存为SVG矢量图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['capacity'], results['accuracy'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('秘密信息长度 (字符)', fontsize=12)
    ax.set_ylabel('提取准确率', fontsize=12)
    ax.set_title('(a) 嵌入容量与提取准确率关系', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    for i, (x, y) in enumerate(zip(results['capacity'], results['accuracy'])):
        ax.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)

    plt.tight_layout()

    # 保存为SVG矢量图（适合LaTeX）
    plt.savefig('figures/capacity_accuracy.svg', format='svg', bbox_inches='tight')
    # 同时保存PDF（备用）
    plt.savefig('figures/capacity_accuracy.pdf', format='pdf', bbox_inches='tight')
    # 保存PNG作为预览
    plt.savefig('figures/capacity_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n结果已保存至: figures/capacity_accuracy.svg (矢量图) 和 figures/capacity_accuracy.png")
    return results


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    test_capacity_accuracy()