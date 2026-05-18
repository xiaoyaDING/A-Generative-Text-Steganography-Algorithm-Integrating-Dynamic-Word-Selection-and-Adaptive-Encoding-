# extract.py
import torch
import os
import sys
from models.steganography import GenerativeSteganography
from config import config
import glob

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def bits_to_text(bits_tensor):
    """比特流 -> 文本"""
    if bits_tensor is None or len(bits_tensor) == 0:
        return ""

    bits_str = ''.join(str(int(b)) for b in bits_tensor.cpu().numpy())

    if len(bits_str) % 8 != 0:
        bits_str = bits_str[:-(len(bits_str) % 8)]

    byte_array = bytearray()
    for i in range(0, len(bits_str), 8):
        byte = bits_str[i:i + 8]
        if len(byte) == 8:
            byte_array.append(int(byte, 2))

    try:
        text = byte_array.decode('utf-8', errors='ignore').strip('\x00')
        return text
    except Exception as e:
        print(f"解码失败: {e}")
        return ""


def find_model_file():
    """查找训练好的模型文件"""
    # 获取项目根目录
    current_file = os.path.abspath(__file__)
    project_dir = os.path.dirname(current_file)

    # checkpoints目录
    checkpoint_dir = os.path.join(project_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        # 尝试当前工作目录
        work_dir = os.getcwd()
        checkpoint_dir = os.path.join(work_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            return None

    # 优先级列表
    priority_list = [
        os.path.join(checkpoint_dir, "best_model.pth"),
        os.path.join(checkpoint_dir, "final_model.pth"),
    ]

    # 添加end_to_end模型（按epoch倒序）
    e2e_files = glob.glob(os.path.join(checkpoint_dir, "end_to_end_model_epoch_*.pth"))
    e2e_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch_' in x else 0, reverse=True)
    priority_list.extend(e2e_files)

    # 添加epoch模型
    epoch_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
    epoch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch_' in x else 0, reverse=True)
    priority_list.extend(epoch_files)

    # 添加所有pth文件
    all_pth = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    priority_list.extend(all_pth)

    # 去重并返回第一个存在的文件
    seen = set()
    for f in priority_list:
        if f not in seen and os.path.exists(f):
            seen.add(f)
            return f

    return None


def extract_secret(stego_text, model_path=None):
    """从隐写文本中提取秘密信息"""
    print("\n" + "=" * 50)
    print("开始提取秘密信息")
    print("=" * 50)

    # 查找模型文件
    if model_path is None:
        model_path = find_model_file()

    if not model_path or not os.path.exists(model_path):
        print(f"错误: 未找到模型文件")
        print(f"请确保 checkpoints/ 目录下有训练好的模型")
        return None, 0.0

    print(f"使用模型: {model_path}")

    # 初始化模型
    print("初始化模型...")
    model = GenerativeSteganography(config)

    try:
        # 加载训练好的模型权重
        print(f"加载模型权重...")
        checkpoint = torch.load(model_path, map_location=config.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 从checkpoint加载模型成功")
            if 'val_accuracy' in checkpoint:
                print(f"  模型验证准确率: {checkpoint['val_accuracy']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ 加载模型权重成功")

    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return None, 0.0

    model.to(config.device)
    model.eval()
    print(f"模型已移动到: {config.device}")

    # 准备隐写文本
    tokenizer = model.tokenizer
    print(f"\n隐写文本: {stego_text[:100]}..." if len(stego_text) > 100 else f"\n隐写文本: {stego_text}")

    encoding = tokenizer.encode_plus(
        stego_text,
        max_length=config.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)

    # 提取秘密信息
    print("\n提取中...")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, mode='extract')

    # 获取提取的比特
    extracted_bits = outputs['extracted_bits'].cpu()
    print(f"提取到 {extracted_bits.shape[1]} 比特")

    # 转换为文本
    secret_text = bits_to_text(extracted_bits.flatten())

    # 计算置信度
    confidence = outputs.get('confidence_scores', torch.ones(1)).mean().item()

    print(f"✓ 提取完成")
    print(f"提取的秘密: {secret_text}")
    print(f"置信度: {confidence:.4f}")

    return secret_text, confidence


def test_extraction():
    """测试提取功能"""
    print("=" * 60)
    print("测试隐写提取功能")
    print("=" * 60)

    # 查找模型文件
    model_path = find_model_file()
    if model_path is None:
        print("错误: 没有找到训练好的模型文件")
        print("请先运行: python train.py")
        return

    print(f"找到模型: {model_path}")

    # 测试用例（需要从embed.py生成的实际隐写文本）
    test_cases = [
        {
            "stego_text": "The quick brown fox jumps over the lazy dog.",
            "description": "示例隐写文本 1"
        },
        {
            "stego_text": "Deep learning models require large datasets for training.",
            "description": "示例隐写文本 2"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i} ({test_case['description']}):")
        print(f"隐写文本: {test_case['stego_text'][:50]}...")

        secret_text, confidence = extract_secret(test_case['stego_text'], model_path)

        if secret_text:
            print(f"✓ 提取的秘密: {secret_text}")
            print(f"✓ 提取置信度: {confidence:.4f}")

            # 保存结果
            os.makedirs("output", exist_ok=True)
            with open(f"output/extract_test_{i}.txt", "w", encoding="utf-8") as f:
                f.write(f"隐写文本: {test_case['stego_text']}\n")
                f.write(f"提取的秘密: {secret_text}\n")
                f.write(f"提取置信度: {confidence:.4f}\n")
        else:
            print("✗ 提取失败")


def extract_from_file(input_file, model_path=None, output_file=None):
    """从文件中的隐写文本批量提取秘密"""
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return

    # 查找模型文件
    if model_path is None:
        model_path = find_model_file()

    if model_path is None or not os.path.exists(model_path):
        print("错误: 没有找到可用的模型文件")
        return

    # 读取隐写文本
    with open(input_file, 'r', encoding='utf-8') as f:
        stego_texts = [line.strip() for line in f if line.strip()]

    print(f"从文件读取 {len(stego_texts)} 条隐写文本")

    # 批量提取
    results = []
    for i, stego_text in enumerate(stego_texts, 1):
        if len(stego_text) < 10:
            continue

        print(f"处理第 {i}/{len(stego_texts)} 条...")
        secret_text, confidence = extract_secret(stego_text, model_path)

        if secret_text:
            results.append({
                "index": i,
                "stego_text": stego_text[:100] + "..." if len(stego_text) > 100 else stego_text,
                "secret_text": secret_text,
                "confidence": confidence
            })

    # 输出结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"索引: {result['index']}\n")
                f.write(f"隐写文本: {result['stego_text']}\n")
                f.write(f"提取的秘密: {result['secret_text']}\n")
                f.write(f"置信度: {result['confidence']:.4f}\n")
                f.write("-" * 50 + "\n")

        print(f"\n结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 首先检查是否有模型文件
    model_path = find_model_file()
    if model_path:
        print(f"找到模型: {model_path}")
        # 运行测试
        test_extraction()
    else:
        print("错误: 没有找到训练好的模型文件")
        print("请先运行: python train.py")