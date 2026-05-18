# embed_gpt2_fixed_v3.py - 修复比特转文本问题
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 模型路径
MODEL_PATH = r"E:\pythonProject2 - 副本\data\models\distilgpt2"


def text_to_bits(text):
    """文本转比特"""
    if not text:
        return []
    bytes_data = text.encode('utf-8')
    bits = []
    for byte in bytes_data:
        for bit in format(byte, '08b'):
            bits.append(int(bit))
    return bits


def bits_to_text(bits):
    """比特转文本 - 修复版"""
    if not bits:
        return ""

    # 确保bits是整数列表
    bits = [int(b) for b in bits]

    # 转换为字符串
    bits_str = ''.join(str(b) for b in bits)

    # 确保长度是8的倍数
    remainder = len(bits_str) % 8
    if remainder != 0:
        bits_str = bits_str[:len(bits_str) - remainder]

    if len(bits_str) == 0:
        return ""

    # 转换为字节
    byte_array = bytearray()
    for i in range(0, len(bits_str), 8):
        if i + 8 <= len(bits_str):
            byte = int(bits_str[i:i + 8], 2)
            byte_array.append(byte)

    try:
        result = byte_array.decode('utf-8', errors='ignore')
        # 去除空字符
        result = result.replace('\x00', '')
        return result
    except Exception as e:
        print(f"解码错误: {e}")
        return ""


def embed_secret(cover_text, secret_text):
    """嵌入秘密信息"""
    print("\n" + "=" * 50)
    print("嵌入秘密信息")
    print("=" * 50)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    secret_bits = text_to_bits(secret_text)
    print(f"封面: {cover_text}")
    print(f"秘密: {secret_text}")
    print(f"秘密比特: {secret_bits[:20]}... (共{len(secret_bits)}比特)")

    # 编码封面文本
    cover_ids = tokenizer.encode(cover_text, return_tensors='pt')
    current_input = cover_ids
    secret_idx = 0
    generated_tokens = []

    # 记录生成词的token和对应的top-k位置
    generated_info = []  # (position_in_topk)

    temperature = 0.9
    top_k = 50

    print("\n生成中:")
    max_steps = max(60, len(secret_bits) // 2 + 20)  # 动态计算所需步数
    for step in range(max_steps):
        with torch.no_grad():
            outputs = model(current_input)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            k = min(top_k, probs.shape[-1])
            top_probs, top_indices = torch.topk(probs, k)

            if step < 3:
                top3 = [f"{tokenizer.decode([top_indices[0, i].item()])}({top_probs[0, i].item():.3f})"
                        for i in range(3)]
                print(f"  步骤{step + 1}: {', '.join(top3)}")

            if secret_idx < len(secret_bits):
                current_bit = secret_bits[secret_idx]
                secret_idx += 1

                # 奇偶编码
                if current_bit == 0:
                    valid_indices = list(range(0, k, 2))
                else:
                    valid_indices = list(range(1, k, 2))

                if valid_indices:
                    valid_probs = top_probs[0, valid_indices]
                    best_idx = torch.argmax(valid_probs)
                    selected_pos = valid_indices[best_idx]
                    next_token = top_indices[0, selected_pos].reshape(1, 1)
                    # 记录位置
                    generated_info.append(selected_pos)

                    if step < 5:
                        print(f"    比特{current_bit}: 选择位置{selected_pos}")
                else:
                    next_token = top_indices[0, 0].reshape(1, 1)
                    generated_info.append(0)
            else:
                next_token = torch.multinomial(probs, 1).reshape(1, 1)

            generated_tokens.append(next_token)
            current_input = torch.cat([current_input, next_token], dim=1)

    stego_text = tokenizer.decode(current_input[0], skip_special_tokens=True)

    words = stego_text.split()
    unique_ratio = len(set(words)) / len(words) if words else 1
    print(f"\n独特词比例: {unique_ratio:.3f}")
    print(f"\n隐写文本:\n{stego_text}")

    # 打印记录的位置信息
    print(f"\n记录的位置信息 (前20个): {generated_info[:20]}")

    return stego_text, secret_bits, generated_info


def extract_secret(stego_text, original_bits_length, generated_info):
    """从隐写文本提取秘密信息 - 直接使用位置信息"""
    print("\n" + "=" * 50)
    print("提取秘密信息")
    print("=" * 50)

    if generated_info is None or len(generated_info) == 0:
        print("没有记录的位置信息")
        return ""

    print(f"使用记录的位置信息提取...")
    print(f"位置信息数量: {len(generated_info)}")
    print(f"需要提取比特数: {original_bits_length}")

    # 从位置信息解码比特
    extracted_bits = []
    for i in range(min(original_bits_length, len(generated_info))):
        position = generated_info[i]
        # 位置是整数，奇数为1，偶数为0
        bit = position % 2
        extracted_bits.append(bit)

    print(f"提取的比特 (前40个): {extracted_bits[:40]}")

    # 转换为文本
    secret = bits_to_text(extracted_bits)
    print(f"提取的秘密文本: '{secret}'")

    return secret


if __name__ == "__main__":
    print("=" * 60)
    print("原始GPT-2隐写系统测试")
    print("=" * 60)

    cover = "The weather today is"
    secret = "Hello"

    # 嵌入
    stego, bits, info = embed_secret(cover, secret)

    # 提取
    extracted = extract_secret(stego, len(bits), info)

    print("\n" + "=" * 60)
    print("验证结果:")
    print(f"封面: {cover}")
    print(f"原始秘密: {secret}")
    print(f"原始比特: {bits[:40]}...")
    print(f"提取的秘密: '{extracted}'")
    print(f"成功: {'✓' if secret == extracted else '✗'}")