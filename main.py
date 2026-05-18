# main.py
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='demo', choices=['embed', 'extract', 'demo'])
    parser.add_argument('--cover', type=str)
    parser.add_argument('--secret', type=str)
    parser.add_argument('--stego', type=str)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'embed':
        if not args.cover or not args.secret:
            print("请提供封面文本和秘密文本")
            return
        from embed import embed_secret
        stego, _, _ = embed_secret(args.cover, args.secret)
        print(f"\n隐写文本:\n{stego}")

    elif args.mode == 'extract':
        if not args.stego:
            print("请提供隐写文本")
            return
        from embed import extract_secret
        # 需要知道原始比特长度，这里用40作为默认值
        secret = extract_secret(args.stego, 40, None)
        print(f"\n提取的秘密: {secret}")

    elif args.mode == 'demo':
        print("=" * 50)
        print("文本隐写系统演示 (奇偶编码 1比特/词)")
        print("=" * 50)
        from embed import embed_secret, extract_secret

        cover = "The weather today is"
        secret = "Hello"
        print(f"封面: {cover}")
        print(f"秘密: {secret}")

        # 嵌入 - embed_secret 返回3个值: stego, bits, info
        stego, bits, info = embed_secret(cover, secret)
        print(f"隐写: {stego}")

        # 提取 - extract_secret 只需要3个参数: stego, bits长度, info
        # 注意：extract_secret 返回的是秘密字符串，不是元组
        extracted = extract_secret(stego, len(bits), info)
        print(f"提取: {extracted}")
        print(f"结果: {'✓ 成功' if secret == extracted else '✗ 失败'}")


if __name__ == "__main__":
    main()