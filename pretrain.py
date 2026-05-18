# data/preprocess.py
import os
import json
import random
import re
from pathlib import Path
from tqdm import tqdm


class SportsDataProcessor:
    """体育新闻数据处理"""

    def __init__(self, data_dir="data/datasets/THUCNews/THUCNews/体育", max_files=None):
        self.data_dir = Path(data_dir)
        self.max_files = max_files

        # 检查路径是否存在
        if not self.data_dir.exists():
            print(f"❌ 数据目录不存在: {self.data_dir}")
            print(f"  当前工作目录: {Path.cwd()}")

            # 尝试其他可能的路径
            possible_paths = [
                "data/datasets/THUCNews/THUCNews/体育",
                "datasets/THUCNews/THUCNews/体育",
                "THUCNews/体育",
                "../data/datasets/THUCNews/THUCNews/体育"
            ]

            for path in possible_paths:
                test_path = Path(path)
                if test_path.exists():
                    print(f"✅ 找到路径: {test_path}")
                    self.data_dir = test_path
                    break

            if not self.data_dir.exists():
                print("❌ 未找到任何有效的数据目录")
                return

        # 获取所有txt文件
        self.txt_files = list(self.data_dir.glob("*.txt"))
        if not self.txt_files:
            print(f"❌ 在目录 {self.data_dir} 中未找到.txt文件")
            # 尝试查找所有文件
            all_files = list(self.data_dir.glob("*"))
            print(f"  目录中的文件: {len(all_files)} 个")
            for i, file in enumerate(all_files[:10]):
                print(f"    {i + 1}. {file.name}")
            return

        print(f"数据目录: {self.data_dir}")
        print(f"找到 {len(self.txt_files)} 个文本文件")

    def read_single_file(self, file_path):
        """读取单个文件，尝试多种编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                return content.strip()
            except (UnicodeDecodeError, LookupError):
                continue

        # 如果所有编码都失败，尝试二进制读取
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # 尝试utf-8解码，忽略错误
                return content.decode('utf-8', errors='ignore').strip()
        except:
            return ""

    def clean_text(self, text):
        """清理文本，但不要清理得太严格"""
        if not text:
            return ""

        # 去除空白字符
        text = re.sub(r'\s+', ' ', text)

        # 保留中文、英文、数字和常见标点
        text = re.sub(r'[^\u4e00-\u9fff\w\d，。！？、；："\'\-\—\s]', ' ', text)

        # 再次合并空白字符
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def process_files(self):
        """处理所有文件"""
        if not hasattr(self, 'txt_files') or not self.txt_files:
            print("没有可处理的文件")
            return []

        all_texts = []
        error_files = []

        print("开始处理文件...")

        # 限制文件数量，如果指定了max_files
        files_to_process = self.txt_files
        if self.max_files:
            files_to_process = self.txt_files[:self.max_files]
            print(f"只处理前 {self.max_files} 个文件进行测试")

        for file_path in tqdm(files_to_process, desc="处理文件"):
            try:
                content = self.read_single_file(file_path)

                if not content:
                    error_files.append(str(file_path))
                    continue

                # 清理文本
                cleaned_content = self.clean_text(content)

                # 检查清理后的文本长度
                if cleaned_content and len(cleaned_content) >= 20:  # 至少20个字符
                    all_texts.append(cleaned_content)

                    # 显示第一个文件的样本
                    if len(all_texts) == 1:
                        print(f"\n第一个文件的样本（前200字符）:")
                        sample = cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
                        print(f"  {sample}")

            except Exception as e:
                error_files.append(f"{file_path}: {str(e)}")
                continue

        print(f"\n处理完成:")
        print(f"  成功读取: {len(all_texts)} 个文件")
        print(f"  读取失败: {len(error_files)} 个文件")

        if error_files and len(error_files) <= 10:
            print(f"  失败的文件:")
            for err in error_files[:10]:
                print(f"    - {err}")

        return all_texts

    def create_datasets(self, texts, output_dir="data/processed"):
        """创建训练/验证/测试数据集"""
        if not texts:
            print("错误: 没有文本数据!")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n创建数据集...")
        print(f"  总文本数: {len(texts)}")

        # 打乱数据
        random.shuffle(texts)

        # 分割数据集 (80%训练, 10%验证, 10%测试)
        total = len(texts)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)

        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size + val_size]
        test_texts = texts[train_size + val_size:]

        print(f"  训练集: {len(train_texts)} 条")
        print(f"  验证集: {len(val_texts)} 条")
        print(f"  测试集: {len(test_texts)} 条")

        # 计算平均长度
        avg_length = sum(len(text) for text in texts) / len(texts)
        print(f"  平均文本长度: {avg_length:.1f} 字符")

        # 保存数据集
        with open(output_dir / "train.txt", 'w', encoding='utf-8') as f:
            for text in train_texts:
                f.write(text + "\n")

        with open(output_dir / "val.txt", 'w', encoding='utf-8') as f:
            for text in val_texts:
                f.write(text + "\n")

        with open(output_dir / "test.txt", 'w', encoding='utf-8') as f:
            for text in test_texts:
                f.write(text + "\n")

        # 保存统计信息
        stats = {
            "total_samples": total,
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "test_samples": len(test_texts),
            "avg_length": avg_length,
            "min_length": min(len(text) for text in texts),
            "max_length": max(len(text) for text in texts)
        }

        with open(output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\n数据集已保存到: {output_dir}")

        # 显示样本
        print(f"\n训练集样本（前3条）:")
        for i, text in enumerate(train_texts[:3]):
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"  {i + 1}. {preview}")

        return output_dir

    def run(self):
        """运行完整处理流程"""
        print("=" * 60)
        print("体育新闻数据预处理")
        print("=" * 60)

        # 1. 处理文件
        texts = self.process_files()

        if not texts:
            print("\n❌ 没有处理出任何文本!")

            # 尝试直接读取前5个文件
            if hasattr(self, 'txt_files') and self.txt_files:
                print("尝试直接读取一些文件...")
                test_texts = []
                for i, file_path in enumerate(self.txt_files[:5]):
                    print(f"\n直接读取文件 {i + 1}: {file_path.name}")
                    try:
                        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                            content = f.read()
                            print(f"  前200字符: {content[:200]}")
                            test_texts.append(content[:200])
                    except Exception as e:
                        print(f"  读取失败: {e}")

                # 使用测试数据继续
                if test_texts:
                    print("\n使用测试数据继续...")
                    texts = test_texts
                else:
                    return None

        # 2. 创建数据集
        output_dir = self.create_datasets(texts)

        print("\n" + "=" * 60)
        print("数据处理完成!")
        print("=" * 60)

        return output_dir


def main():
    """主函数"""
    # 指定正确的数据路径
    data_dir = "data/datasets/THUCNews/THUCNews/体育"

    print(f"尝试使用数据目录: {data_dir}")
    print(f"当前工作目录: {Path.cwd()}")

    # 创建处理器
    processor = SportsDataProcessor(data_dir=data_dir, max_files=None)

    # 检查是否初始化成功
    if not hasattr(processor, 'txt_files') or not processor.txt_files:
        print("❌ 处理器初始化失败")
        return

    # 运行处理
    output_dir = processor.run()

    if output_dir:
        print(f"\n下一步:")
        print(f"  1. 训练数据已保存到: {output_dir}")
        print(f"  2. 可以运行训练: python train.py")
        print(f"  3. 如果训练有问题，可以先测试少量数据: python train_small.py")
    else:
        print("\n❌ 数据处理失败")


if __name__ == "__main__":
    main()