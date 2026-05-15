# 文件路径: verify_phase4.py
import os
import torch
from collections import Counter


def verify_artifacts(domain_name="yelp_restaurant"):
    print(f"开始执行阶段四产物质量核查 (Domain: {domain_name})...\n")
    data_dir = f"data/yelp_restaurant"

    # 核查 1: 伪标签分布是否健康
    label_file = os.path.join(data_dir, "label-sentences.txt")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"严重缺失：未找到伪标签文件 {label_file}")

    print("[1/2] 校验伪标签分布 (label-sentences.txt)...")
    aspect_counts = Counter()
    pol_counts = Counter()

    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 2 == 1:  # 标签在奇数行
                cat, pol = line.strip().split()
                aspect_counts[cat] += 1
                pol_counts[pol] += 1

    print(f"      - 总共成功打标句子数: {sum(aspect_counts.values())}")
    print(f"      - 维度分布: {dict(aspect_counts)}")
    print(f"      - 情感分布: {dict(pol_counts)}")
    if not aspect_counts:
        print("      ⚠️ 警告：伪标签生成失败（数量为0），请检查阶段二的种子词和阶段一的数据。")

    # 核查 2: 模型权重是否合法
    model_file = os.path.join(data_dir, "model_final.pth")
    print(f"\n[2/2] 校验引擎物理权重 (model_final.pth)...")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"· 严重缺失：未找到最终权重文件 {model_file}。请检查阶段三的强制持久化代码。")

    file_size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"      - 权重文件大小: {file_size_mb:.2f} MB")
    if file_size_mb < 100:
        print("      ⚠️ 警告：文件体积过小（正常 BERT 权重应 >300MB），可能是空壳保存。")

    try:
        # 尝试加载模型到内存，检查是否损坏
        model = torch.load(model_file, map_location='cpu')
        print(f"      - 权重加载测试: 成功！架构类型 -> {type(model).__name__}")
    except Exception as e:
        print(f"      严重错误：权重文件已损坏或无法加载！\n{e}")

    print("\n✅ 阶段四核查结束！如果上述指标均正常，你的专属 Yelp 抽取大脑已准备就绪，可随时接入阶段五的批量推理！")


if __name__ == "__main__":
    verify_artifacts("yelp_restaurant")