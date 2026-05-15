import csv
import re
import math
import numpy as np
import nltk
from collections import defaultdict

# 引入纯净推理接口
from extractor_api import YelpFeatureExtractor


def execute_phase_5_pipeline(rich_csv_path, clean_text_path, output_dataset_path, batch_size=64):
    print("🚀 启动交互级细粒度情感打分与 SID 数据对齐流水线 (含情感掩码)...\n")

    # =====================================================================
    # Step 1: 抽取引擎挂载与预加载富信息 (适配全新 SID 字段)
    # =====================================================================
    print("[1/5] 执行热加载并加载时空富信息字典...")
    extractor = YelpFeatureExtractor(domain_config_name="yelp_restaurant")

    rich_interactions = {}
    with open(rich_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rich_interactions[row['Review_ID']] = row
            # 兜底机制：如果没有抓到 Stars，默认给中立的 3.0 分
            if 'Stars' not in row:
                rich_interactions[row['Review_ID']]['Stars'] = 3.0

    print(f"      => 成功加载 {len(rich_interactions)} 条富交互元数据。")

    # 用于暂存推算的 Soft Probabilities
    review_sentiment_mass = defaultdict(lambda: defaultdict(lambda: {'pos': 0.0, 'neg': 0.0}))

    # =====================================================================
    # Step 2: 读取文本并进行微观切分
    # =====================================================================
    print("\n[2/5] 读取纯净文本并进行微观粒度切分...")
    sentence_buffer = []
    cursor_buffer = []

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    with open(clean_text_path, 'r', encoding='utf-8') as tf:
        for line in tf:
            match = re.match(r'^\[([^\]]+)\]\s(.*)', line)
            if not match:
                continue

            r_id = match.group(1)
            text = match.group(2)

            # 只处理成功映射的记录
            if r_id not in rich_interactions:
                continue

            sentences = nltk.tokenize.sent_tokenize(text)
            for sent in sentences:
                if len(sent.strip()) >= 4:
                    sentence_buffer.append(sent)
                    cursor_buffer.append(r_id)

    print(f"      => 成功将语料拆解为 {len(sentence_buffer)} 个独立单句待推理。")

    # =====================================================================
    # Step 3: 高并发流式推理与动态水位线过滤 (软衰减底噪机制)
    # =====================================================================
    print("\n[3/5] 启动动态批处理与情感质量累加...")
    total_batches = math.ceil(len(sentence_buffer) / batch_size)

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sentence_buffer))

        batch_sentences = sentence_buffer[start_idx:end_idx]
        batch_cursors = cursor_buffer[start_idx:end_idx]

        batch_probs = extractor.extract_sentences_soft(batch_sentences)
        if not batch_probs:
            continue

        # 计算批次动态水位线，防止空载
        all_max_asp_probs = [max(p['aspects'].values()) for p in batch_probs if p.get('aspects')]
        all_max_pol_probs = [max(p['polarities'].values()) for p in batch_probs if p.get('polarities')]

        if all_max_asp_probs and all_max_pol_probs:
            dynamic_asp_threshold = max(0.2, np.percentile(all_max_asp_probs, 20))
            dynamic_pol_threshold = max(0.5, np.percentile(all_max_pol_probs, 20))
        else:
            dynamic_asp_threshold = 0.2
            dynamic_pol_threshold = 0.5

        for cursor, probs in zip(batch_cursors, batch_probs):
            aspect_probs = probs.get('aspects', {})
            polarity_probs = probs.get('polarities', {})

            if not aspect_probs or not polarity_probs:
                continue

            cur_max_asp = max(aspect_probs.values())
            cur_max_pol = max(polarity_probs.values())

            if cur_max_asp < dynamic_asp_threshold or cur_max_pol < dynamic_pol_threshold:
                continue

            for aspect_name, p_asp in aspect_probs.items():
                # 冷门维度阈值放宽
                min_pass_prob = 0.3 if aspect_name in ['ambience', 'price'] else 0.45

                # 软衰减机制：保留10%的弱信号作为打破平局的底噪
                weight = 1.0 if p_asp > min_pass_prob else 0.1

                p_pos = polarity_probs.get('positive', 0.0)
                p_neg = polarity_probs.get('negative', 0.0)

                review_sentiment_mass[cursor][aspect_name]['pos'] += (p_asp * p_pos) * weight
                review_sentiment_mass[cursor][aspect_name]['neg'] += (p_asp * p_neg) * weight

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"      - 推理进度: {i + 1} / {total_batches} Batches 完成")

    # =====================================================================
    # Step 4 & 5: 贝叶斯平滑折算与 SID 宽表最终落盘 (生成 Mask 掩码)
    # =====================================================================
    print("\n[4/5] 执行贝叶斯平滑算法，折算包含情感方差的 1~5 标量评分...")
    print("[5/5] 掩码生成与数据对齐：生成 8 维情感特征并缝合时空富信息！")

    aspects = ['food', 'service', 'ambience', 'price']

    # 【对齐核心】：严格遵照 SID 的字段大小写和新增的 Mask 列
    header = ['Review_ID', 'UId', 'PId', 'Category', 'Latitude', 'Longitude', 'TimeOffset', 'UTCTime',
              'Food_Score', 'Service_Score', 'Ambience_Score', 'Price_Score',
              'Food_Mask', 'Service_Mask', 'Ambience_Mask', 'Price_Mask']

    valid_interactions = 0

    # 维度先验偏差 (餐厅常识)
    aspect_priors = {
        'food': 0.15,
        'service': 0.05,
        'ambience': 0.00,
        'price': -0.15
    }

    with open(output_dataset_path, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        for r_id, rich_data in rich_interactions.items():
            scores = []
            masks = []
            mass = review_sentiment_mass.get(r_id, {})
            raw_stars = float(rich_data.get('Stars', 3.0))

            for asp in aspects:
                asp_mass = mass.get(asp, {'pos': 0.0, 'neg': 0.0})
                pos_val = asp_mass['pos']
                neg_val = asp_mass['neg']
                total_mass = pos_val + neg_val

                prior_score = min(5.0, max(1.0, raw_stars + aspect_priors.get(asp, 0.0)))

                # 【掩码核心机制】
                if total_mass < 0.01:
                    # 如果用户完全没提这个维度，使用先验兜底分，但掩码打上 0
                    final_score = prior_score
                    mention_mask = 0
                else:
                    # 正常极性对冲
                    net_polarity = (pos_val - neg_val) / total_mass
                    extracted_score = 3.0 + 2.0 * net_polarity

                    # 贝叶斯平滑
                    alpha = 0.5
                    final_score = (total_mass * extracted_score + alpha * prior_score) / (total_mass + alpha)

                    # 用户真实提及，掩码打上 1
                    mention_mask = 1

                scores.append(round(final_score, 3))
                masks.append(mention_mask)

            # 严格按照 Header 顺序拼接，使用安全 get()
            row_data = [
                r_id,
                rich_data.get('UId', ''),
                rich_data.get('PId', ''),
                rich_data.get('Category', ''),
                rich_data.get('Latitude', ''),
                rich_data.get('Longitude', ''),
                rich_data.get('TimeOffset', ''),
                rich_data.get('UTCTime', ''),
                scores[0],  # Food_Score
                scores[1],  # Service_Score
                scores[2],  # Ambience_Score
                scores[3],  # Price_Score
                masks[0],  # Food_Mask
                masks[1],  # Service_Mask
                masks[2],  # Ambience_Mask
                masks[3]  # Price_Mask
            ]
            writer.writerow(row_data)
            valid_interactions += 1

    print(f"\n✅ 阶段二执行完毕！特征工厂全链路打通。")
    print(f"已生成高度定制的 SID 推荐特征表: {output_dataset_path}")
    print(f"总计产出富交互记录: {valid_interactions} 条 (包含双重校验的 UId, PId, Latitude 及 8 维情感掩码)。")


if __name__ == "__main__":
    RICH_CSV_PATH = "data/yelp_restaurant/review_to_poi_rich_interactions.csv"
    CLEAN_TEXT_PATH = "data/yelp_restaurant/train.txt"
    SID_FINAL_OUTPUT = "data/yelp_restaurant/sid_interaction_dataset.csv"

    execute_phase_5_pipeline(
        rich_csv_path=RICH_CSV_PATH,
        clean_text_path=CLEAN_TEXT_PATH,
        output_dataset_path=SID_FINAL_OUTPUT,
        batch_size=64
    )