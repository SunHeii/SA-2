import json
import re
import random
import os
from datetime import datetime, timedelta


def calculate_time_features(local_time_str, longitude):
    """
    【时间逆向工程核心】
    Yelp 的时间是 Local Time。SID 需要 UTCTime 和 TimeOffset(分钟)。
    这里我们利用经度粗略推算时区，并逆向求出 UTCTime。
    """
    try:
        local_dt = datetime.strptime(local_time_str, '%Y-%m-%d %H:%M:%S')
        # 经度每 15 度为一个时区 (1小时 = 60分钟)
        time_offset_minutes = int(round(longitude / 15.0) * 60)

        # 逆推 UTC 时间：UTC = Local - Offset
        utc_dt = local_dt - timedelta(minutes=time_offset_minutes)
        utc_time_str = utc_dt.strftime('%Y-%m-%d %H:%M:%S')

        return utc_time_str, time_offset_minutes
    except Exception as e:
        return None, None


def get_int_id(string_id, mapping_dict):
    """
    将长字符串 ID 映射为从 0 开始的稠密整数 ID
    """
    if string_id not in mapping_dict:
        mapping_dict[string_id] = len(mapping_dict)
    return mapping_dict[string_id]


def yelp_phase_one_pipeline(business_json_path, review_json_path, output_dir, max_reviews=10000):
    """
    Yelp 数据预处理与管道设计 - SID 终极对齐版
    """
    print(f"🚀 Yelp -> SID 数据底座管道启动，目标抽取数量: {max_reviews} 条\n")

    os.makedirs(output_dir, exist_ok=True)
    out_txt_path = os.path.join(output_dir, "train.txt")
    out_csv_path = os.path.join(output_dir, "review_to_poi_rich_interactions.csv")
    user_map_path = os.path.join(output_dir, "user_map.json")
    poi_map_path = os.path.join(output_dir, "poi_map.json")

    # =====================================================================
    # Step 1: 确定数据边界与富信息提取 (加入 Latitude 与 PId 映射)
    # =====================================================================
    print("[1/5] 正在解析 Business 节点，提取空间特征并构建 PId 映射...")

    valid_businesses = {}
    poi_map = {}  # POI 的 String -> Int 映射字典

    with open(business_json_path, 'r', encoding='utf-8') as bf:
        for line in bf:
            b_data = json.loads(line)
            categories = b_data.get('categories')

            if categories and 'Restaurants' in categories:
                cat_str = categories.replace('"', '').replace(',', '|')
                longitude = b_data.get('longitude', 0.0)
                latitude = b_data.get('latitude', 0.0)  # 【新增】：SID 必需的纬度

                # 获取整数 PId
                p_int_id = get_int_id(b_data['business_id'], poi_map)

                valid_businesses[b_data['business_id']] = {
                    'PId': p_int_id,
                    'category': cat_str,
                    'longitude': longitude,
                    'latitude': latitude
                }

    print(f"      => 成功捕获 {len(valid_businesses)} 个目标 POI。")

    # =====================================================================
    # Step 2: 流式清洗评论，构建纯文本语料与【SID 交互宽表】
    # =====================================================================
    print("\n[2/5] 开始流式清洗评论，执行时间逆推与 UId 映射...")

    user_map = {}  # User 的 String -> Int 映射字典
    mapping_keys_for_validation = set()
    processed_count = 0

    with open(review_json_path, 'r', encoding='utf-8') as rf, \
            open(out_txt_path, 'w', encoding='utf-8') as out_txt, \
            open(out_csv_path, 'w', encoding='utf-8') as out_csv:

        # 【核心对齐】：表头严格使用 SID 的命名 (UId, PId, Latitude 等)
        # 保留 Stars 用于后续 Phase 5 的情感兜底
        out_csv.write("Review_ID,UId,PId,Category,Latitude,Longitude,TimeOffset,UTCTime,Stars\n")

        for line in rf:
            r_data = json.loads(line)
            b_id_str = r_data.get('business_id')
            r_id = r_data.get('review_id')
            u_id_str = r_data.get('user_id')
            text = r_data.get('text', '')
            date_str = r_data.get('date', '')
            stars = r_data.get('stars', 3.0)

            if not all([b_id_str, r_id, u_id_str, date_str]):
                continue

            if b_id_str in valid_businesses and len(text.split()) >= 15:
                cleaned_text = ' '.join(text.split())
                cleaned_text = re.sub(r'^[^a-zA-Z0-9]+', '', cleaned_text).strip()

                if not cleaned_text:
                    continue

                poi_info = valid_businesses[b_id_str]

                # 时间逆推工程
                utc_time_str, time_offset_mins = calculate_time_features(date_str, poi_info['longitude'])
                if utc_time_str is None:
                    continue

                    # 获取整数 UId
                u_int_id = get_int_id(u_id_str, user_map)

                # 构造 NLP 模型所需的输入文本 (带锚点)
                formatted_line = f"[{r_id}] {cleaned_text}\n"
                out_txt.write(formatted_line)

                # 构造 SID 所需的交互宽表
                csv_row = f"{r_id},{u_int_id},{poi_info['PId']},{poi_info['category']},{poi_info['latitude']},{poi_info['longitude']},{time_offset_mins},{utc_time_str},{stars}\n"
                out_csv.write(csv_row)

                mapping_keys_for_validation.add(r_id)
                processed_count += 1

                if processed_count >= max_reviews:
                    break

    print(f"      => 成功清洗并输出 {processed_count} 条交互记录。")
    print(f"      => 累计映射 {len(user_map)} 个独立 User。")

    # =====================================================================
    # Step 3: 持久化 ID 映射字典 (极其重要)
    # =====================================================================
    print("\n[3/5] 保存全局 ID 映射字典...")
    with open(user_map_path, 'w', encoding='utf-8') as f:
        json.dump(user_map, f)
    with open(poi_map_path, 'w', encoding='utf-8') as f:
        json.dump(poi_map, f)
    print("      => user_map.json 与 poi_map.json 写入完成。")

    # =====================================================================
    # Step 4: 一致性强校验
    # =====================================================================
    print("\n[4/5] 启动一致性强校验机制...")
    txt_line_count = sum(1 for _ in open(out_txt_path, 'r', encoding='utf-8'))
    csv_line_count = sum(1 for _ in open(out_csv_path, 'r', encoding='utf-8')) - 1

    if txt_line_count != csv_line_count or txt_line_count != len(mapping_keys_for_validation):
        raise RuntimeError("致命错误：文本行数与映射表数不一致！")
    print("      => [通过] 基数核对：100% 一致。")

    print(f"\n✅ 阶段一执行完毕！系统已为 SID 准备好完美的数据底座。")


if __name__ == "__main__":
    BUSINESS_JSON = "data/yelp/yelp_academic_dataset_business.json"
    REVIEW_JSON = "data/yelp/yelp_academic_dataset_review.json"
    OUTPUT_DIR = "data/yelp_restaurant"

    # 您可以把这里的 max_reviews 调大 (比如 50000) 以获取更多数据
    yelp_phase_one_pipeline(
        business_json_path=BUSINESS_JSON,
        review_json_path=REVIEW_JSON,
        output_dir=OUTPUT_DIR,
        max_reviews=10000
    )