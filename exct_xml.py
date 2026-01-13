import re
import json
from collections import defaultdict

def parse_bilibili_danmaku_file(xml_path: str, save_path="danmaku_grouped.json"):
    """从文件读取Bilibili弹幕XML，按整秒分组并按权重排序"""
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    pattern = re.compile(r'<d p="([^"]+)">(.*?)</d>')
    grouped = defaultdict(list)

    for match in pattern.findall(xml_text):
        p_fields = match[0].split(",")
        if len(p_fields) < 8:
            continue
        item = {
            "time": float(p_fields[0]),
            "mode": int(p_fields[1]),
            "font_size": int(p_fields[2]),
            "color": int(p_fields[3]),
            "timestamp": int(p_fields[4]),
            "pool": int(p_fields[5]),
            "uid_hash": p_fields[6],
            "cid": p_fields[7],
            "weight": int(p_fields[8]) if len(p_fields) > 8 else 0,
            "text": match[1]
        }
        key = int(float(p_fields[0]))  # 取整秒作为分组键
        grouped[key].append(item)

    # 每个时间段内按权重排序（高到低）
    for key in grouped:
        grouped[key].sort(key=lambda x: x["weight"], reverse=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"✅ 从 {xml_path} 解析完成，共 {len(grouped)} 个时间段，已保存到 {save_path}")

# 示例调用
parse_bilibili_danmaku_file("hw.xml", "danmaku_grouped.json")
