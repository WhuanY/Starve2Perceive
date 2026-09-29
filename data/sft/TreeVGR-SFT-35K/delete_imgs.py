import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 加载 img_path_set
with open("img_path_set.json", "r", encoding="utf-8") as f:
    img_path_set = json.load(f)

# 构建绝对路径集合
img_base_dir = "/home/zzhengbo/vlmpaper/data/TreeVGR-SFT-35K/llava_next_raw_format"
img_path_set_abs = {os.path.join(img_base_dir, img_path) for img_path in img_path_set}

# 删除文件的函数
def delete_file(file_path):
    os.remove(file_path)

# 使用多线程并行删除
with ThreadPoolExecutor(max_workers=64) as executor:
    for root, _, files in os.walk(img_base_dir):
        abs_files = {os.path.join(root, file) for file in files}
        to_delete = abs_files - img_path_set_abs  # 找到需要删除的文件
        list(tqdm(executor.map(delete_file, to_delete), desc="Deleting images", total=len(to_delete)))

print("ok")