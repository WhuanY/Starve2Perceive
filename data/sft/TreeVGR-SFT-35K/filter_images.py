import json

path = "/home/zzhengbo/vlmpaper/data/TreeVGR-SFT-35K/TreeVGR-SFT-35K_rewrited.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    assert len(d['images']) == 1

img_path_set = set()
for d in data:
    img_path_set.add(d['images'][0])

print(len(img_path_set))
# save it
with open("img_path_set.json", "w", encoding="utf-8") as f:
    json.dump(list(img_path_set), f, ensure_ascii=False, indent=4)

print("ok")
