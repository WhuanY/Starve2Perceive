import json 



with open("/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/FixRetina_SFT/sft_training_data/v4/pxr_traj_img_restored.json", "r") as f:
    data = json.load(f)




new_d = []
for d in data:
    if "gqa" in d['index']:
        continue
    elif len(d['conversations']) == 3:
        continue
    else:
        new_d.append(d)
output_path = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/FixRetina_SFT/sft_training_data/v4/pxr_traj_img_restored_no_gqa_no_singlepass.json"
with open(output_path, "w") as f:
    json.dump(new_d, f, ensure_ascii=False, indent=4)