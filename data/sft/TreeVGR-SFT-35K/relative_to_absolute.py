import json
import os

new_sys_prompt = """You are a helpful assistant.

# Tools
You are provided with the function signature within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "focus",
        "description": "Request a high-resolution local region of the first image and zoom in",
        "parameters": {
            "type": "object",
            "properties": {
                "bboxes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "The bounding box of the region to crop, as [x1, y1, x2, y2] in ABSOLUTE PIXEL COORDINATES of the first image."
                    },
                    "description": "A list of bounding boxes to zoom in on. You can request 1-3 bboxes at a turn."
                }
            },
            "required": ["bboxes"]
        }
    }
}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "focus", "arguments": {"bboxes": [[10, 20, 100, 200]]}}  
</tool_call>
"""
# 输入和输出文件路径
input_file = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K/treevgr_rewrited_llamafacotry.json"
output_file = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/TreeVGR-SFT-35K/treevgr_rewrited_llamafacotry_abs_img_path.json"

# 获取输入文件所在目录作为基准路径
base_dir = os.path.dirname(input_file)

# 读取JSON文件
print(f"Reading {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Processing {len(data)} items...")

for item in data:
    # 转换images字段中的相对路径为绝对路径
    if 'images' in item and isinstance(item['images'], list):
        item['images'] = [os.path.join(base_dir, img) if not os.path.isabs(img) else img 
                          for img in item['images']]
    # 更改system prompt
    if 'conversations' in item:
        conv = item['conversations']
        for c in conv:
            if c['from'] == 'system':
                c['value'] = new_sys_prompt


# 保存到输出文件
print(f"Writing to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Done!")

