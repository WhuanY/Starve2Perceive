import json
from PIL import Image
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import logging
import os
import sys
from copy import deepcopy
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
import json_repair

# ================= Configuration =================
# 请替换为你的实际 API Key 和 Base URL
API_KEY = "none" 
BASE_URL = "http://localhost:8000/v1" 
MODEL_NAME = "judge" 

# 数据集路径配置
INPUT_JSON = "/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/Mini-o3-Coldstart-Dataset/Mini-o3-Coldstart.json"
OUTPUT_JSON = "Mini-o3_Coldstart_Active_Refactored.json"

# 图片根目录（根据脚本中的路径推断）
# 原始路径示例: /map-vepfs/haozhe/yhwu/vlmpaper/data/sft/Mini-o3-Coldstart-Dataset/images/minio3_coldstart_1/original_image.jpg
# 注意：代码中会直接使用 JSON 里的绝对路径，如果 JSON 里是相对路径，请修改此处
IMAGE_ROOT = ""

# ================= Logger Setup =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("minio3_refactor.log", encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("minio3_refactor")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= System Prompt =================
# 专门针对 Mini-o3 的 Prompt，强调相对坐标 -> 绝对坐标的转换
REFACTOR_SYSTEM_PROMPT = """You are a strictly logical data converter.
Your task is to refactor a "Grounding" reasoning chain (which uses RELATIVE coordinates 0.0-1.0) into an "Active Visual Search" (Multi-turn) format (which MUST use ABSOLUTE PIXEL coordinates).

**YOUR TASK:**
Convert the [ORIGINAL THOUGHT (With Relative Coords)] into a version that uses ABSOLUTE PIXEL coordinates intermediate thinking process.
1. **Coordinate Conversion (CRITICAL):** - You MUST convert ALL relative coordinates [x0_rel, y0_rel, x1_rel, y1_rel] in the text and bounding boxes to ABSOLUTE PIXELS [x0_abs, y0_abs, x1_abs, y1_abs] based on the provided Width and Height.
   - Formula: Pixel_X = Rel_X * Width, Pixel_Y = Rel_Y * Height.
   - Example: If W=1000, H=1000, a box [0.0, 0.5, 0.5, 1.0] becomes [0, 500, 500, 1000].
   - If the ORIGINAL THOUGHT text integrates relative coordinates in its reasoning path and prepare to zoom in. You should convert the relative coordinates to absolute pixel coordinates. For example, if the ORIGINAL THOUGHT text says "zoom into the bottom-left (0.0 to 0.5)", and Width=1000, Height=1000, rewrite it as "zoom into the bottom-left (pixels 0 to 500)".
2. **Maintain other parts AS IT IS**: The original text and structure should be preserved AS MUCH AS POSSIBLE.
3. If the ORIGINAL THOUGHT contains the name `grounding`, you should rename it to `focus`.
4. ONLY output the new thinking process. DONOT output any other texts.
"""


def get_ice_example():
    ic = "Here are a few examples of how to convert relative coordinates to absolute pixels and rename tool call name:\n"
    ice_example_1 = """
[IMAGE DIMENSIONS]
Width: 2016, Height: 2016
[ORIGINAL THOUGHT (With Relative Coords)]
The user wants me to identify a green object on a piece of furniture within the "Caption samples" section of the image. 

1.  **Identify the target section:** The question specifically refers to the "Caption samples" figure. Looking at the original image, I can see three columns: "Unlabeled data", "Caption samples", and "Language conditioned image samples". The middle column is the one I need to focus on.

2.  **Analyze the content of "Caption samples":** This section contains two examples, each with an image and a corresponding text caption. I need to examine both to find a green object placed on a piece of furniture.

3.  **Examine the top sample:** 
    *   **Image:** It shows a room with a brown/orange piece of furniture (like a dresser or table). On top of this furniture, there is a small, roundish green object. There is also a lamp.
    *   **Caption:** "I can see a white bed where a green ball and a green duck are on it."
    *   **Analysis:** The caption mentions a "green ball" and a "green duck" on a "white bed". While the image does show a green object on furniture, the furniture itself does not look like a white bed. The object is either a ball or a duck. 

4.  **Examine the bottom sample:**
    *   **Image:** It shows a room with a blue chair. 
    *   **Caption:** "i can see a olive green color bus on the floor in front of blue color chair..."
    *   **Analysis:** This caption mentions a green bus, but it's on the floor, not on a piece of furniture. Therefore, this sample does not fit the question's criteria.

5.  **Focus on the top sample:** Since the bottom sample is irrelevant, the answer must come from the top sample. The question asks what object is represented. The image shows a green object on furniture. Let's look at the options:
    *   A. A green duck on a white bed.
    *   B. A green plane on a purple table.
    *   C. A green ball on a blue chair.
    *   D. A green bus on a pink rack.

The caption for the top image mentions both a "green ball" and a "green duck". The image shows one green object that could be either. However, option A says it's on a "white bed", which is inconsistent with the image. The other options are clearly incorrect based on the visual information (it's not a plane, ball on a chair, or bus). 

There seems to be a discrepancy between the image and the caption/options. However, the most plausible scenario is described in the top caption. To confirm the details, I need to zoom in on the top image in the "Caption samples" section. I'll select the bounding box that covers this specific image. This area is in the middle-top of the original image. I will choose the coordinates [0.32, 0.15, 0.58, 0.45] to isolate this image and its caption. 
[CURRENT GROUNDING TARGET (Relative)]
{"bbox_2d": [0.32,0.15,0.58,0.45], "source": "original_image"}
[ABSOLUTE PIXEL COORDINATES]
[645, 186, 1169, 337]
[EXAMPLE OUTPUT]:
The user wants me to identify a green object on a piece of furniture within the "Caption samples" section of the image. 

1.  **Identify the target section:** The question specifically refers to the "Caption samples" figure. Looking at the original image, I can see three columns: "Unlabeled data", "Caption samples", and "Language conditioned image samples". The middle column is the one I need to focus on.

2.  **Analyze the content of "Caption samples":** This section contains two examples, each with an image and a corresponding text caption. I need to examine both to find a green object placed on a piece of furniture.

3.  **Examine the top sample:** * **Image:** It shows a room with a brown/orange piece of furniture (like a dresser or table). On top of this furniture, there is a small, roundish green object. There is also a lamp.
    * **Caption:** "I can see a white bed where a green ball and a green duck are on it."
    * **Analysis:** The caption mentions a "green ball" and a "green duck" on a "white bed". While the image does show a green object on furniture, the furniture itself does not look like a white bed. The object is either a ball or a duck. 

4.  **Examine the bottom sample:**
    * **Image:** It shows a room with a blue chair. 
    * **Caption:** "i can see a olive green color bus on the floor in front of blue color chair..."
    * **Analysis:** This caption mentions a green bus, but it's on the floor, not on a piece of furniture. Therefore, this sample does not fit the question's criteria.

5.  **Focus on the top sample:** Since the bottom sample is irrelevant, the answer must come from the top sample. The question asks what object is represented. The image shows a green object on furniture. Let's look at the options:
    * A. A green duck on a white bed.
    * B. A green plane on a purple table.
    * C. A green ball on a blue chair.
    * D. A green bus on a pink rack.

The caption for the top image mentions both a "green ball" and a "green duck". The image shows one green object that could be either. However, option A says it's on a "white bed", which is inconsistent with the image. The other options are clearly incorrect based on the visual information (it's not a plane, ball on a chair, or bus). 

There seems to be a discrepancy between the image and the caption/options. However, the most plausible scenario is described in the top caption. To confirm the details, I need to zoom in on the top image in the "Caption samples" section. I'll select the bounding box that covers this specific image. This area is in the middle-top of the original image. I will choose the absolute pixel coordinates [645, 186, 1169, 337] to isolate this image and its caption.
"""
    ice_example_2 = """[IMAGE DIMENSIONS]
Width: 2016
Height: 1478
[ORIGINAL THOUGHT (With Relative Coords)]
The user wants to find which bin has the highest value at ℓ = 30. 
1.  First, I need to locate ℓ = 30 on the x-axis of the plot. The x-axis ranges from 0 to 50.
2.  Then, I need to look at the corresponding y-values for all the bins at this x-value. The y-axis is ε_ℓ.
3.  The legend indicates five bins (0 to 4), each with a distinct color and a solid and dashed line.
4.  I will trace a vertical line from ℓ = 30 on the x-axis upwards.
5.  I will then compare the vertical positions of all the lines (solid and dashed) at this point. The highest line corresponds to the highest value.
6.  Visually inspecting the plot, at ℓ = 30, the purple dashed line (bin 4) appears to be at the highest position. The green dashed line (bin 2) also has a peak nearby, but it seems to be slightly to the left of ℓ=30. The red dashed line (bin 3) has a peak to the right of ℓ=30.
7.  To confirm this observation, I will zoom in on the region around ℓ = 30. A good bounding box would be from x-coordinate corresponding to ℓ=25 to ℓ=35. The x-axis is from 0 to 50, so the normalized width would be from 25/50 = 0.5 to 35/50 = 0.7. I will keep the full height to see all the lines. So, I will use the bounding box [0.5, 0.0, 0.7, 1.0]. 

[CURRENT GROUNDING TARGET (Relative)]
{"bbox_2d": [0.5,0.0,0.7,1.0], "source": "original_image"}

[ABSOLUTE PIXEL COORDINATES]
[1008, 0, 1411, 1478]
[EXAMPLE OUTPUT]:
<start_of_example_output>
The user wants to find which bin has the highest value at ℓ = 30. 
1.  First, I need to locate ℓ = 30 on the x-axis of the plot. The x-axis ranges from 0 to 50.
2.  Then, I need to look at the corresponding y-values for all the bins at this x-value. The y-axis is ε_ℓ.
3.  The legend indicates five bins (0 to 4), each with a distinct color and a solid and dashed line.
4.  I will trace a vertical line from ℓ = 30 on the x-axis upwards.
5.  I will then compare the vertical positions of all the lines (solid and dashed) at this point. The highest line corresponds to the highest value.
6.  Visually inspecting the plot, at ℓ = 30, the purple dashed line (bin 4) appears to be at the highest position. The green dashed line (bin 2) also has a peak nearby, but it seems to be slightly to the left of ℓ=30. The red dashed line (bin 3) has a peak to the right of ℓ=30.
7.  To confirm this observation, I will zoom in on the region around ℓ = 30. A good bounding box would be from x-coordinate corresponding to ℓ=25 to ℓ=35. This area is in the middle-center of the original image. I will choose the absolute pixel coordinates [1008, 0, 1411, 1478] to isolate this region.
<end_of_example_output>
"""
    ice_example_3 = """[IMAGE DIMENSIONS]
Width: 1424
Height: 1837
[ORIGINAL THOUGHT (With Relative Coords)]
The user wants to identify which DLE size corresponds to an aspect ratio of approximately 40 on the graph's trend line.
1.  **Analyze the Graph:** The x-axis represents the 'Aspect Ratio α', and the y-axis represents the 'Packing Fraction φ'. The graph contains several data points, each represented by a unique symbol, and a solid black trend line labeled 'RCM'. The legend in the top-left corner maps each symbol to a specific 'D/L' value, which I understand to be the DLE size.
2.  **Locate the Target:** I need to find the point on the x-axis where the 'Aspect Ratio α' is 40.
3.  **Find the Corresponding Data Point:** From the x-axis value of 40, I will trace vertically up to the trend line. I can see a data point that lies very close to the trend line at this x-coordinate.
4.  **Identify the Symbol:** The data point at an aspect ratio of 40 is represented by an upside-down triangle (nabla symbol, ∇).
5.  **Match Symbol to Legend:** Now, I need to determine what DLE size (D/L value) this '∇' symbol represents. To do this, I will need a clear view of the legend.
6.  **Action Plan:** I will use the `grounding` action to zoom in on the legend located in the top-left quadrant of the image. A bounding box of `[0.1, 0.1, 0.4, 0.4]` should be sufficient to isolate and clearly read the legend. 
[CURRENT GROUNDING TARGET (Relative)]
{"bbox_2d": [0.1,0.1,0.4,0.4], "source": "original_image"}
[ABSOLUTE PIXEL COORDINATES]
[142, 183, 596, 735]
[EXAMPLE OUTPUT]:
<start_of_example_output>
The user wants to identify which DLE size corresponds to an aspect ratio of approximately 40 on the graph's trend line.
1.  **Analyze the Graph:** The x-axis represents the 'Aspect Ratio α', and the y-axis represents the 'Packing Fraction φ'. The graph contains several data points, each represented by a unique symbol, and a solid black trend line labeled 'RCM'. The legend in the top-left corner maps each symbol to a specific 'D/L' value, which I understand to be the DLE size.
2.  **Locate the Target:** I need to find the point on the x-axis where the 'Aspect Ratio α' is 40.
3.  **Find the Corresponding Data Point:** From the x-axis value of 40, I will trace vertically up to the trend line. I can see a data point that lies very close to the trend line at this x-coordinate.
4.  **Identify the Symbol:** The data point at an aspect ratio of 40 is represented by an upside-down triangle (nabla symbol, ∇).
5.  **Match Symbol to Legend:** Now, I need to determine what DLE size (D/L value) this '∇' symbol represents. To do this, I will need a clear view of the legend.
6.  **Action Plan:** I will use the `focus` action to zoom in on the legend located in the top-left quadrant of the image. A bounding box of `[142, 183, 570, 695]` should be sufficient to isolate and clearly read the legend. 
<end_of_example_output>
"""
    ice_example_4 = """[IMAGE DIMENSIONS]
Width: 1909
Height: 2016
[ORIGINAL THOUGHT (With Relative Coords)]
The user wants me to identify the best description for the Antlia galaxy shown in Figure c. 
First, I need to locate Figure c in the composite image. The image is divided into four panels labeled a, b, c, and d. Figure c is in the bottom-left corner, labeled "Transition Dwarf Galaxy - Antlia".

Next, I need to carefully observe the morphology of the galaxy in Figure c. The galaxy appears as a collection of stars. I need to determine if it has a specific structure like a spiral, an ellipse, or a ring, or if it's irregular.

Looking at the image, the galaxy in panel c does not have the distinct, swirling arms characteristic of a spiral galaxy (Option D). It also doesn't have the smooth, uniform, and generally featureless appearance of an elliptical galaxy (Option B). There is no prominent ring structure, so it's not a ring galaxy (Option C). 

The distribution of stars appears clumpy and lacks a defined, symmetrical shape. This disorganized and somewhat chaotic structure is the hallmark of an irregular galaxy.

To be absolutely sure, I will zoom in on panel c to examine the details of the galaxy's structure more closely. Panel c is located in the bottom-left quadrant of the original image. I will select the bounding box corresponding to this area, which I estimate to be from (0, 0.5) to (0.5, 1.0). 
[CURRENT GROUNDING TARGET (Relative)]
{"bbox_2d": [0,0.5,0.5,1.0], "source": "original_image"}
[ABSOLUTE PIXEL COORDINATES]
[0, 1008, 954, 2016]
[EXAMPLE OUTPUT]:
<start_of_example_output>
The user wants me to identify the best description for the Antlia galaxy shown in Figure c. 
First, I need to locate Figure c in the composite image. The image is divided into four panels labeled a, b, c, and d. Figure c is in the bottom-left corner, labeled "Transition Dwarf Galaxy - Antlia".

Next, I need to carefully observe the morphology of the galaxy in Figure c. The galaxy appears as a collection of stars. I need to determine if it has a specific structure like a spiral, an ellipse, or a ring, or if it's irregular.

Looking at the image, the galaxy in panel c does not have the distinct, swirling arms characteristic of a spiral galaxy (Option D). It also doesn't have the smooth, uniform, and generally featureless appearance of an elliptical galaxy (Option B). There is no prominent ring structure, so it's not a ring galaxy (Option C). 

The distribution of stars appears clumpy and lacks a defined, symmetrical shape. This disorganized and somewhat chaotic structure is the hallmark of an irregular galaxy.

To be absolutely sure, I will zoom in on panel c to examine the details of the galaxy's structure more closely. Panel c is located in the bottom-left quadrant of the original image. I will select the bounding box corresponding to this area, which I estimate to be from the top-left corner.  
<end_of_example_output>
"""
    examples = [ice_example_1, ice_example_2, ice_example_3, ice_example_4]
    for example in examples:
        ic += example + "\n\n\n"

    ic += "\n\n\n"
    return ic
def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
        print(f"Original data length: {len(data)}") # 7267
        return data

def get_image_size(path):
    with Image.open(path) as img:
        return img.size  # (width, height)
    

from tenacity import retry, stop_after_attempt, wait_exponential
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15)
)
def convert_coords(rel_coords, w, h):
    # rel_coords: [x0, y0, x1, y1] (0-1)
    # 转换为绝对像素坐标 [x1, y1, x2, y2]
    return [
        int(rel_coords[0] * w),
        int(rel_coords[1] * h),
        int(rel_coords[2] * w),
        int(rel_coords[3] * h)
    ]

def rewrite_gpt(tk_orig, g_rel_json_str, w, h, abs_bboxes):
    """
    使用 LLM 局部重构思维链：将相对坐标的逻辑描述转换为绝对像素坐标描述。
    """
    # 构造发送给 LLM 的指令信息
    user_content = f"""[IMAGE DIMENSIONS]
Width: {w},Height: {h}
[ORIGINAL THOUGHT (With Relative Coords)]
{tk_orig}
[CURRENT GROUNDING TARGET (Relative)]
{g_rel_json_str}
[ABSOLUTE PIXEL COORDINATES]
{abs_bboxes}
[YOUR OUTPUT]:
"""

    # 调用 OpenAI API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": REFACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": get_ice_example() + user_content}
        ],
        temperature=0.0, # 保持低随机性，确保计算逻辑严谨
        stop=["<|im_end|>", "<grounding>", "<tool_call>"],
    )
    import random 
    if random.random() < 0.1:
        print(f"[DEBUG] response: {response}\n\n\n", file=open("/map-vepfs/haozhe/yhwu/vlmpaper/tmp/check_lasj/rewrite_gpt_debug.txt", "a"))


    # 提取重构后的思维链内容
    tk_new = response.choices[0].message.content.strip()
    
    # 清理可能存在的 Markdown 标签包裹（如果 LLM 自作聪明加了 ```json 或 <think>）
    tk_new = tk_new.replace("```json", "").replace("```", "").replace("<think>", "").replace("</think>", "").strip()
    
    return tk_new




def postprocess_item(item):
    try:
        w, h = get_image_size(item['images'][0])
    except Exception as e:
        kill
    my_system_prompt="""
You are a helpful assistant.

# Tools
You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"focus","description":"Request a high-resolution local region of the first image and zoom in","parameters":{"type":"object","properties":{"bboxes":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"array","items":{"type":"integer"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to crop, as [x1, y1, x2, y2] in ABSOLUTE PIXEL COORDINATES of the first image."},"description":"A list of bounding boxes to zoom in on. You can request 1-3 bboxes at a turn."}},"required":["bboxes"]}}}
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
    my_user_prompt = "<image>\nThink first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question:\n"
    new_item = deepcopy(item)

    for turn_idx, msg in enumerate(new_item['conversations']):
        if msg['from'] == "system":
           msg['value'] = my_system_prompt
        elif msg['from'] == "human" and turn_idx == 1:
           msg['value'] = my_user_prompt + msg['value'].replace("<image>", "")
        elif msg['from'] == "human" and turn_idx >= 3:
            # only keep
            assert msg['value'].count("<image>") == 1
            msg['value'] = "<image>"
        elif msg['from'] == "gpt":
            if "<grounding>" in msg["value"] and "</grounding>" in msg["value"]:
                g = msg["value"].split("<grounding>")[1].split("</grounding>")[0]
                g_json = json.loads(g)
                rel_bboxes = g_json.get("bbox_2d", [])
                assert len(rel_bboxes) == 4 and isinstance(rel_bboxes[0], (int, float)) and isinstance(rel_bboxes, list), "what0"
                abs_bboxes = convert_coords(rel_bboxes, w, h)
                tc = f'<tool_call>\n{{"name": "focus", "arguments": {{"bboxes": [{abs_bboxes}]}}}}\n</tool_call>'
                tk_orig = msg['value'].split("<think>")[1].split("</think>")[0]
                tk_new = rewrite_gpt(tk_orig, g, w, h, abs_bboxes)
                msg['value'] = f'<think>{tk_new}</think>\n{tc}'
            elif "<answer>" in msg["value"] and "</answer>" in msg["value"]:
                pass 
        else:
           kill # ????
    # return None
    return new_item



def main():
    # 注意：确保其他函数如 load_json, postprocess_item 已在外部定义
    data = load_json("/map-vepfs/haozhe/yhwu/vlmpaper/data/sft/Mini-o3-Coldstart-Dataset/Mini-o3-Coldstart.json")
    
    pre_filtered_data = []
    filter_reason = {}
    
    print("Starting data filtering...")
    for d in tqdm(data):
        if len(d['conversations']) <= 3:
            filter_reason["Filter1: Length <= 3"] = filter_reason.get("Filter1: Length <= 3", 0) + 1
            continue
        
        second_order_grounding = False
        invalid_grounding = False
        invalid_format = False
        
        for msg in d['conversations']:
            if msg['from'] == "gpt":
                if not ("<think>" in msg["value"] and "</think>" in msg["value"]):
                    kill # 保持原样
                if "<grounding>" in msg["value"] and "</grounding>" in msg["value"]:
                    assert "<answer>" not in msg["value"], "what1"
                    tc = msg["value"].split("<grounding>")[1].split("</grounding>")[0]
                    try:
                        tc_dict = json.loads(tc)
                        if "observation_" in tc_dict['source']:
                            second_order_grounding = True
                    except:
                        invalid_grounding = True
                elif "<answer>" in msg["value"] and "</answer>" in msg["value"]:
                    pass
                else:
                    invalid_format = True
            elif msg['from'] == "human":
                pass
            elif msg['from'] == "system":
                pass
            else:
                kill # 保持原样
        
        if invalid_format:
            filter_reason["Filter2: Invalid Format"] = filter_reason.get("Filter2: Invalid Format", 0) + 1
            continue
        if invalid_grounding:
            filter_reason["Filter3: Invalid Grounding"] = filter_reason.get("Filter3: Invalid Grounding", 0) + 1
            continue
        if second_order_grounding:
            filter_reason["Filter4: Second Order Grounding"] = filter_reason.get("Filter4: Second Order Grounding", 0) + 1
            continue
            
        pre_filtered_data.append(d)

    print(f"Filter reason: {filter_reason}")
    print(f"Items to process: {len(pre_filtered_data)}")

    # ================= 多线程处理开始 =================
    new_data = []
    max_workers = 10  # 根据你的 API 限制调整并发数
    
    print(f"Processing with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 dict 映射 future 到原始索引，以防需要保持顺序（可选）
        future_to_item = {executor.submit(postprocess_item, item): item for item in pre_filtered_data}
        
        for future in tqdm(as_completed(future_to_item), total=len(future_to_item)):
            try:
                result = future.result()
                if result is not None:
                    new_data.append(result)
            except Exception as e:
                # 仅打印错误，不停止主流程
                print(f"Generated an exception: {e}")
    # ================= 多线程处理结束 =================

    # 统一存入文件
    with open("new_data_coldstart_wo_compression.json", "w") as f:
        json.dump(new_data, f, indent=4)
    
    print(f"Final saved items: {len(new_data)}")
    print(f"Filter reason: {filter_reason}")

if __name__ == "__main__":
    main()