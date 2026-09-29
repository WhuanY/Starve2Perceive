new_system_prompt = """You are a helpful assistant.

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

import duckdb
from copy import deepcopy
from pandas.core.internals.construction import new_block
from tqdm import tqdm
def parquet_to_df_duckdb(con, parquet_path, query=None):
    con = duckdb.connect()
    if query is None:
        query = f"SELECT * FROM '{parquet_path}';"
    result = con.execute(query).fetchdf()  
    return result

"""
Image processing utilities for tool calling.
"""
import base64
import math
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Image processing constants
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO=200

# ---- Added for Fix Retina ----
# 28: Pixel Length for a Patch of QwenVL Tokenizer; 
# 16: Draw inspiration from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" 
# Under this setting and by proper processing, the maximum tokens viewed by qwenvl will not exceed 256. 
# Such setting is heuristic and may not be optimal.

MAX_VIEW_PIXELS = (28 * 16) ** 2 

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    """Encode a PIL Image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, 
    width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, 
    max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    """
    Smart resize image dimensions to fit within pixel constraints.
    
    Args:
        height: Original height
        width: Original width
        factor: Rounding factor for dimensions
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
        
    Returns:
        Tuple of (new_height, new_width)
    """
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    
    return h_bar, w_bar


# ---- Added for Fix Retina ----
def count_tokens(img_input=None, width=None, height=None):
    """
    Count the number of tokens for an image using Qwen-VL tokenizer logic.
    """
    # If width and height are provided, use them directly
    if width is not None and height is not None:
        pass  # Use provided dimensions
    # If img_input is a string (path), load the image
    elif isinstance(img_input, str):
        img = Image.open(img_input)
        width, height = img.size
    # If img_input is a PIL Image
    elif hasattr(img_input, 'size'):  # Check for PIL Image (duck typing)
        width, height = img_input.size
    else:
        raise ValueError(
            "Must provide either:\n"
            "  - img_input as str (file path)\n"
            "  - img_input as PIL.Image.Image\n"
            "  - both width and height parameters"
        )
    
    # Calculate tokens using smart_resize logic
    resized_h, resized_w = smart_resize(height, width)
    image_tokens = (resized_h // IMAGE_FACTOR) * (resized_w // IMAGE_FACTOR)
    
    return image_tokens


def constrain_image_size(img: Image.Image, max_pixels: int = MAX_VIEW_PIXELS) -> Tuple[Image.Image, Tuple[float, float]]:
    """
    Constrain the image size to the maximum pixel budget. 
    
    Returns:
        (resized_img, (scale_x, scale_y)): Tuple of (resized_img, (scale_x, scale_y))
    """
    if max_pixels is None:
        return img, (1.0, 1.0)
        
    w, h = img.size
    
    target_h, target_w = smart_resize(h, w, 
                                      factor=IMAGE_FACTOR, 
                                      min_pixels=MIN_PIXELS, 
                                      max_pixels=max_pixels)
    
    if target_h == h and target_w == w:
        return img, (1.0, 1.0)
    
    resized_img = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    
    scale_x, scale_y = target_w / w, target_h / h
    
    return resized_img, (scale_x, scale_y)

def pil2bytes(pil_image):
    # Ensure the image is in RGB mode
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert the image to bytes
    with BytesIO() as buffer:
        try:
            pil_image.save(buffer, format='jpeg')
            return buffer.getvalue()
        except Exception as e:
            raise ValueError(f"Failed to convert PIL image to bytes: {e}")


new_system_prompt = """You are a helpful assistant.

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
new_user_prompt = """<image>\nThink first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question:\n"""
# new_user_prompt="""<image>
# Think carefully before answering. If the current view is not enough (e.g., the image is too large/unclear/complicated)  or if a previous zoom-in was incorrect, call **focus** to gather better evidence and do not give <answer> in the same step. 
# Keep track of the original question across multiple steps and always make the zoom-in region highly relevant to the question. 
# Only when you are certain about the final answer and no further tool call is needed, then output:
# <answer>...</answer>.
# **Format stricly as either:**
# <think>...</think>
# <tool_call>...</tool_call>
# **OR**
# <think>...</think><answer></answer>
# Here is the question:
# """ # clear prompt from *virl*
def change_prompt(row_dict):
    prompt = row_dict['prompt']
    question = row_dict['extra_info']['question']
    options = row_dict['extra_info']['options']
    abc_mapping = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for msg in prompt:
        if msg['role'] == 'system':
            msg['content'] = new_system_prompt
        elif msg['role'] == 'user':
            msg['content'] = new_user_prompt + question
            if options:
                msg['content'] += '\n'.join(f"{abc_mapping[i]}: {opt}" for i, opt in enumerate(options))
    return prompt

import re

def split_question_options(i, question_str):
    """
    Returns:
        tuple: A tuple containing:
            question (str): The question part of the input string.
            options (list): A list of options in the order they appear.
    """
    # Use regex to extract the question and options part
    # Use DOTALL flag to match newlines
    match = re.match(r"^(.*?)choices:\s*(.*)$", question_str, re.IGNORECASE | re.DOTALL)
    if not match:
        print(f"ERROR!!!! {i} not match")
        raise ValueError("Input string is not in the expected format.")
    
    # Extract question
    question = match.group(1).strip(" ?\n")
    if question[-1] != '?':
        question = question + '?'

    # Extract options
    options_part = match.group(2).strip()
    # Match options that can be separated by commas or newlines
    # Pattern: [A-Z]: followed by content until next [A-Z]: or end of string
    options = re.findall(r"[A-Z]:\s*([^\n,]+(?:\n|,|$))", options_part)
    
    # Clean up options: remove trailing commas, newlines, and whitespace
    options = [opt.rstrip(',\n ').strip() for opt in options]

    
    return question, options

# = test = 
# question_str = "
# question_strs = ["What type of cutting board is described in the region?\nchoices:\nA: a grass green rough pentagonal cutting board\nB: a black glossy circular cutting board\nC: a brightly colored rectangular cutting board\nD: a yellow translucent triangular cutting board\n",
# "Where are papers posted? choices: A: On the fridge, B: On the wall, C: On the cork board, D: The desk"
# ]
# for i,qs in enumerate(question_strs):
#     question, options = split_question_options(i,qs)
#     print(question)
#     print(options)
# kill

# result = parquet_to_df_duckdb(con, path_2)
# results_dict = result.to_dict(orient='records')
# for i, row_dict in enumerate(results_dict):
#     question_str = row_dict['extra_info']['question']
#     question, options = split_question_options(i, question_str)
#     file = open("tmp.txt", "a")
#     print(i, '\n-----------------\n', question, '\n-----------------\n',options, '\n-----------------\n', file=file)
#     row_dict['extra_info']['question'] = question
#     row_dict['extra_info']['options'] = options
# print("YEAAAAA, PASS")
# kill



def change_extra_info(i, row_dict, compression_ratio):
    extra_info = {'answer': ['2014'],
    'index': '0ed7643c-6bbe-44e5-b783-0ec6e31c75cd',
    'options': [],
    'question': 'When was the padding on pants compulsory, 1920, 1960, 2000, or 2014?',
    'seed_img': {'bytes': b'',
        'scale': [1.0, 1.0]
        },
    'split': 'train',
    'answer': ["A"],
    'pixel_budget_per_image': MAX_VIEW_PIXELS
}
    # exchange postiion of seed_img and overview_img
    raw_img_bytes = row_dict['images']
    extra_info['seed_img']['bytes'] = raw_img_bytes[0]['bytes']
    original_pil_img = Image.open(BytesIO(raw_img_bytes[0]['bytes'])).convert("RGB")

    # ----- determine the pixel_budegt_per_image based on the original image size and the compression ratio -----
    max_view_pixels = MAX_VIEW_PIXELS
    if compression_ratio:
        compressed_width = original_pil_img.size[0] * compression_ratio
        compressed_height = original_pil_img.size[1] * compression_ratio
        compressed_max_pixels = compressed_width * compressed_height
        max_view_pixels = min(compressed_max_pixels, MAX_PIXELS) # max_view_pixels is impossible to be larger than original pixels.
        if compression_ratio == 1:
            max_view_pixels = MAX_PIXELS # original setting
    else: # fixretina setting or original setting
        original_width, original_height = original_pil_img.size
        original_pixels = original_width * original_height
        if original_pixels > MAX_VIEW_PIXELS:
            max_view_pixels = MAX_VIEW_PIXELS
        else:
            max_view_pixels = MAX_PIXELS # Fallback to naive think-with-image

    # ----- generate the overview image and calculate the scale -----
    overview_pil_img, scale = constrain_image_size(original_pil_img, max_pixels=max_view_pixels)
    extra_info['seed_img']['scale'] = scale
    pixel_budget_per_image = max_view_pixels
    extra_info['pixel_budget_per_image'] = pixel_budget_per_image # finally, the pixel budget per image is the mapping of the original image size and the compression rate.

    # index, options, questions, split
    extra_info['index'] = row_dict['extra_info']['index']
    question, options = split_question_options(i, row_dict['extra_info']['question'])
    if len(options) == 0:
        options = None
    extra_info['options'] = options
    extra_info['question'] = question
    try:
        no_boxed_answer = row_dict['extra_info']['answer'].split(r'oxed{')[1].split(r'}')[0]
        extra_info['answer'] = [no_boxed_answer]
    except:
        raise ValueError("answer is not boxed")
    return overview_pil_img, extra_info

def change_reward_model(i, row_dict):
    reward_model = {
        'ground_truth': ['2014'], 
        'style': 'multiple_choice',
        'ground_truth_bboxes': [], 
        'question': row_dict['extra_info']['question'],
        'path': None
    }
    try:
        no_boxed_answer = row_dict['reward_model']['ground_truth'].split(r'oxed{')[1].split(r'}')[0]
        reward_model['ground_truth'] = [no_boxed_answer]
    except:
        print(f"{i} no oxed!!!")
        raise ValueError("answer is not boxed")
    try:
        reward_model['ground_truth_bboxes'] = [row_dict['reward_model']['ground_truth_bboxes']]
        # should be None / List[List[float]]
        # 检查是否是**嵌套**列表List[List[float]]
        
    except:
        print(f"{i} no ground_truth_bboxes!!!")
        raise ValueError("ground_truth_bboxes is not present")
    try:
        reward_model['question'] = row_dict['extra_info']['question']
    except:
        print(f"{i} no question!!!")
        raise ValueError("question is not present")

    return reward_model


def change_row(i, row_dict, compression_ratio):
    row_dict['data_source'] = 'virl'
    row_dict['env_name'] = 'fixretina'
    #===extrainfo===
    overview_pil_img, row_dict['extra_info'] = change_extra_info(i, row_dict, compression_ratio)
    overview_img_bytes = pil2bytes(overview_pil_img)
    # ====images====  
    row_dict['images'][0]['bytes'] = overview_img_bytes
    row_dict['images'][0]['path'] = None
    #===rewardmodel===
    row_dict['reward_model'] = change_reward_model(i, row_dict)
    #===prompt===
    row_dict['prompt'] = change_prompt(row_dict)
    return row_dict


def change_parquet(result, compression_ratio=None):
    fixretina_result = [] # [(orig_idx, new_row_dict), ...]
    for orig_idx, row_dict in tqdm(result):
        try:
            new_row_dict = change_row(orig_idx, row_dict, compression_ratio)
            fixretina_result.append((orig_idx, new_row_dict))
        except:
            import traceback
            traceback.print_exc()
            print(f"{orig_idx} error!!!")
            continue
    return fixretina_result

def pixel_filter(results_dict, min_pixels, max_pixels):
    if max_pixels is None:
        max_pixels = float('inf')
    if min_pixels is None:
        min_pixels = MIN_PIXELS
    results_dict_filtered = []
    for i, res_dict in enumerate(results_dict):
        img = Image.open(BytesIO(res_dict['images'][0]['bytes'])).convert("RGB")
        if img.size[0] * img.size[1] <= max_pixels and img.size[0] * img.size[1] > min_pixels:
            results_dict_filtered.append((i, res_dict))
    return results_dict_filtered


if __name__ == "__main__": 
    import pyarrow as pa
    import pyarrow.parquet as pq
    in_path = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/ViRL8k/train_general_filter.parquet"
    out_base_dir = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/ViRL8k"
    out_file_tmpl = "virl8k_cr{compression_ratio}_res_{min_pixel}to{max_pixel}_fixretina_ap.parquet" # NOTE: cp: clear prompt, ap: ambiguous prompt
    out_path_tmpl = out_base_dir + "/" + out_file_tmpl
    MAX_PIXELS_VQA = None
    MIN_PIXELS_VQA = 512*512
    # COMPRESSION_RATIO = [0.1, 0.125, 0.2, 0.25, 1] # (1/10, 1/8, 1/4, 1/2, 1)
    # COMPRESSION_RATIO = [0.5]
    # COMPRESSION_RATIO=[1]
    COMPRESSION_RATIO=[None]
    con = duckdb.connect()
    result = parquet_to_df_duckdb(con, in_path)
    results_dict = result.to_dict(orient='records')
    # high_res_filter
    results_dict_high_res = []
    results_dict_filtered = pixel_filter(results_dict, MIN_PIXELS_VQA, MAX_PIXELS_VQA)
    print(f"There are {len(results_dict_filtered)} images with more than {MIN_PIXELS_VQA} pixels and less than {MAX_PIXELS_VQA} pixels.")
    # create new parquets
    for cr in COMPRESSION_RATIO:
        out_path = out_path_tmpl.format(compression_ratio=cr, min_pixel=MIN_PIXELS_VQA, max_pixel=MAX_PIXELS_VQA)
        results_dict_filtered_copied = deepcopy(results_dict_filtered)
        fixretina_result = change_parquet(results_dict_filtered_copied, cr)
        print(f"There are {len(fixretina_result)} records converted to fixretina dataset format for compression ratio {cr}.")
        fixretina_result_dict = [row_dict for _, row_dict in fixretina_result]
        table = pa.Table.from_pylist(fixretina_result_dict)
        pq.write_table(table, out_path)
        print(f"Created {out_path} with compression ratio {cr}")
    print("DONE!!!")

