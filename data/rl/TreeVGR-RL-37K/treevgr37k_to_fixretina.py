import os
import re
import math
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Dict, Any
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq    
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# --- Constants from virl8k_to_fixretina.py ---
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
# Fix Retina specific
MAX_VIEW_PIXELS = (28 * 16) ** 2 

# --- Image Helper Functions (Copied & Adapted) ---
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, 
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
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

def constrain_image_size(img: Image.Image, max_pixels: int = MAX_VIEW_PIXELS) -> Tuple[Image.Image, Tuple[float, float]]:
    if max_pixels is None:
        return img, (1.0, 1.0)
    w, h = img.size
    target_h, target_w = smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=max_pixels)
    
    if target_h == h and target_w == w:
        return img, (1.0, 1.0)
    
    resized_img = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    scale_x, scale_y = target_w / w, target_h / h
    return resized_img, (scale_x, scale_y)

def pil2bytes(pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    with BytesIO() as buffer:
        pil_image.save(buffer, format='jpeg')
        return buffer.getvalue()

def split_question_options(i, question_str):
    """
    Parses question string to extract question text and options (if any).
    Supports:
      1. Standard "choices:" format (A: ...)
      2. TreeVGR "Options:" format ((A) ...)
      3. Free-form (returns empty options)
    """
    # 0. Clean preamble like <image>
    question_str = question_str.replace('<image>', '').strip()

    # 1. Try finding split marker (Options: or choices:)
    # We use a pattern that matches "Options:" or "choices:" case-insensitive, followed by content
    split_pattern = r"^(.*?)(?:Options:|choices:)\s*(.*)$"
    match = re.match(split_pattern, question_str, re.IGNORECASE | re.DOTALL)
    
    if match:
        # --- Multiple Choice Case ---
        question = match.group(1).strip()
        options_part = match.group(2).strip()
        
        # Pattern 1: (A) Option text (TreeVGR style)
        # Matches "(A)" or "(A) " at start of line or after newline
        options_vgr = re.findall(r"\([A-Z]\)\s*([^\n]+)", options_part)
        
        # Pattern 2: A: Option text (Original style)
        options_std = re.findall(r"[A-Z]:\s*([^\n,]+(?:\n|,|$))", options_part)
        
        if options_vgr:
            options = [opt.strip() for opt in options_vgr]
        elif options_std:
            options = [opt.rstrip(',\n ').strip() for opt in options_std]
        else:
            # Fallback: Split by newline if we found "Options:" but regex failed
            options = [line.strip() for line in options_part.split('\n') if line.strip()]

    else:
        # --- Free Form Case ---
        question = question_str
        options = []

    # Ensure question ends with ?
    if question and question[-1] not in ['?', '.']:
        question += '?' # Optional normalization
        
    return question, options

# --- TEST EXAMPLES ---
print("--- Test 1: TreeVGR MC ---")
q_mc = "<image>\nUsing visual data, determine the total number of motor visible. Options:\n(A) 5\n(B) 7\n(C) 9"
q, opts = split_question_options(1, q_mc)
print(f"Q: {q}\nOpts: {opts}")

print("\n--- Test 2: TreeVGR Free Form ---")
q_ff = "<image>\nWhat is the color of the container?"
q, opts = split_question_options(2, q_ff)
print(f"Q: {q}\nOpts: {opts}")

NEW_SYSTEM_PROMPT = """You are a helpful assistant.

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

def change_prompt(row_dict, question, options):
    """
    Constructs the messages list (System + User).
    Note: TreeVGR raw data doesn't have a 'prompt' field like previous datasets might.
    We construct it from scratch using the question.
    """
    # Construct User Query
    user_content = ""
    # Append instructions
    user_content = "<image>\nThink first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question: " + question
    if options:
        # Reconstruct options string for the prompt
        choices_str = ", ".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
        user_content += f" choices: {choices_str}"
    messages = [
        {'role': 'system', 'content': NEW_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content}
    ]
    return np.array(messages, dtype=object)

# --- TEST EXAMPLE ---
print("\n--- Test 3: Prompt Construction ---")
q = "Where is the cat?"
opts = ["Left", "Right"]
msgs = change_prompt({}, q, opts)
print(msgs[1]['content'][:10000])

IMAGE_ROOT = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/TreeVGR-RL-37K/"

def change_extra_info(i, row_dict, compression_ratio=None):
    # 1. Parse Question/Options
    question_str = row_dict['problem']
    question, options = split_question_options(i, question_str)
    if not options:
        options = None
    
    # 2. Handle Answer
    raw_answer = row_dict['answer']
    # If standard MC (A, B, C), keep as is. If free text, keep as is.
    answer_list = [raw_answer]

    # 3. Handle Image
    # TreeVGR path example: 'images/2241.jpg' inside a numpy array
    rel_path = row_dict['images'][0] 
    full_path = os.path.join(IMAGE_ROOT, rel_path)
    
    try:
        original_pil_img = Image.open(full_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {full_path}: {e}")
        raise e

    # Create Seed Image (Resize for Overview)
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
    
    # Convert original image to bytes
    original_img_bytes = pil2bytes(original_pil_img)
    
    # 4. Construct Dict
    pixel_budget_per_image = max_view_pixels
    extra_info = {
        'answer': answer_list,
        'index': f"treevgr_{i}", # Synthesize index
        'options': options,
        'question': question,
        'seed_img': {
            'bytes': original_img_bytes,
            'scale': scale
        },
        'split': 'train',
        'pixel_budget_per_image': pixel_budget_per_image,
        'orig_path': rel_path 
    }
    
    return overview_pil_img, extra_info

# --- TEST EXAMPLE ---
# Mocking a row
print("\n--- Test 4: Extra Info ---")
mock_row = {
    'problem': '<image>\nIs this valid?',
    'answer': 'Yes',
    'images': np.array(['images/dummy.jpg'])
}
# img, info = change_extra_info(999, mock_row)
# print(info['question'], info['answer'], info['options'])

def change_reward_model(i, row_dict, extra_info):
    # Flatten bboxes from target_instances
    # Source: [{'bbox': array([x1, y1, x2, y2]), ...}, ...]
    target_inst = row_dict.get('target_instances', [])
    gt_bboxes = []
    
    if isinstance(target_inst, (list, np.ndarray)):
        for inst in target_inst:
            if isinstance(inst, dict) and 'bbox' in inst:
                # Convert numpy array to list
                bbox = inst['bbox']
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                gt_bboxes.append(bbox)
    if not gt_bboxes:
        gt_bboxes = None
    reward_model = {
        'ground_truth': extra_info['answer'],
        'style': 'multiple_choice' if extra_info['options'] else 'free_form',
        'ground_truth_bboxes': gt_bboxes, # None / List[List[float]]
        'question': extra_info['question'],
        'path': None 
    }
    return reward_model

# --- TEST EXAMPLE ---
print("\n--- Test 5: Reward Model ---")
mock_target = np.array([{'bbox': np.array([10., 20., 30., 40.])}, {'bbox': np.array([50., 60., 70., 80.])}])
mock_row_rm = {'target_instances': mock_target}
mock_info = {'answer': ['A'], 'options': ['A','B'], 'question': 'Q?'}

rm = change_reward_model(1, mock_row_rm, mock_info)
print(rm)

def change_row(i, row_dict, compression_ratio=None):
    """
    Transforms a TreeVGR row to FixRetina format.
    """
    new_row = {}
    
    # 1. Extra Info & Image Processing
    overview_pil_img, extra_info = change_extra_info(i, row_dict, compression_ratio)
    
    # Convert image to bytes
    overview_img_bytes = pil2bytes(overview_pil_img)

    # 2. Prompt
    new_row['prompt'] = change_prompt(row_dict, extra_info['question'], extra_info['options'])
    
    # 3. Images Field
    # Format: array([{'bytes': b'...', 'path': None}], dtype=object)
    new_row['images'] = np.array([{'bytes': overview_img_bytes, 'path': None}], dtype=object)
    
    # 4. Metadata
    new_row['ability'] = 'vqa' # General ability tag
    new_row['env_name'] = 'fixretina'
    new_row['data_source'] = 'treevgr'
    
    # 5. Reward Model
    new_row['reward_model'] = change_reward_model(i, row_dict, extra_info)
    
    # 6. Extra Info Finalize
    new_row['extra_info'] = extra_info
    
    return new_row

def process_wrapper(args):
    """
    包装函数，用于在多线程中处理单行数据并捕获异常
    """
    i, row_dict, compression_ratio = args
    try:
        return change_row(i, row_dict, compression_ratio)
    except Exception as e:
        print(f"{i} Error: {e}")
        return None

if __name__ == "__main__":
    con = duckdb.connect()
    in_path="/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/TreeVGR-RL-37K/treevgr_cs1to3.parquet"
    # out_path = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/TreeVGR-RL-37K/treevgr_cs1to3_fixretina.parquet"
    out_path="/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/TreeVGR-RL-37K/treevgr_cs1to3_valina.parquet"
    MIN_PIXELS_VQA=512*512
    # COMPRESSION_RATIO=[None]
    COMPRESSION_RATIO=[1]

    query = f"SELECT * FROM '{in_path}';"
    result = con.execute(query).fetchdf()  
    d_result = result.to_dict(orient='records')
    fixretina_result = []
    d_result_high_res = []
    from PIL import Image
    for i, dr in enumerate(d_result):
        pil_img = Image.open(os.path.join(IMAGE_ROOT, dr['images'][0]))
        assert len(dr['images']) == 1
        if pil_img.size[0] * pil_img.size[1] > MIN_PIXELS_VQA:
            d_result_high_res.append((i, dr))

    print(f"There are {len(d_result_high_res)} images with more than {MIN_PIXELS_VQA} pixels.")
    for compression_ratio in COMPRESSION_RATIO:
        args_list = [(i, row, compression_ratio) for i, row in d_result_high_res]

        fixretina_result = []
        print(f"Starting multithreaded processing with {len(args_list)} records...")
        with ThreadPoolExecutor(max_workers=100) as executor:
            results = list(tqdm(executor.map(process_wrapper, args_list), total=len(args_list)))

        fixretina_result = [r for r in results if r is not None]

        print(f"Done! Successfully processed {len(fixretina_result)}/{len(d_result)} records.")
        # save results 

        table = pa.Table.from_pylist(fixretina_result)
        pq.write_table(table, out_path)


        print("LOAD!!!")
        # load results to test if right
        table = pq.read_table(out_path)
        print("DONE!!!")


