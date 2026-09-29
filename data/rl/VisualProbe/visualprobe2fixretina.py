import os
import json
import re
import math
from copy import deepcopy
import base64
import numpy as np
import logging
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
# TODO: Update this to the folder containing the 'VisualProbe_train' directory
IMAGE_ROOT = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/" 

# Image Processing Constants
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384  # Added: was missing but referenced in smart_resize
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_VIEW_PIXELS = (28 * 16) ** 2 

# --- Image Utilities ---
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

# --- Parsing Logic ---
def split_question_options(i, question_str):
    # 0. Clean preamble
    question_str = question_str.replace('<image>', '').strip()

    # 1. Try finding split marker
    split_pattern = r"^(.*?)(?:Options:|choices:)\s*(.*)$"
    match = re.match(split_pattern, question_str, re.IGNORECASE | re.DOTALL)
    
    if match:
        question = match.group(1).strip()
        options_part = match.group(2).strip()
        
        # Check patterns for options
        options_vgr = re.findall(r"\([A-Z]\)\s*([^\n]+)", options_part)
        options_std = re.findall(r"[A-Z]:\s*([^\n,]+(?:\n|,|$))", options_part)
        
        if options_vgr:
            options = [opt.strip() for opt in options_vgr]
        elif options_std:
            options = [opt.rstrip(',\n ').strip() for opt in options_std]
        else:
            options = [line.strip() for line in options_part.split('\n') if line.strip()]
    else:
        question = question_str
        options = []

    if question and question[-1] not in ['?', '.']:
        question += '?' 
        
    return question, options

# --- Prompt Construction ---
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
user_prompt="""<image>\nThink first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question:\n"""
# user_prompt="""<image>
# Think carefully before answering. If the current view is not enough (e.g., the image is too large/unclear/complicated)  or if a previous zoom-in was incorrect, call **image_zoom_in** to gather better evidence and do not give <answer> in the same step. 
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
def change_prompt(question, options):
    user_content = user_prompt + question
    if options:
        choices_str = ", ".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
        user_content += f" choices: {choices_str}"

    messages = [
        {'role': 'system', 'content': NEW_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content}
    ]
    return np.array(messages, dtype=object)

# --- Component Conversion ---

def change_extra_info(i, row_dict, compression_ratio=None):
    # 1. Parse Question (using 'problem')
    question_str = row_dict['problem']
    question, options = split_question_options(i, question_str)
    if len(options) == 0:
        options = None
    
    # 2. Handle Answer (using 'solution')
    raw_answer = row_dict['solution']
    answer_list = [raw_answer]

    # 3. Handle Image
    # VisualProbe Format: ['VisualProbe_train/data/...']
    if 'images' not in row_dict or not row_dict['images']:
        raise ValueError(f"Row {i} missing 'images' field or empty images list")
    rel_path = row_dict['images'][0]
    full_path = os.path.join(IMAGE_ROOT, rel_path)
    
    try:
        original_pil_img = Image.open(full_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {full_path}: {e}")
        raise e

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
    pixel_budget_per_image = max_view_pixels

    
    # Get bytes of ORIGINAL image for seed_img (for future cropping)
    original_bytes = pil2bytes(original_pil_img)

    # 4. Construct Dict
    extra_info = {
        'answer': answer_list,
        'index': row_dict.get('doc_id', f"vp_{i}"), 
        'options': options,
        'question': question,
        'seed_img': {
            'bytes': original_bytes, 
            'scale': scale
        },
        'split': 'train',
        'orig_path': rel_path, 
        "pixel_budget_per_image": pixel_budget_per_image
    }
    
    assert pixel_budget_per_image <= MAX_PIXELS, f'[ERROR] {pixel_budget_per_image=} > {MAX_PIXELS}'
    return overview_pil_img, extra_info

def change_reward_model(row_dict, extra_info):
    # VisualProbe example doesn't show bboxes, defaulting to empty list
    gt_bboxes = None
    

    reward_model = {
        'ground_truth': extra_info['answer'],
        'style': 'multiple_choice' if extra_info['options'] else 'free_form',
        'ground_truth_bboxes': gt_bboxes,
        'question': extra_info['question'],
        'path': None 
    }
    return reward_model

def change_row(i, row_dict, compression_ratio=None): # cr = None, fallback to default fix_retina setting
    new_row = {}
    
    # 1. Extra Info & Image Processing
    overview_pil_img, extra_info = change_extra_info(i, row_dict, compression_ratio)
    
    # 2. Convert overview image to bytes (for the 'images' field)
    overview_img_bytes = pil2bytes(overview_pil_img)
    
    # 3. Prompt Construction
    new_row['prompt'] = change_prompt(extra_info['question'], extra_info['options'])
    
    # 4. Images Field
    new_row['images'] = np.array([{'bytes': overview_img_bytes, 'path': None}], dtype=object)
    
    # 5. Metadata
    new_row['ability'] = 'visual_probe' 
    new_row['env_name'] = 'fixretina'
    new_row['data_source'] = row_dict.get('data_source', 'visual_probe') # training set: visual_probe_train
    
    # 6. Reward Model
    new_row['reward_model'] = change_reward_model(row_dict, extra_info)
    
    # 7. Final Extra Info
    new_row['extra_info'] = extra_info
    
    return i, new_row

# --- Multithreaded Execution ---
def process_wrapper(args):
    i, row_dict, compression_ratio = args
    try:
        return change_row(i, row_dict, compression_ratio)
    except Exception as e:
        print(f"Error at index {i} ({row_dict.get('doc_id')}): {e}")
        return i, None

def pixel_filter(results_dict, min_pixels, max_pixels):
    if max_pixels is None:
        max_pixels = float('inf')
    if min_pixels is None:
        min_pixels = 0
    results_dict_filtered = []
    for i, res_dict in enumerate(results_dict):
        assert len(res_dict['images']) == 1
        img_path = os.path.join(IMAGE_ROOT, res_dict['images'][0])
        try:
            # We open to check size, similar to original logic
            with Image.open(img_path) as img:
                if img.size[0] * img.size[1] <= max_pixels and img.size[0] * img.size[1] > min_pixels:
                    results_dict_filtered.append((i, res_dict))
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
    return results_dict_filtered

def main(type='Easy'):
    assert type in ['Easy', 'Medium', 'Hard', 'train']
    import pyarrow as pa
    import pyarrow.parquet as pq

    # input_json_path = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_train/train.json"
    if type == 'train':
        input_json_path = f"/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_{type}/train.json"
    else:
        input_json_path = f"/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_{type}/val.json"

    # Output template configuration
    output_dir = f"/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_{type}/"
    output_file_tmpl = "visualprobe_cr{compression_ratio}_res_{min_pixel}to{max_pixel}_fixretina_ap.parquet"
    output_path_tmpl = output_dir + output_file_tmpl
    
    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    print(f"There are originally {len(data)} records.")
    MIN_PIXELS_VQA = 512*512
    MAX_PIXELS_VQA = None
    results_dict_filtered = pixel_filter(data, min_pixels=MIN_PIXELS_VQA, max_pixels=MAX_PIXELS_VQA)
    print(f"There are {len(results_dict_filtered)} images with more than {MIN_PIXELS_VQA} pixels and less than {MAX_PIXELS_VQA} pixels.")
    
    # --- Iterate Compression Ratios ---
    # COMPRESSION_RATIO = [0.1, 0.125, 0.2, 0.25, 0.5, 1] # Add [0.5, 0.25] etc. if needed
    COMPRESSION_RATIO = [None]
    COMPRESSION_RATIO = [1]
    for cr in COMPRESSION_RATIO:
        if cr is None:
            print("Processing default fix_retina setting...")
        else:
            print(f"Processing compression ratio: {cr}")
        output_parquet_path = output_path_tmpl.format(
            compression_ratio=cr, 
            min_pixel=MIN_PIXELS_VQA, 
            max_pixel="Inf" if MAX_PIXELS_VQA is None else MAX_PIXELS_VQA
        )
        # Run with ThreadPool
        current_data_batch = deepcopy(results_dict_filtered)
        args_list = [(i, row_dict, cr) for i, row_dict in current_data_batch]

        results = []
        with ThreadPoolExecutor(max_workers=100) as executor:
            results = list(tqdm(executor.map(process_wrapper, args_list), total=len(args_list)))

        results.sort(key=lambda x: x[0])
        fixretina_result = [r[1] for r in results if r[1] is not None]
        
        print(f"Successfully processed {len(fixretina_result)} records.")

        if len(fixretina_result) > 0:
            # Smart chunking: split into multiple parquet files, each <= 1.5GB
            MAX_FILE_SIZE_GB = 1.5
            MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 1024 * 1024 * 1024
            
            # Base output path (without extension)
            base_output_path = output_parquet_path.rsplit('.', 1)[0]
            
            # Start with a reasonable batch size
            batch_size = 1000
            min_batch_size = 100  # Minimum batch size to avoid too many small files
            total_records = len(fixretina_result)
            chunk_idx = 0
            start_idx = 0
            
            print(f"Starting to save {total_records} records in chunks (max {MAX_FILE_SIZE_GB}GB per file)...")
            
            while start_idx < total_records:
                end_idx = min(start_idx + batch_size, total_records)
                batch = fixretina_result[start_idx:end_idx]
                
                # Generate chunk filename
                if chunk_idx == 0:
                    chunk_path = output_parquet_path  # First chunk uses original name
                else:
                    chunk_path = f"{base_output_path}_part{chunk_idx}.parquet"
                
                # Write batch to parquet
                table = pa.Table.from_pylist(batch)
                pq.write_table(table, chunk_path)
                
                # Check file size
                file_size = os.path.getsize(chunk_path)
                file_size_gb = file_size / (1024 * 1024 * 1024)
                
                print(f"  Chunk {chunk_idx}: {len(batch)} records, {file_size_gb:.2f}GB -> {chunk_path}")
                
                # If file is too large, reduce batch size and retry this chunk
                if file_size > MAX_FILE_SIZE_BYTES and len(batch) > 1:
                    print(f"    Warning: File exceeds {MAX_FILE_SIZE_GB}GB, reducing batch size and retrying...")
                    os.remove(chunk_path)  # Remove oversized file
                    new_batch_size = max(min_batch_size, int(batch_size * 0.7))  # Reduce by 30%, but not below min
                    if new_batch_size == batch_size and batch_size == min_batch_size:
                        # Already at minimum, but still too large - this shouldn't happen with reasonable data
                        print(f"    Error: Cannot reduce batch size further. Batch of {len(batch)} records exceeds {MAX_FILE_SIZE_GB}GB!")
                        raise ValueError(f"Batch of {len(batch)} records exceeds {MAX_FILE_SIZE_GB}GB limit even at minimum batch size")
                    batch_size = new_batch_size
                    continue  # Retry with smaller batch
                
                # Successfully saved this chunk
                chunk_idx += 1
                start_idx = end_idx
                
                # If batch was small and file is well under limit, try increasing batch size for next chunk
                if file_size < MAX_FILE_SIZE_BYTES * 0.5 and batch_size < 10000:
                    batch_size = min(10000, int(batch_size * 1.2))
            
            print(f"Successfully saved {chunk_idx} chunk(s) to parquet files.")
            if chunk_idx > 1:
                print(f"  Main file: {output_parquet_path}")
                print(f"  Additional chunks: {base_output_path}_part1.parquet ... {base_output_path}_part{chunk_idx-1}.parquet")
                # we know more than one chunk, so it is more elegant to rename the first chunk as ...part_0.parquet
                os.rename(output_parquet_path, f"{base_output_path}_part0.parquet")

        else:
            print("No records to save!")

    

if __name__ == "__main__":
    main(type='train')

